"""ASR service: bus → scheduler → backend → guards → transcript events.

Threads:
* reader: pulls FAST/REFINE requests from the bus into the scheduler while it has space
  (backpressure stays in the durable bus, not in RAM).
* main: inference loop (one backend call at a time per device).

Delivery: at-least-once. A request is acked only after its event is published. A job that makes
inference raise is retried individually; after ``MAX_ATTEMPTS`` it is published as an ``error``
event and acked (poison message protection) so one bad segment cannot wedge the service.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from dataclasses import dataclass

from live_stt import metrics as M
from live_stt.asr.backends.base import ASRBackend, AudioJob, DecodeOptions
from live_stt.asr.guards import Guards
from live_stt.asr.prompts import build_hotwords, build_prompt, load_glossary
from live_stt.asr.scheduler import FAST, REFINE, Job, Scheduler
from live_stt.asr.streaming_sessions import STREAM_KINDS, StreamingSessions, rounds, words_payload
from live_stt.bus.base import Bus, pcm16_to_f32
from live_stt.logging import get_logger
from live_stt.settings import Settings
from live_stt.text.confidence import mean_confidence
from live_stt.text.urdu_normalize import normalize

log = get_logger(__name__)
GROUP = "asr"
MAX_ATTEMPTS = 3
EVENT_TYPE = {FAST: "final", REFINE: "refined"}


@dataclass
class ChannelCtx:
    glossary: list[str]
    hotwords: str | None
    last_text: str = ""


class AsrService:
    def __init__(self, settings: Settings, bus: Bus, backend: ASRBackend,
                 consumer: str | None = None) -> None:
        self.s = settings
        self.bus = bus
        self.backend = backend
        self.consumer = consumer or f"asr-{socket.gethostname()}-{settings.asr.device_index}"
        a = settings.asr
        self.sched = Scheduler(
            fast_max=a.fast_queue_max, refine_max=a.refine_queue_max,
            refine_max_lag_s=a.refine_max_lag_s, max_batch=a.max_batch,
            max_wait_s=a.max_wait_ms / 1000, batched=a.batched, pack_s=a.pack_s,
        )
        self.guards = Guards(settings.guards)
        self.channels: dict[str, ChannelCtx] = {}
        self.attempts: dict[str, int] = {}
        self._stop = threading.Event()
        self.processed = 0
        self.streams = {FAST: settings.bus.stream_fast, REFINE: settings.bus.stream_refine}
        st = settings.streaming
        self.la = StreamingSessions(backend, self.guards, trim_s=max(st.max_buffer_s - 5, 5),
                                    max_buffer_s=st.max_buffer_s,
                                    prompt_chars=settings.asr.prompt_max_chars)
        self._rtf_ema: float | None = None

    def stop(self) -> None:
        self._stop.set()

    # -------------------------------------------------------------- channels
    def _ctx(self, channel_id: str) -> ChannelCtx:
        ctx = self.channels.get(channel_id)
        if ctx is None:
            gl_path = None
            try:
                gl_path = self.s.channel(channel_id).glossary_path
            except KeyError:
                pass
            gl = load_glossary(gl_path)
            ctx = ChannelCtx(gl, build_hotwords(gl) if self.s.asr.use_hotwords else None)
            self.channels[channel_id] = ctx
        return ctx

    # ---------------------------------------------------------------- reader
    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                got = 0
                space = self.sched.space(FAST) if self.s.asr.serve_fast else 0
                if space:
                    for mid, msg in self.bus.consume(self.streams[FAST], GROUP, self.consumer,
                                                     count=min(space, 32), block_ms=100):
                        self._intake(mid, FAST, msg)
                        got += 1
                if self.s.asr.refine_enabled and self.sched.space(REFINE):
                    for mid, msg in self.bus.consume(self.streams[REFINE], GROUP, self.consumer,
                                                     count=min(self.sched.space(REFINE), 8)):
                        self._intake(mid, REFINE, msg)
                        got += 1
                if not got and not space:
                    time.sleep(0.05)
            except Exception as e:  # noqa: BLE001 - never let the reader die
                log.error("reader_error", error=str(e))
                time.sleep(1.0)

    def _claim_loop(self) -> None:
        """Periodically take over requests stranded on dead consumers (other hosts/names)."""
        claim = getattr(self.bus, "claim_stale", None)
        if claim is None:
            return
        while not self._stop.wait(30):
            for p in (FAST, REFINE):
                if (p == FAST and not self.s.asr.serve_fast) or (
                        p == REFINE and not self.s.asr.refine_enabled):
                    continue
                try:
                    if self.sched.space(p):
                        for mid, msg in claim(self.streams[p], GROUP, self.consumer,
                                              min_idle_ms=120_000, count=self.sched.space(p)):
                            self._intake(mid, p, msg)
                except Exception as e:  # noqa: BLE001
                    log.warning("claim_error", error=str(e))

    REQUIRED = ("channel_id", "utc_start")

    def _intake(self, mid: str, pass_: str, msg: dict) -> None:
        """Validate a request; invalid ones are acked and logged (never crash-loop on them)."""
        try:
            kind = msg.get("kind")
            need = ("channel_id", "delta_utc") if kind == "stream" else self.REQUIRED
            for k in need:
                if msg.get(k) is None:
                    raise ValueError(f"missing {k}")
            float(msg.get("utc_start", msg.get("delta_utc")))
            if not isinstance(msg.get("pcm", b""), (bytes, bytearray)):
                raise ValueError("pcm must be bytes")
            if len(msg.get("pcm", b"")) > 2 * 16_000 * 120:
                raise ValueError("audio longer than 120 s")
            job = self._job(mid, pass_, msg)
        except (ValueError, TypeError, AttributeError) as e:
            log.error("invalid_request_dropped", error=str(e), msg_id=mid)
            M.GUARD_DROPS.labels(pass_, "invalid_request").inc()
            self.bus.ack(self.streams[pass_], GROUP, [mid])
            return
        self.sched.push(job)

    @staticmethod
    def _job(mid: str, pass_: str, msg: dict) -> Job:
        return Job(
            msg_id=mid, pass_=pass_,
            duration_s=float(msg.get("duration_s", 0.0)),
            t_enqueued=float(msg.get("t_enqueued", time.time())),
            segment_key=str(msg.get("segment_key", ""))
            if pass_ == FAST and msg.get("kind") in ("partial", "final") else "",
            is_partial=msg.get("kind") == "partial",
            payload=msg,
        )

    # ------------------------------------------------------------------ main
    def warmup(self) -> None:
        if not self.s.asr.warmup or not hasattr(self.backend, "warmup"):
            return
        models = {self.s.asr.fast.model} if self.s.asr.serve_fast else set()
        if self.s.asr.refine_enabled:
            models.add(self.s.asr.refine.model)
        for m in models:
            dt = self.backend.warmup(m)
            log.info("model_warm", model=m, warmup_s=round(dt, 2))

    def run(self, health_every_s: float = 5.0) -> None:
        threads = [threading.Thread(target=self._reader, name="asr-reader", daemon=True),
                   threading.Thread(target=self._claim_loop, name="asr-claim", daemon=True)]
        for t in threads:
            t.start()
        last_health = 0.0
        while not self._stop.is_set():
            self.step(wait_s=0.1)
            now = time.time()
            if now - last_health >= health_every_s:
                last_health = now
                self._health()

    def step(self, wait_s: float = 0.0) -> int:
        """One scheduling decision. Returns number of jobs processed (incl. skipped)."""
        self.sched.wait(wait_s)
        d = self.sched.next()
        for job, reason in d.skipped:
            self._skip(job, reason)
        for p in (FAST, REFINE):
            M.QUEUE_DEPTH.labels(p).set(self.sched.depth(p))
            M.QUEUE_LAG.labels(p).set(self.sched.lag_s(p))
        lag = self.sched.lag_s(FAST)
        M.FAST_LAG_ALERT.set(1 if lag > self.s.asr.fast_lag_alert_s else 0)
        if not d.jobs:
            return len(d.skipped)
        try:
            self._process(d.pass_, d.jobs)
        except Exception as e:  # noqa: BLE001 - e.g. bus down beyond retries, bad payload
            log.error("process_failed", error=f"{type(e).__name__}: {e}", jobs=len(d.jobs))
            for j in d.jobs:
                n = self.attempts.get(j.msg_id, 0) + 1
                self.attempts[j.msg_id] = n
                if n >= MAX_ATTEMPTS:
                    try:
                        self._publish_error(j, str(e))
                    except Exception:  # noqa: BLE001 - bus still down: leave it pending
                        pass
                    self.attempts.pop(j.msg_id, None)
                # otherwise it stays un-acked in the bus and is redelivered after restart/claim
            time.sleep(0.5)
        return len(d.jobs) + len(d.skipped)

    def _decode_opts(self, pass_: str) -> DecodeOptions:
        dc = self.s.asr.fast if pass_ == FAST else self.s.asr.refine
        return DecodeOptions(model=dc.model, beam_size=dc.beam_size, best_of=dc.best_of,
                             temperature=dc.temperature)

    def _audio_job(self, job: Job) -> AudioJob:
        msg = job.payload
        ctx = self._ctx(msg["channel_id"])
        prompt = build_prompt(ctx.glossary, ctx.last_text, self.s.asr.prompt_max_chars)
        return AudioJob(
            job_id=job.msg_id, audio=pcm16_to_f32(msg.get("pcm", b"")),
            language=msg.get("language") or "ur", initial_prompt=prompt,
            hotwords=ctx.hotwords, meta=msg,
        )

    def _process(self, pass_: str, jobs: list[Job]) -> None:
        stream = [j for j in jobs if (j.payload or {}).get("kind") in STREAM_KINDS]
        if stream:
            self._process_stream(stream)
            jobs = [j for j in jobs if j not in stream]
            if not jobs:
                return
        opts = self._decode_opts(pass_)
        ajobs = [self._audio_job(j) for j in jobs]
        t0 = time.time()
        try:
            results = self.backend.transcribe(ajobs, opts)
        except Exception as e:  # noqa: BLE001
            log.error("inference_error", error=str(e), batch=len(jobs), pass_=pass_)
            if len(jobs) > 1:  # isolate the bad job
                for j in jobs:
                    self._process(pass_, [j])
                return
            j = jobs[0]
            n = self.attempts.get(j.msg_id, 0) + 1
            self.attempts[j.msg_id] = n
            if n >= MAX_ATTEMPTS:
                self._publish_error(j, str(e))
                self.attempts.pop(j.msg_id, None)
            else:
                self.sched.push(j)  # retry later
            return
        t1 = time.time()
        M.BATCH_SIZE.labels(pass_).observe(len(jobs))
        M.INFER_SECONDS.labels(pass_).observe(t1 - t0)
        for j, aj, res in zip(jobs, ajobs, results):
            self._emit(pass_, j, aj, res, t0, t1)

    def _process_stream(self, jobs: list[Job]) -> None:
        """local_agreement mode: rounds of ≤1 job per channel, one backend call per round."""
        opts = self._decode_opts(FAST)
        gl = {}
        for j in jobs:
            ch = j.payload["channel_id"]
            gl[ch] = build_prompt(self._ctx(ch).glossary, "", self.s.asr.prompt_max_chars // 2) or ""
        for rnd in rounds(jobs):
            t0 = time.time()
            try:
                outs = self.la.process_round(rnd, opts, gl)
            except Exception as e:  # noqa: BLE001
                log.error("stream_inference_error", error=str(e))
                for j in rnd:
                    self._publish_error(j, str(e))
                continue
            t1 = time.time()
            M.INFER_SECONDS.labels(FAST).observe(t1 - t0)
            M.BATCH_SIZE.labels(FAST).observe(len(rnd))
            for j, kind, words, extra in outs:
                m = j.payload
                text = " ".join(w.text.strip() for w in words if w.text.strip())
                now = time.time()
                if kind == "final":
                    speech_end = float(m.get("speech_end_utc", now))
                    start, end = float(m["utc_start"]), float(m["utc_start"]) + float(m["duration_s"])
                    M.LAT_END_FINAL.labels(m["channel_id"]).observe(max(0.0, now - speech_end))
                else:
                    speech_end = float(m["delta_utc"]) + float(m.get("duration_s", 0.0))
                    start = words[0].start if words else float(m["delta_utc"])
                    end = words[-1].end if words else speech_end
                    M.LAT_SPEECH_PARTIAL.labels(m["channel_id"]).observe(max(0.0, now - speech_end))
                audio_s = max(extra.get("audio_s", 0.0), 1e-6)
                rtf = extra.get("infer_s", 0.0) / max(float(self.s.streaming.step_s), 1e-6)
                self._publish({
                    "type": kind, "channel_id": m["channel_id"],
                    "segment_id": m.get("segment_id"), "segment_key": m.get("segment_key"),
                    "seq": m.get("seq", 0), "pass": FAST, "model": opts.model,
                    "start_utc": start, "end_utc": end, "text_raw": text,
                    "text_norm": normalize(text), "words": words_payload(words),
                    "avg_logprob": None, "no_speech_prob": None, "compression_ratio": None,
                    "flags": extra.get("flags", []) + ["local_agreement"],
                    "timings": {"t_enqueued": m.get("t_enqueued"), "t_infer_start": t0,
                                "t_infer_end": t1, "t_published": now,
                                "infer_s": round(extra.get("infer_s", 0.0), 4),
                                "rtf_step": round(rtf, 4), "buffer_s": round(audio_s, 2),
                                "speech_end_utc": speech_end,
                                "latency_s": round(now - speech_end, 3)},
                })
                if kind == "final" and text:
                    self._ctx(m["channel_id"]).last_text = text
                self.bus.ack(self.streams[FAST], GROUP, [j.msg_id])
                self.processed += 1

    def _emit(self, pass_, job: Job, aj: AudioJob, res, t0: float, t1: float) -> None:
        msg = job.payload
        dur = aj.duration_s
        ctx_pre = float(msg.get("ctx_pre_s", 0.0))
        ctx_post = float(msg.get("ctx_post_s", 0.0))
        g = self.guards.apply(res, dur, ctx_pre, ctx_post)
        for r in g.drops:
            M.GUARD_DROPS.labels(pass_, r).inc()
        if pass_ == REFINE and not g.text and not g.drops:
            # Nothing usable (e.g. every word fell into the context margins of a tiny segment)
            # and no hallucination verdict: keep the FAST text rather than blanking it.
            self._skip(job, "refine_empty")
            self.attempts.pop(job.msg_id, None)
            self.processed += 1
            return
        rtf = res.infer_s / max(dur, 1e-6)
        M.RTF.labels(pass_).observe(rtf)
        if pass_ == FAST and not job.is_partial:
            self._rtf_ema = rtf if self._rtf_ema is None else 0.9 * self._rtf_ema + 0.1 * rtf
        utc0 = float(msg["utc_start"])
        etype = "partial" if job.is_partial else EVENT_TYPE[pass_]
        now = time.time()
        speech_end = float(msg.get("speech_end_utc", utc0 + dur))
        ev = {
            "type": etype,
            "channel_id": msg["channel_id"],
            "segment_id": msg["segment_id"],
            "segment_key": msg.get("segment_key"),
            "seq": msg.get("seq", 0),
            "pass": pass_,
            "model": res.model,
            "start_utc": utc0 + ctx_pre,
            "end_utc": utc0 + dur - ctx_post,
            "text_raw": g.text,
            "text_norm": normalize(g.text),
            "words": [
                {"w": w.word.strip(), "s": round(utc0 + w.start, 3), "e": round(utc0 + w.end, 3),
                 "p": round(w.probability, 3)}
                for w in g.words
            ],
            "avg_logprob": g.avg_logprob,
            "no_speech_prob": g.no_speech_prob,
            "compression_ratio": g.compression_ratio,
            "flags": g.flags,
            "language": res.language,
            "timings": {
                "t_enqueued": msg.get("t_enqueued"),
                "t_infer_start": t0, "t_infer_end": t1, "t_published": now,
                "infer_s": round(res.infer_s, 4), "rtf": round(rtf, 4),
                "audio_s": round(dur, 3), "speech_end_utc": speech_end,
                "latency_s": round(now - speech_end, 3),
                "cut": msg.get("cut_reason"),
            },
        }
        ch = msg["channel_id"]
        conf = mean_confidence(ev["words"])
        if conf is not None and etype in ("final", "refined"):
            M.SEG_CONFIDENCE.labels(ch, etype).observe(conf)
        if etype == "final":
            M.LAT_END_FINAL.labels(ch).observe(max(0.0, now - speech_end))
            if g.text:
                self._ctx(ch).last_text = g.text
        elif etype == "refined":
            M.LAT_END_REFINED.labels(ch).observe(max(0.0, now - speech_end))
        elif etype == "partial":
            M.LAT_SPEECH_PARTIAL.labels(ch).observe(max(0.0, now - speech_end))
        self._publish(ev)
        self.bus.ack(self.streams[pass_], GROUP, [job.msg_id])
        self.attempts.pop(job.msg_id, None)
        self.processed += 1

    def _publish(self, ev: dict) -> None:
        self.bus.publish(self.s.bus.stream_transcripts, ev)
        try:
            self.bus.broadcast(self.s.bus.pubsub_live, ev)
        except Exception as e:  # noqa: BLE001 - live fan-out is best effort
            log.warning("broadcast_failed", error=str(e))

    def _skip(self, job: Job, reason: str) -> None:
        msg = job.payload or {}
        if job.pass_ == REFINE:
            M.REFINE_SKIPPED.inc()
            self._publish({
                "type": "refine_skipped", "channel_id": msg.get("channel_id"),
                "segment_id": msg.get("segment_id"), "seq": msg.get("seq", 0),
                "pass": REFINE, "start_utc": msg.get("utc_start"),
                "flags": [f"refine_skipped:{reason}"],
                "timings": {"t_enqueued": msg.get("t_enqueued"), "t_published": time.time()},
            })
        self.bus.ack(self.streams[job.pass_], GROUP, [job.msg_id])

    def _publish_error(self, job: Job, error: str) -> None:
        msg = job.payload or {}
        log.error("job_failed_permanently", segment_id=msg.get("segment_id"), error=error)
        self._publish({
            "type": "error", "channel_id": msg.get("channel_id"),
            "segment_id": msg.get("segment_id"), "pass": job.pass_,
            "start_utc": msg.get("utc_start"), "flags": ["asr_error"], "error": error[:500],
        })
        self.bus.ack(self.streams[job.pass_], GROUP, [job.msg_id])

    def _health(self) -> None:
        try:
            lag = self.sched.lag_s(FAST)
            # local_agreement backpressure: ask workers to step less often while FAST lags.
            scale = 2.0 if lag > self.s.asr.fast_lag_alert_s else 1.0
            self.bus.set_value("stt:control:step_scale", {"scale": scale, "lag_s": lag}, ttl_s=30)
            if self._rtf_ema is not None:
                self.bus.set_value("stt:asr:rtf_fast", {"rtf": self._rtf_ema,
                                                        "device": self.backend.device}, ttl_s=3600)
            self.bus.set_value(f"stt:health:asr:{self.consumer}", {
                "pid": os.getpid(), "t": time.time(), "processed": self.processed,
                "fast_depth": self.sched.depth(FAST), "refine_depth": self.sched.depth(REFINE),
                "fast_lag_s": round(self.sched.lag_s(FAST), 2), "device": self.backend.device,
            }, ttl_s=30)
        except Exception as e:  # noqa: BLE001
            log.warning("health_write_failed", error=str(e))
