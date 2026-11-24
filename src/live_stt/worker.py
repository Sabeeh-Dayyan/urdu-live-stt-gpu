"""Channel worker: ingest → ring buffer → VAD → streaming policy → ASR requests.

One OS process per channel. The pipeline core (`ChannelPipeline`) is independent of PyAV and the
bus transport so it can be driven by synthetic audio in tests.
"""

from __future__ import annotations

import signal
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from live_stt import metrics as M
from live_stt.bus.base import Bus, f32_to_pcm16
from live_stt.ingest.clock import ClockEvent
from live_stt.ingest.ringbuffer import OverwrittenError, RingBuffer
from live_stt.ingest.source import AudioChunk, AudioSource, SourceEvent
from live_stt.logging import get_logger
from live_stt.settings import ChannelConfig, Settings, resolve_path
from live_stt.streaming.segment import SegmentCut, make_segment_id
from live_stt.streaming.segment_policy import SegmentPolicy
from live_stt.vad.silero_onnx import FRAME, FrameAccumulator

log = get_logger(__name__)


@dataclass
class PendingRefine:
    cut: SegmentCut
    segment_id: str
    ready_at: int  # absolute sample index needed for post-context


class ChannelPipeline:
    def __init__(self, settings: Settings, channel: ChannelConfig, bus: Bus, vad,
                 policy=None, wall=time.time) -> None:
        self.s = settings
        self.ch = channel
        self.bus = bus
        self.vad = vad
        self.wall = wall
        self.sr = settings.ingest.sample_rate
        self.ring = RingBuffer(settings.ingest.ring_buffer_s, self.sr)
        self.policy = policy or SegmentPolicy(settings.vad, settings.streaming, self.sr, FRAME)
        self.acc = FrameAccumulator(FRAME)
        self.warm = False
        self.held: list[SegmentCut] = []  # cuts decided before the clock was warm
        self.pending_refine: deque[PendingRefine] = deque()
        self.ctx = int(settings.asr.refine_context_s * self.sr)
        self.refine = settings.asr.refine_enabled
        self._speech_frames: deque[bool] = deque(maxlen=int(60 * self.sr / FRAME))
        self._speech_count = 0
        self.la = settings.streaming.mode == "local_agreement"
        self.step = int(settings.streaming.step_s * self.sr)
        self.step_scale = 1.0
        self.last_sent = 0  # local_agreement: ring index up to which audio was streamed
        self.session_start = 0  # ring index where the current connection's audio begins
        self.stats = {"segments": 0, "partials": 0, "refine": 0, "lost": 0, "events": 0}

    # ---------------------------------------------------------------- events
    def on_source_event(self, ev: SourceEvent) -> None:
        if ev.type == "stream_down":
            self._flush()  # emits held cuts first: never lose speech decided before warmup
            self.vad.reset()
            self.acc.reset()
            self.policy.reset()
            self.warm = False
            M.STREAM_UP.labels(self.ch.id).set(0)
        elif ev.type == "stream_up":
            self.session_start = self.ring.end
            M.STREAM_UP.labels(self.ch.id).set(1)
        self._system_event(ev.type, ev.utc, ev.details)

    def on_clock_event(self, ev: ClockEvent, clock_warm: bool) -> None:
        if ev.type == "startup_correction":
            self.ring.shift_utc(ev.details["shift_s"], from_idx=self.session_start)
        elif ev.type == "discontinuity":
            # Audio before/after the jump has different UTC bases: never span a segment across it.
            self._flush()
        if ev.type in ("discontinuity", "drift", "startup_correction"):
            self._system_event(ev.type, ev.utc, ev.details)

    def _system_event(self, etype: str, utc: float, details: dict) -> None:
        self.stats["events"] += 1
        M.STREAM_EVENTS.labels(self.ch.id, etype).inc()
        self.bus.publish(self.s.bus.stream_system, {
            "type": etype, "channel_id": self.ch.id, "at_utc": utc, "details": details,
        })
        log.info("stream_event", type=etype, **{k: v for k, v in details.items() if k != "url"})

    # ----------------------------------------------------------------- audio
    def on_chunk(self, chunk: AudioChunk, clock_warm: bool = True) -> None:
        for ev in chunk.events:
            self.on_clock_event(ev, clock_warm)
        x = chunk.samples
        if len(x) == 0:
            return
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        first = self.ring.write(x, chunk.utc)
        M.AUDIO_SECONDS.labels(self.ch.id).inc(len(x) / self.sr)
        if clock_warm and not self.warm:  # release held cuts before newer ones (keeps order)
            self.warm = True
            self._release_held()
        for idx, frame in self.acc.push(x, first):
            prob = self.vad(frame)
            rms = float(np.sqrt(np.mean(frame * frame)))
            speech = prob >= self.s.vad.threshold
            if len(self._speech_frames) == self._speech_frames.maxlen:
                self._speech_count -= self._speech_frames[0]
            self._speech_frames.append(speech)
            self._speech_count += speech
            for cut in self.policy.process(idx, prob, rms):
                self._handle_cut(cut, clock_warm)
        if self.la and clock_warm and self.policy.triggered:
            self._stream_step()
        self._release_refines()
        if self._speech_frames:
            M.VAD_SPEECH_RATIO.labels(self.ch.id).set(self._speech_count / len(self._speech_frames))

    def _handle_cut(self, cut: SegmentCut, clock_warm: bool) -> None:
        if not clock_warm:
            if cut.kind == "final":
                self.held.append(cut)
            return
        self._emit(cut)

    def _release_held(self) -> None:
        held, self.held = self.held, []
        for cut in held:
            self._emit(cut)

    def _flush(self) -> None:
        self._release_held()
        for cut in self.policy.flush(self.ring.end):
            self._handle_cut(cut, True)
        self._release_refines(force=True)

    # --------------------------------------------------------------- publish
    def _stream_step(self) -> None:
        if self.last_sent < self.policy.seg_start:
            self.last_sent = max(self.policy.seg_start, self.ring.start)
        if self.ring.end - self.last_sent >= int(self.step * self.step_scale):
            self._send_delta(self.last_sent, self.ring.end, kind="stream")

    def _send_delta(self, start: int, end: int, kind: str, cut: SegmentCut | None = None,
                    seg_id: str | None = None) -> None:
        start = max(start, self.ring.start)
        audio = self.ring.read(start, end) if end > start else np.zeros(0, np.float32)
        key_utc = self.ring.index_to_utc(self.policy.segment_key if cut is None else cut.segment_key)
        msg = {
            "type": "asr_request", "channel_id": self.ch.id, "kind": kind, "pass": "fast",
            "segment_key": f"{self.ch.id}:{int(round(key_utc * 1000))}",
            "delta_utc": self.ring.index_to_utc(start), "duration_s": len(audio) / self.sr,
            "language": self.ch.language, "sample_rate": self.sr, "t_enqueued": self.wall(),
            "pcm": f32_to_pcm16(audio),
        }
        if cut is not None:
            msg.update({
                "segment_id": seg_id, "seq": cut.seq, "cut_reason": cut.reason,
                "utc_start": self.ring.index_to_utc(cut.start),
                "duration_s": (cut.end - cut.start) / self.sr,
                "speech_end_utc": self.ring.index_to_utc(cut.speech_end),
            })
        self.bus.publish(self.s.bus.stream_fast, msg)
        self.last_sent = max(self.last_sent, end)

    def _emit(self, cut: SegmentCut) -> None:
        if self.la and cut.kind == "final":
            utc0 = self.ring.index_to_utc(cut.start)
            seg_id = make_segment_id(self.ch.id, utc0)
            try:
                self._send_delta(max(self.last_sent, cut.start), max(cut.end, self.last_sent),
                                 "stream_end", cut, seg_id)
            except (OverwrittenError, IndexError) as e:
                self.stats["lost"] += 1
                log.error("segment_audio_unavailable", error=str(e))
                return
            self.stats["segments"] += 1
            M.SEGMENTS_EMITTED.labels(self.ch.id, cut.reason).inc()
            if self.refine:
                self.pending_refine.append(PendingRefine(cut, seg_id, cut.end + self.ctx))
            return
        if self.la:
            return  # partials come from the streaming session
        try:
            audio = self.ring.read(cut.start, cut.end)
        except (OverwrittenError, IndexError) as e:
            self.stats["lost"] += 1
            log.error("segment_audio_unavailable", error=str(e), start=cut.start, end=cut.end)
            return
        utc0 = self.ring.index_to_utc(cut.start)
        key_utc = self.ring.index_to_utc(cut.segment_key)
        seg_id = make_segment_id(self.ch.id, utc0) if cut.kind == "final" else \
            f"{self.ch.id}:p{int(round(key_utc * 1000))}"
        msg = {
            "type": "asr_request", "channel_id": self.ch.id, "segment_id": seg_id,
            "segment_key": f"{self.ch.id}:{int(round(key_utc * 1000))}",
            "seq": cut.seq, "kind": cut.kind, "pass": "fast", "cut_reason": cut.reason,
            "utc_start": utc0, "duration_s": len(audio) / self.sr,
            "speech_start_utc": self.ring.index_to_utc(cut.speech_start),
            "speech_end_utc": self.ring.index_to_utc(cut.speech_end),
            "ctx_pre_s": 0.0, "ctx_post_s": 0.0,
            "language": self.ch.language, "sample_rate": self.sr,
            "t_enqueued": self.wall(), "pcm": f32_to_pcm16(audio),
        }
        self.bus.publish(self.s.bus.stream_fast, msg)
        if cut.kind == "final":
            self.stats["segments"] += 1
            M.SEGMENTS_EMITTED.labels(self.ch.id, cut.reason).inc()
            if self.refine:
                self.pending_refine.append(PendingRefine(cut, seg_id, cut.end + self.ctx))
        else:
            self.stats["partials"] += 1

    def _release_refines(self, force: bool = False) -> None:
        while self.pending_refine and (force or self.pending_refine[0].ready_at <= self.ring.end):
            p = self.pending_refine.popleft()
            start = max(p.cut.start - self.ctx, self.ring.start)
            end = min(p.ready_at, self.ring.end)
            try:
                audio = self.ring.read(start, end)
            except (OverwrittenError, IndexError) as e:
                log.warning("refine_audio_unavailable", error=str(e))
                continue
            utc0 = self.ring.index_to_utc(start)
            self.bus.publish(self.s.bus.stream_refine, {
                "type": "asr_request", "channel_id": self.ch.id, "segment_id": p.segment_id,
                "seq": p.cut.seq, "kind": "final", "pass": "refine", "cut_reason": p.cut.reason,
                "utc_start": utc0, "duration_s": len(audio) / self.sr,
                "speech_start_utc": self.ring.index_to_utc(p.cut.speech_start),
                "speech_end_utc": self.ring.index_to_utc(p.cut.speech_end),
                "ctx_pre_s": (p.cut.start - start) / self.sr,
                "ctx_post_s": max(0, end - p.cut.end) / self.sr,
                "language": self.ch.language, "sample_rate": self.sr,
                "t_enqueued": self.wall(), "pcm": f32_to_pcm16(audio),
            })
            self.stats["refine"] += 1


def resolve_auto_mode(settings: Settings, bus: Bus) -> str:
    """`auto`: local_agreement iff the ASR backend is CUDA and measured FAST RTF < threshold."""
    try:
        info = bus.get_value("stt:asr:rtf_fast") or {}
    except Exception:  # noqa: BLE001
        info = {}
    if info.get("device") == "cuda" and float(info.get("rtf", 9)) < settings.streaming.auto_rtf_threshold:
        return "local_agreement"
    return "segment"


def make_relay(settings: Settings, channel_id: str):
    """HLS relay for the UI video player, or None when disabled."""
    r = settings.relay
    if not r.enabled:
        return None
    import re

    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", channel_id) or channel_id in (".", ".."):
        log.warning("relay_disabled_unsafe_channel_id", channel_id=channel_id)
        return None
    from live_stt.ingest.relay import HlsRelay

    return HlsRelay(resolve_path(r.dir) / channel_id, r.window_s, r.segment_s,
                    archive_hours=r.archive_hours)


def run_worker(settings: Settings, channel_id: str, bus: Bus, max_connects: int | None = None,
               stop_after_s: float | None = None) -> ChannelPipeline:
    from live_stt.vad.silero_onnx import SileroVAD

    ch = settings.channel(channel_id)
    if settings.streaming.mode == "auto":
        settings.streaming.mode = resolve_auto_mode(settings, bus)
        log.info("streaming_mode", mode=settings.streaming.mode)
    vad = SileroVAD(str(resolve_path(settings.vad.model_path)))
    pipe = ChannelPipeline(settings, ch, bus, vad)
    src = AudioSource(
        ch.url, ch.audio_track, settings.ingest, settings.clock, max_connects=max_connects,
        on_packet=lambda n: (M.PACKETS_IN.labels(ch.id).inc(), M.BYTES_IN.labels(ch.id).inc(n)),
        relay=make_relay(settings, ch.id),
    )

    def _term(*_):
        src.stop()

    try:
        signal.signal(signal.SIGTERM, _term)
        signal.signal(signal.SIGINT, _term)
    except ValueError:  # not in main thread (tests)
        pass

    t_end = time.time() + stop_after_s if stop_after_s else None
    last_health = 0.0
    for item in src:
        if isinstance(item, SourceEvent):
            pipe.on_source_event(item)
        else:
            warm = src.clock.warm if src.clock else True
            pipe.on_chunk(item, warm)
            if src.clock:
                M.DRIFT_MS.labels(ch.id).set(src.clock.drift_ms)
        now = time.time()
        if now - last_health > 5:
            last_health = now
            try:
                if pipe.la:
                    ctl = bus.get_value("stt:control:step_scale") or {}
                    pipe.step_scale = float(ctl.get("scale", 1.0))
                bus.set_value(f"stt:health:worker:{ch.id}", {
                    "t": now, **pipe.stats, "decode_errors": src.decode_errors,
                    "drift_ms": round(src.clock.drift_ms, 1) if src.clock else 0,
                    "connects": src.connects,
                }, ttl_s=30)
            except Exception:  # noqa: BLE001
                pass
        if t_end and now > t_end:
            src.stop()
    pipe._flush()
    log.info("worker_exit", **pipe.stats)
    return pipe
