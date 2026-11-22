"""ASR-service side of the `local_agreement` policy: per-channel LocalAgreement streamers.

Workers in local_agreement mode send ``kind="stream"`` jobs (new audio since the last step) and a
``kind="stream_end"`` job when the segment policy closes a segment. Jobs are processed in rounds
with at most one job per channel, so one backend call transcribes every active channel's buffer
(cross-channel micro-batching on GPU).
"""

from __future__ import annotations

from live_stt.asr.backends.base import ASRBackend, AudioJob, DecodeOptions
from live_stt.asr.guards import Guards
from live_stt.bus.base import pcm16_to_f32
from live_stt.streaming.local_agreement import LocalAgreementStreamer, TWord

STREAM_KINDS = ("stream", "stream_end")


def rounds(jobs: list) -> list[list]:
    """Split jobs into ordered rounds with at most one job per channel each."""
    out: list[list] = []
    level: dict[str, int] = {}
    for j in jobs:
        ch = j.payload["channel_id"]
        k = level.get(ch, -1) + 1
        level[ch] = k
        while len(out) <= k:
            out.append([])
        out[k].append(j)
    return out


class StreamingSessions:
    def __init__(self, backend: ASRBackend, guards: Guards, trim_s: float, max_buffer_s: float,
                 prompt_chars: int = 200) -> None:
        self.backend = backend
        self.guards = guards
        self.trim_s = trim_s
        self.max_buffer_s = max_buffer_s
        self.prompt_chars = prompt_chars
        self.sessions: dict[str, LocalAgreementStreamer] = {}

    def session(self, ch: str) -> LocalAgreementStreamer:
        if ch not in self.sessions:
            self.sessions[ch] = LocalAgreementStreamer(self.trim_s, self.max_buffer_s,
                                                       self.prompt_chars)
        return self.sessions[ch]

    def process_round(self, jobs: list, opts: DecodeOptions, glossary_prompt: dict[str, str]):
        """Returns list of (job, event_kind, words, extra) with event_kind partial|final."""
        for j in jobs:
            m = j.payload
            pcm = m.get("pcm") or b""
            if pcm:
                self.session(m["channel_id"]).append(pcm16_to_f32(pcm), float(m["delta_utc"]))
        active = [j for j in jobs if self.session(j.payload["channel_id"]).buffer_s > 0.3]
        ajobs = []
        for j in active:
            ch = j.payload["channel_id"]
            s = self.session(ch)
            prompt = " ".join(x for x in (glossary_prompt.get(ch, ""), s.prompt()) if x)
            ajobs.append(AudioJob(j.msg_id, s.audio, language=j.payload.get("language") or "ur",
                                  initial_prompt=prompt[-self.prompt_chars:] or None))
        results = self.backend.transcribe(ajobs, opts) if ajobs else []
        by_id = {}
        for j, aj, res in zip(active, ajobs, results):
            g = self.guards.apply(res, aj.duration_s)
            by_id[j.msg_id] = (res, g)
        out = []
        for j in jobs:
            m = j.payload
            s = self.session(m["channel_id"])
            res_g = by_id.get(j.msg_id)
            if res_g:
                res, g = res_g
                step = s.update([(w.start, w.end, w.word, w.probability) for w in g.words])
                flags = g.flags
                infer_s = res.infer_s
            else:
                step, flags, infer_s = None, [], 0.0
            if m.get("kind") == "stream_end":
                cut = float(m["utc_start"]) + float(m["duration_s"])
                words = s.end_segment(cut)
                out.append((j, "final", words, {"flags": flags, "infer_s": infer_s,
                                                "audio_s": s.buffer_s}))
            else:
                words = list(s.run_committed) + (step.partial if step else [])
                out.append((j, "partial", words, {"flags": flags, "infer_s": infer_s,
                                                  "committed": len(s.run_committed),
                                                  "audio_s": s.buffer_s}))
        return out


def words_payload(words: list[TWord]) -> list[dict]:
    return [{"w": w.text.strip(), "s": round(w.start, 3), "e": round(w.end, 3),
             "p": round(w.p, 3)} for w in words]
