"""Deterministic fake ASR: each 'word' is a tone burst whose frequency encodes its id."""

from __future__ import annotations

import numpy as np

from live_stt.asr.backends.base import AsrResult, ASRBackend, SegmentResult, WordResult

SR = 16_000


def word_audio(ids, word_s=0.3, gap_s=0.12, amp=0.3):
    parts = []
    for i in ids:
        t = np.arange(int(word_s * SR)) / SR
        parts.append((amp * np.sin(2 * np.pi * (300 + 40 * i) * t)).astype(np.float32))
        parts.append(np.zeros(int(gap_s * SR), np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def detect_words(audio):
    a = np.abs(np.asarray(audio, np.float32))
    win = int(0.01 * SR)
    if len(a) < win:
        return []
    env = np.convolve(a, np.ones(win) / win, mode="same") > 0.05
    out, i, n = [], 0, len(env)
    while i < n:
        if env[i]:
            j = i
            while j < n and env[j]:
                j += 1
            if j - i >= int(0.12 * SR):
                seg = np.asarray(audio[i:j], np.float64)
                spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=1 << 16))
                f = np.argmax(spec) * SR / (1 << 16)
                wid = int(round((f - 300) / 40))
                out.append((i / SR, j / SR, f" w{wid}"))
            i = j
        else:
            i += 1
    return out


class FakeASR(ASRBackend):
    name = "fake"

    def __init__(self, fail_on=None, infer_s=0.01):
        self.calls = []
        self.fail_on = fail_on  # callable(job) -> bool
        self.infer_s = infer_s

    def transcribe(self, batch, opts):
        self.calls.append((opts.model, len(batch)))
        out = []
        for j in batch:
            if self.fail_on and self.fail_on(j):
                raise RuntimeError("boom")
            words = detect_words(j.audio)
            ws = [WordResult(t, s, e, 0.9) for s, e, t in words]
            text = "".join(w.word for w in ws)
            segs = [SegmentResult(text, ws[0].start, ws[-1].end, -0.2, 0.05, 1.2, ws)] if ws else []
            out.append(AsrResult(j.job_id, segs, "ur", self.infer_s, opts.model))
        return out
