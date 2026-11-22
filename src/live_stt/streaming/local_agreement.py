"""`local_agreement` streaming policy — LocalAgreement-2 (Macháček, Dabre & Bojar, 2023,
"Turning Whisper into Real-Time Transcription System", the `whisper_streaming` algorithm).

Every ``step_s`` the uncommitted audio buffer (plus committed text as prompt) is re-transcribed.
The longest common word prefix of the last two hypotheses is **committed** (stable); the rest is
emitted as a partial. The buffer is trimmed at the last committed word end once it exceeds
``trim_s``, and hard-capped at ``max_buffer_s``. All timestamps are absolute UTC, so trimming
never shifts word times.

Runs inside the ASR service (streaming state lives with the consumer of a channel; with Kafka the
partition key = channel_id gives the needed affinity). Pure logic: ``transcribe`` is injected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from live_stt.text.urdu_normalize import normalize

SR = 16_000


@dataclass
class TWord:
    start: float  # absolute UTC
    end: float
    text: str
    p: float = 1.0

    @property
    def key(self) -> str:
        return normalize(self.text)


def common_prefix(a: list[TWord], b: list[TWord]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x.key != y.key or not x.key:
            break
        n += 1
    return n


class HypothesisBuffer:
    """LocalAgreement-2 bookkeeping (mirrors whisper_streaming's HypothesisBuffer)."""

    def __init__(self) -> None:
        self.committed_in_buffer: list[TWord] = []
        self.buffer: list[TWord] = []  # previous hypothesis' uncommitted tail
        self.new: list[TWord] = []
        self.last_committed_time = 0.0

    def insert(self, words: list[TWord]) -> None:
        new = [w for w in words if w.start > self.last_committed_time - 0.1]
        # Drop words re-recognised at the buffer start that duplicate the committed tail.
        if new and self.committed_in_buffer and abs(new[0].start - self.last_committed_time) < 1:
            cn = len(self.committed_in_buffer)
            for i in range(min(cn, len(new), 5), 0, -1):
                tail = [w.key for w in self.committed_in_buffer[-i:]]
                head = [w.key for w in new[:i]]
                if tail == head:
                    new = new[i:]
                    break
        self.new = new

    def flush(self) -> list[TWord]:
        n = common_prefix(self.new, self.buffer)
        commit = self.new[:n]
        if commit:
            self.last_committed_time = commit[-1].end
        self.buffer = self.new[n:]
        self.new = []
        self.committed_in_buffer.extend(commit)
        return commit

    def pop_committed(self, t: float) -> None:
        self.committed_in_buffer = [w for w in self.committed_in_buffer if w.end > t]

    def uncommitted(self) -> list[TWord]:
        return list(self.buffer)


@dataclass
class StepResult:
    committed: list[TWord] = field(default_factory=list)  # newly committed this step
    partial: list[TWord] = field(default_factory=list)  # current uncommitted hypothesis


class LocalAgreementStreamer:
    def __init__(self, trim_s: float = 10.0, max_buffer_s: float = 15.0,
                 prompt_chars: int = 200) -> None:
        self.trim_s = trim_s
        self.max_buffer_s = max(max_buffer_s, trim_s)
        self.prompt_chars = prompt_chars
        self.reset()

    def reset(self) -> None:
        self.audio = np.zeros(0, np.float32)
        self.buffer_utc: float | None = None
        self.hyp = HypothesisBuffer()
        self.run_committed: list[TWord] = []  # committed since the last final (current segment)
        self.context_text = ""  # committed text before the buffer (prompt)

    @property
    def buffer_s(self) -> float:
        return len(self.audio) / SR

    def append(self, samples: np.ndarray, utc: float) -> None:
        x = np.asarray(samples, np.float32).reshape(-1)
        if len(x) == 0:
            return
        if self.buffer_utc is None or len(self.audio) == 0:
            self.buffer_utc = utc
            self.audio = x.copy()
            if self.hyp.last_committed_time < utc - 30:
                self.hyp.last_committed_time = utc - 0.1
            return
        expected = self.buffer_utc + len(self.audio) / SR
        gap = utc - expected
        if 0.02 < gap < 2.0:  # small gap (packet loss): pad silence to keep timing exact
            self.audio = np.concatenate([self.audio, np.zeros(int(gap * SR), np.float32)])
        elif gap >= 2.0 or gap < -0.5:  # discontinuity: restart the buffer here
            self.audio = np.zeros(0, np.float32)
            self.buffer_utc = utc
        self.audio = np.concatenate([self.audio, x])

    def prompt(self) -> str:
        return self.context_text[-self.prompt_chars:].strip()

    def update(self, rel_words: list[tuple[float, float, str, float]]) -> StepResult:
        """Feed a hypothesis for the current buffer (word times relative to buffer start)."""
        if self.buffer_utc is None:
            return StepResult()
        words = [TWord(self.buffer_utc + s, self.buffer_utc + e, t, p)
                 for s, e, t, p in rel_words if t.strip()]
        self.hyp.insert(words)
        commit = self.hyp.flush()
        self.run_committed.extend(commit)
        self._trim()
        return StepResult(commit, self.hyp.uncommitted())

    def _trim(self) -> None:
        if self.buffer_s <= self.trim_s:
            return
        cut_t = None
        if self.hyp.committed_in_buffer:
            cut_t = self.hyp.committed_in_buffer[-1].end
        if cut_t is None or cut_t <= self.buffer_utc:
            if self.buffer_s <= self.max_buffer_s:
                return
            # Nothing committed for max_buffer_s: hard cap (drop oldest audio).
            cut_t = self.buffer_utc + self.buffer_s - self.trim_s
        self._drop_until(cut_t)

    def _drop_until(self, t: float) -> None:
        n = int(round((t - self.buffer_utc) * SR))
        n = max(0, min(n, len(self.audio)))
        if n == 0:
            return
        dropped = [w for w in self.hyp.committed_in_buffer if w.end <= t]
        if dropped:
            self.context_text = (self.context_text + " " +
                                 " ".join(w.text.strip() for w in dropped)).strip()
        self.audio = self.audio[n:]
        self.buffer_utc += n / SR
        self.hyp.pop_committed(self.buffer_utc)
        self.hyp.last_committed_time = max(self.hyp.last_committed_time, self.buffer_utc - 0.1)

    def end_segment(self, cut_utc: float, final_words: list[TWord] | None = None) -> list[TWord]:
        """Close the current segment at ``cut_utc``.

        Returns the segment's words: everything committed so far plus the uncommitted hypothesis
        words whose midpoint is before the cut (end of speech ⇒ no more agreement possible).
        Audio after ``cut_utc`` stays buffered for the next segment.
        """
        tail = final_words if final_words is not None else self.hyp.uncommitted()
        out = [w for w in self.run_committed if (w.start + w.end) / 2 <= cut_utc + 0.05]
        keep_committed = [w for w in self.run_committed if (w.start + w.end) / 2 > cut_utc + 0.05]
        seen_end = out[-1].end if out else -1e18
        for w in tail:
            if (w.start + w.end) / 2 <= cut_utc + 0.05 and w.start >= seen_end - 0.05:
                out.append(w)
                seen_end = w.end
        if out:
            self.context_text = (self.context_text + " " +
                                 " ".join(w.text.strip() for w in out)).strip()[-1000:]
        self.run_committed = keep_committed
        if self.buffer_utc is not None:
            self.hyp.last_committed_time = max(self.hyp.last_committed_time, cut_utc)
            self.hyp.buffer = [w for w in self.hyp.buffer if w.start >= cut_utc]
            self.hyp.committed_in_buffer = [w for w in self.hyp.committed_in_buffer
                                            if w.end > cut_utc]
            self._drop_until(min(cut_utc, self.buffer_utc + self.buffer_s))
        return out
