"""Fixed-capacity float32 audio ring buffer with sample-accurate UTC mapping.

Samples are addressed by an **absolute** index (total samples ever written, int). Each write
carries the UTC of its first sample; a new anchor ``(index, utc)`` is stored only when that UTC
deviates from the continuation of the current anchor by more than ``anchor_tolerance_s``
(re-anchor, discontinuity, drift slew). Any absolute index still in (or slightly before) the
buffer maps back to UTC via the last anchor at or before it.
"""

from __future__ import annotations

import bisect

import numpy as np


class OverwrittenError(IndexError):
    """Requested samples have already been overwritten (consumer too slow)."""


class RingBuffer:
    def __init__(
        self, capacity_s: float, sample_rate: int = 16_000, anchor_tolerance_s: float = 0.001
    ) -> None:
        self.sr = sample_rate
        self.capacity = int(round(capacity_s * sample_rate))
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self.total = 0  # absolute index of the next sample to be written
        self.tol = anchor_tolerance_s
        self._anchor_idx: list[int] = []
        self._anchor_utc: list[float] = []

    # ---------------------------------------------------------------- extent
    @property
    def start(self) -> int:
        """Oldest absolute index still readable."""
        return max(0, self.total - self.capacity)

    @property
    def end(self) -> int:
        return self.total

    # ----------------------------------------------------------------- write
    def write(self, samples: np.ndarray, utc_first: float) -> int:
        """Append samples; returns the absolute index of the first sample written."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        first = self.total
        if not self._anchor_idx or abs(self.index_to_utc(first) - utc_first) > self.tol:
            self._anchor_idx.append(first)
            self._anchor_utc.append(float(utc_first))
        n = x.shape[0]
        if n >= self.capacity:  # only the tail fits
            x = x[-self.capacity :]
            self.total += n - self.capacity
            n = self.capacity
        pos = self.total % self.capacity
        k = min(n, self.capacity - pos)
        self._buf[pos : pos + k] = x[:k]
        if k < n:
            self._buf[: n - k] = x[k:]
        self.total += n
        self._prune()
        return first

    def _prune(self) -> None:
        # Keep the last anchor at or before `start` (it still covers start..next anchor).
        i = bisect.bisect_right(self._anchor_idx, self.start) - 1
        if i > 0:
            del self._anchor_idx[:i]
            del self._anchor_utc[:i]

    def shift_utc(self, delta_s: float, from_idx: int = 0) -> None:
        """Shift anchors at/after ``from_idx`` (clock's one-time startup correction for the
        current connection). If ``from_idx`` falls inside an anchor's span, that span is split."""
        if from_idx > 0 and self._anchor_idx and from_idx < self.total:
            i = bisect.bisect_right(self._anchor_idx, from_idx) - 1
            if i >= 0 and self._anchor_idx[i] != from_idx:
                u = self.index_to_utc(from_idx)
                self._anchor_idx.insert(i + 1, from_idx)
                self._anchor_utc.insert(i + 1, u)
        self._anchor_utc = [
            u + delta_s if idx >= from_idx else u
            for idx, u in zip(self._anchor_idx, self._anchor_utc)
        ]

    # ------------------------------------------------------------------ read
    def read(self, start: int, end: int) -> np.ndarray:
        """Copy of samples [start, end) by absolute index."""
        if start < self.start:
            raise OverwrittenError(f"index {start} overwritten (oldest {self.start})")
        if end > self.total or start > end:
            raise IndexError(f"range [{start},{end}) outside [{self.start},{self.total})")
        n = end - start
        pos = start % self.capacity
        k = min(n, self.capacity - pos)
        out = np.empty(n, dtype=np.float32)
        out[:k] = self._buf[pos : pos + k]
        if k < n:
            out[k:] = self._buf[: n - k]
        return out

    # --------------------------------------------------------------- mapping
    def index_to_utc(self, idx: int) -> float:
        if not self._anchor_idx:
            raise LookupError("no anchors yet")
        i = bisect.bisect_right(self._anchor_idx, idx) - 1
        if i < 0:
            i = 0  # before the first retained anchor: extrapolate backwards
        return self._anchor_utc[i] + (idx - self._anchor_idx[i]) / self.sr

    def utc_to_index(self, utc: float) -> int:
        """Inverse mapping (nearest sample). Uses the last anchor whose UTC <= utc."""
        if not self._anchor_idx:
            raise LookupError("no anchors yet")
        i = bisect.bisect_right(self._anchor_utc, utc) - 1
        i = max(i, 0)
        idx = self._anchor_idx[i] + int(round((utc - self._anchor_utc[i]) * self.sr))
        if i + 1 < len(self._anchor_idx):
            idx = min(idx, self._anchor_idx[i + 1])
        return idx

    @property
    def anchors(self) -> list[tuple[int, float]]:
        return list(zip(self._anchor_idx, self._anchor_utc))
