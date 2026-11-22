"""`segment` streaming policy: VAD-delimited segments with forced cuts.

Pure logic. Input: per-VAD-frame (absolute start index, speech probability, RMS energy).
Output: ``SegmentCut`` decisions. The worker maps indices to UTC and reads audio.

State machine (hysteresis like Silero's VADIterator):
* idle → speech when prob ≥ threshold.
* speech → candidate end at the first frame with prob < neg_threshold; confirmed after
  ``min_silence_ms`` of continuous non-speech. Speech runs shorter than ``min_speech_ms`` are
  discarded.
* Forced cut once the open segment reaches ``max_segment_s``: cut at the frame in the last
  ``cut_search_s`` with the lowest (normalised energy + speech prob), i.e. prefer a VAD dip /
  quiet point between words. The remainder continues as a new segment with no gap or overlap.
* Padding: ``speech_pad_ms`` on both sides, never overlapping the previous emitted segment and
  never beyond audio already received (pad ≤ min_silence guarantees the right pad exists).
* Optional partials every ``partial_interval_s`` while speech is open.
"""

from __future__ import annotations

from collections import deque

from live_stt.settings import StreamingConfig, VadConfig
from live_stt.streaming.segment import SegmentCut


class SegmentPolicy:
    def __init__(
        self,
        vad: VadConfig,
        streaming: StreamingConfig,
        sample_rate: int = 16_000,
        frame: int = 512,
    ) -> None:
        self.sr = sample_rate
        self.frame = frame
        self.thr = vad.threshold
        self.neg = vad.neg_threshold if vad.neg_threshold is not None else max(vad.threshold - 0.15, 0.01)
        self.min_speech = int(vad.min_speech_ms * sample_rate / 1000)
        self.min_silence = int(vad.min_silence_ms * sample_rate / 1000)
        self.pad = min(int(vad.speech_pad_ms * sample_rate / 1000), self.min_silence)
        self.max_seg = max(int(streaming.max_segment_s * sample_rate), 4 * frame)
        self.cut_search = min(int(streaming.cut_search_s * sample_rate), self.max_seg // 2)
        self.partials = streaming.partials_enabled
        self.partial_every = int(streaming.partial_interval_s * sample_rate)
        # Frame history for forced-cut search (start_idx, prob, rms)
        self._hist: deque[tuple[int, float, float]] = deque(
            maxlen=max(1, self.cut_search // frame + 2)
        )
        self.seq = 0
        self.reset()

    def reset(self) -> None:
        """Forget open speech (after reconnect). Does not reset ``seq``."""
        self.triggered = False
        self.seg_start = 0  # padded start of the open segment
        self.speech_start = 0  # first speech frame of the open segment
        self.temp_end: int | None = None
        self.last_emitted_end = 0
        self.last_partial = 0
        self.head = 0  # absolute index after the last processed frame
        self.segment_key = 0
        self._hist.clear()

    # ------------------------------------------------------------------ main
    def process(self, start: int, prob: float, rms: float) -> list[SegmentCut]:
        out: list[SegmentCut] = []
        end = start + self.frame
        self.head = end
        self._hist.append((start, prob, rms))

        if prob >= self.thr:
            self.temp_end = None
            if not self.triggered:
                self.triggered = True
                self.speech_start = start
                self.seg_start = max(start - self.pad, self.last_emitted_end, 0)
                self.last_partial = start
                self.segment_key = start
        elif self.triggered and prob < self.neg:
            if self.temp_end is None:
                self.temp_end = start
            if end - self.temp_end >= self.min_silence:
                cut = self._close(self.temp_end, "silence", right_pad=True)
                if cut:
                    out.append(cut)

        if self.triggered and end - self.seg_start >= self.max_seg:
            out.append(self._force_cut(end))

        if (
            self.partials
            and self.triggered
            and self.temp_end is None
            and end - self.last_partial >= self.partial_every
        ):
            self.last_partial = end
            self.seq += 1
            out.append(
                SegmentCut("partial", self.seg_start, end, self.speech_start, end, "partial",
                           self.seq, self.segment_key)
            )
        return out

    def flush(self, at: int | None = None) -> list[SegmentCut]:
        """Close any open segment at ``at`` (default: last processed sample)."""
        if not self.triggered:
            return []
        at = self.head if at is None else at
        end = self.temp_end if self.temp_end is not None else at
        cut = self._close(min(end, at), "flush", right_pad=False, limit=at)
        return [cut] if cut else []

    # --------------------------------------------------------------- helpers
    def _close(self, speech_end: int, reason: str, right_pad: bool, limit: int | None = None):
        self.triggered = False
        self.temp_end = None
        if speech_end - self.speech_start < self.min_speech:
            return None
        end = speech_end + (self.pad if right_pad else 0)
        end = min(end, self.head if limit is None else limit)
        if end <= self.seg_start:
            return None
        self.seq += 1
        cut = SegmentCut("final", self.seg_start, end, self.speech_start, speech_end, reason,
                         self.seq, self.segment_key)
        self.last_emitted_end = end
        return cut

    def _force_cut(self, head: int) -> SegmentCut:
        lo = head - self.cut_search
        cands = [h for h in self._hist if h[0] >= lo and h[0] > self.seg_start + self.frame]
        if cands:
            peak = max(c[2] for c in cands) or 1.0
            best = min(cands, key=lambda c: (c[2] / peak) + c[1])
            cut_at = best[0] + self.frame // 2  # middle of the quietest frame
        else:
            cut_at = head
        cut_at = min(max(cut_at, self.seg_start + self.frame), head)
        self.seq += 1
        cut = SegmentCut("final", self.seg_start, cut_at, self.speech_start, cut_at, "max_len",
                         self.seq, self.segment_key)
        self.last_emitted_end = cut_at
        # Continue speech as a new segment starting exactly at the cut.
        self.seg_start = cut_at
        self.speech_start = cut_at
        self.segment_key = cut_at
        self.last_partial = cut_at
        if self.temp_end is not None and self.temp_end < cut_at:
            self.temp_end = cut_at
        return cut
