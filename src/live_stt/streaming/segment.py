"""Segment data types shared by streaming policies and the worker."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SegmentCut:
    """A span of audio (absolute ring-buffer sample indices) decided by a streaming policy.

    ``kind``:  "final"   — finished speech segment (send FAST, later REFINE)
               "partial" — growing segment snapshot (FAST only, superseded by final)
    ``reason``: "silence" | "max_len" | "flush" | "partial"
    ``speech_start`` / ``speech_end`` exclude padding (used for latency accounting).
    """

    kind: str
    start: int
    end: int
    speech_start: int
    speech_end: int
    reason: str
    seq: int = 0
    segment_key: int = 0  # start index of the speech run (stable across partials → final)
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.end - self.start


def make_segment_id(channel_id: str, start_utc: float) -> str:
    """Stable, unique per channel: channel + millisecond UTC of the segment start.

    UTC from the PTS clock is monotonic per channel, and a re-anchor after restart uses the wall
    clock, so collisions require two segments starting in the same millisecond on one channel.
    """
    return f"{channel_id}:{int(round(start_utc * 1000))}"
