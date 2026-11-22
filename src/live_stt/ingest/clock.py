"""PTS-anchored UTC clock.

    utc(frame) = anchor_utc + (pts_unwrapped - anchor_pts) * time_base + slew

Responsibilities (all pure logic; wall clock is injected so it is unit-testable):

* **33-bit wraparound.** MPEG-TS PTS is 33 bits at 90 kHz (wraps every ~26.5 h). The wrap period is
  expressed in the stream's own ``time_base`` so it also works if PyAV rescales.
* **Discontinuities.** A backward jump (> ``backward_tolerance_s``) or a forward jump
  (> ``discontinuity_s``) relative to the expected next PTS re-anchors the clock and reports a
  ``discontinuity`` event. The new anchor is the wall clock minus the steady-state buffering offset,
  clamped so UTC never goes backwards. Looping test streams hit this once per loop.
* **Startup offset.** The first decoded frame is anchored to the wall clock, but frames buffered
  during probing are then decoded in a burst, so the anchor is late by the probe buffer duration
  (B). After ``warmup_s`` of stream time we measure the steady-state ``wall - utc`` offset (≈ -B) and
  report a one-time ``startup_correction`` (shift UTC earlier by B). Callers (ring buffer) apply the
  shift retroactively; workers do not emit segments before the clock is warm.
* **Drift.** EMA of ``wall - utc`` vs. the baseline measured at warmup. If |drift| exceeds
  ``drift_correct_ms`` (100 ms) for ``drift_sustain_s`` of wall time, a ``drift`` event is
  reported (``alert=true`` above ``drift_threshold_ms``, 500 ms) and the clock slews
  (rate-limited, so UTC stays monotonic) until within ``drift_target_ms``. The lower correction
  threshold also removes offsets that build up after warmup (e.g. a sender that bursts ahead for
  its first seconds), keeping timestamps inside the 300 ms accuracy budget.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable

from live_stt.settings import ClockConfig

PTS_BITS = 33
PTS_WRAP_90K = 1 << PTS_BITS
MPEG_CLOCK = 90_000
# Startup corrections smaller than this are jitter, not probe buffering; ignore them.
MIN_STARTUP_CORRECTION_S = 0.005


@dataclass
class ClockEvent:
    type: str  # "anchor" | "discontinuity" | "startup_correction" | "drift"
    utc: float
    details: dict = field(default_factory=dict)


class PtsClock:
    def __init__(
        self,
        time_base: Fraction,
        cfg: ClockConfig | None = None,
        wallclock: Callable[[], float] = time.time,
    ) -> None:
        self.tb = Fraction(time_base)
        self.cfg = cfg or ClockConfig()
        self.wallclock = wallclock
        # Wrap period in stream ticks (2^33 ticks of 90 kHz expressed in time_base units).
        self.wrap_ticks = int(round(Fraction(PTS_WRAP_90K, MPEG_CLOCK) / self.tb))
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Forget everything (call after reconnect). Next frame anchors to wall clock."""
        self.anchor_utc: float | None = None
        self.anchor_pts: int = 0  # unwrapped ticks
        self.wrap_offset: int = 0  # accumulated wraps, ticks
        self.last_raw: int | None = None
        self.expected_pts: int | None = None  # unwrapped, end of last frame
        self.last_end_utc: float | None = None
        self._last_start: float | None = None
        self.slew: float = 0.0  # seconds added to utc (drift correction)
        self.offset_ema: float | None = None  # EMA of (wall - utc)
        self.baseline: float | None = None  # steady-state offset at warmup
        self.warm: bool = False
        self._warm_stream_s: float = 0.0
        self._last_wall: float | None = None
        self._drift_since: float | None = None
        self._slewing: bool = False
        self.wraps: int = 0
        self.discontinuities: int = 0

    @property
    def drift_ms(self) -> float:
        if self.offset_ema is None or self.baseline is None:
            return 0.0
        return (self.offset_ema - self.baseline) * 1000.0

    # --------------------------------------------------------------- helpers
    def _unwrap(self, raw: int) -> tuple[int, bool]:
        wrapped = False
        if self.last_raw is not None:
            half = self.wrap_ticks // 2
            diff = raw - self.last_raw
            if diff < -half:  # e.g. 2^33-100 → 50: forward across the wrap
                self.wrap_offset += self.wrap_ticks
                wrapped = True
            elif diff > half:  # backward across the wrap (rare: reordering/rewind)
                self.wrap_offset -= self.wrap_ticks
        self.last_raw = raw
        return raw + self.wrap_offset, wrapped

    def _ticks_to_s(self, ticks: int) -> float:
        return float(ticks * self.tb)

    def _anchor(self, pts: int, utc: float) -> None:
        self.anchor_pts = pts
        self.anchor_utc = utc
        self.slew = 0.0

    # ------------------------------------------------------------------ main
    def map(self, pts: int, duration_s: float) -> tuple[float, list[ClockEvent]]:
        """Return (utc of frame start, events) for a decoded frame.

        ``pts`` is the raw (possibly wrapped) PTS in ``time_base`` units; ``duration_s`` is the
        frame duration (n_samples / sample_rate).
        """
        events: list[ClockEvent] = []
        wall = self.wallclock()
        upts, wrapped = self._unwrap(int(pts))
        if wrapped:
            self.wraps += 1

        if self.anchor_utc is None:
            self._anchor(upts, wall)
            events.append(ClockEvent("anchor", wall, {"pts": upts}))
        elif self.expected_pts is not None:
            delta_s = self._ticks_to_s(upts - self.expected_pts)
            if delta_s < -self.cfg.backward_tolerance_s or delta_s > self.cfg.discontinuity_s:
                self.discontinuities += 1
                # Wall clock minus buffering offset → best estimate of this frame's air time.
                offset = self.baseline if self.baseline is not None else (self.offset_ema or 0.0)
                new_utc = wall - offset
                if self.last_end_utc is not None:
                    new_utc = max(new_utc, self.last_end_utc)  # monotonic UTC
                events.append(
                    ClockEvent(
                        "discontinuity",
                        new_utc,
                        {"jump_s": round(delta_s, 6), "pts": upts, "expected": self.expected_pts},
                    )
                )
                self._anchor(upts, new_utc)
                self._drift_since = None
                self._slewing = False

        utc = self.anchor_utc + self._ticks_to_s(upts - self.anchor_pts) + self.slew
        # Monotonic frame starts. (A negative drift slew overlaps consecutive frames by at most
        # slew_rate × frame duration, e.g. 0.4 ms per 21 ms AAC frame — intended, not clamped.)
        if self._last_start is not None and utc <= self._last_start:
            utc = self._last_start + 1e-6

        tracked = self._track_offset(wall, utc, duration_s)
        for ev in tracked:
            if ev.type == "startup_correction":
                utc += ev.details["shift_s"]  # this frame is shifted too
        events.extend(tracked)
        # (drift slew is applied to *subsequent* frames)

        self.expected_pts = upts + int(round(Fraction(duration_s) / self.tb))
        self.last_end_utc = utc + duration_s
        self._last_start = utc
        return utc, events

    def _track_offset(self, wall: float, utc: float, duration_s: float) -> list[ClockEvent]:
        events: list[ClockEvent] = []
        offset = wall - utc
        dt_wall = 0.0 if self._last_wall is None else max(0.0, wall - self._last_wall)
        self._last_wall = wall

        if not self.warm:
            # During warmup track the minimum offset: buffered frames decode in a burst, so the
            # minimum (≈ -B) is the steady state once the burst has drained.
            self.offset_ema = offset if self.offset_ema is None else min(self.offset_ema, offset)
            self._warm_stream_s += duration_s
            if self._warm_stream_s >= self.cfg.warmup_s:
                self.warm = True
                correction = min(0.0, self.offset_ema)  # only ever shift earlier
                if correction < -MIN_STARTUP_CORRECTION_S:
                    self.anchor_utc += correction
                    if self.last_end_utc is not None:
                        self.last_end_utc += correction
                    if self._last_start is not None:
                        self._last_start += correction
                    events.append(
                        ClockEvent("startup_correction", utc + correction, {"shift_s": correction})
                    )
                    self.offset_ema -= correction
                else:
                    correction = 0.0
                self.baseline = self.offset_ema
            return events

        tau = max(self.cfg.drift_ema_tau_s, 1e-3)
        alpha = 1.0 - math.exp(-duration_s / tau)
        self.offset_ema += alpha * (offset - self.offset_ema)

        drift_s = self.offset_ema - self.baseline
        alert = self.cfg.drift_threshold_ms / 1000.0
        thr = min(self.cfg.drift_correct_ms, self.cfg.drift_threshold_ms) / 1000.0
        if not self._slewing:
            if abs(drift_s) > thr:
                if self._drift_since is None:
                    self._drift_since = wall
                elif wall - self._drift_since >= self.cfg.drift_sustain_s:
                    self._slewing = True
                    events.append(ClockEvent("drift", utc, {
                        "drift_ms": round(drift_s * 1000, 1), "alert": abs(drift_s) > alert}))
            else:
                self._drift_since = None
        if self._slewing:
            if abs(drift_s) <= self.cfg.drift_target_ms / 1000.0:
                self._slewing = False
                self._drift_since = None
            else:
                # Positive drift: wall ahead of utc → utc too early → add slew.
                # Rate is per second of *media* time so the per-frame overlap stays bounded.
                step = min(abs(drift_s), self.cfg.drift_slew_ms_per_s / 1000.0 * duration_s)
                step = math.copysign(step, drift_s)
                self.slew += step
                self.offset_ema -= step
        return events
