from fractions import Fraction

import pytest

from live_stt.ingest.clock import PTS_WRAP_90K, PtsClock
from live_stt.settings import ClockConfig

TB = Fraction(1, 90_000)
FRAME_S = 1024 / 48_000  # AAC frame
FRAME_TICKS = 1920  # 1024 samples @48k in 90 kHz ticks


class FakeWall:
    def __init__(self, t: float = 1_700_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def make(cfg=None, t0=1_700_000_000.0, tb=TB):
    wall = FakeWall(t0)
    return PtsClock(tb, cfg or ClockConfig(warmup_s=0.5), wallclock=wall), wall


def feed(clock, wall, pts, n, real_time=True, ticks=FRAME_TICKS, dur=FRAME_S):
    out = []
    for _ in range(n):
        utc, ev = clock.map(pts, dur)
        out.append((pts, utc, ev))
        pts += ticks
        if real_time:
            wall.t += dur
    return pts, out


def all_events(out):
    return [e for _, _, ev in out for e in ev]


def test_steady_stream_is_exact_and_anchored_to_wall():
    clock, wall = make()
    t0 = wall.t
    _, out = feed(clock, wall, 1000, 200)
    assert out[0][1] == pytest.approx(t0)
    for i in range(1, len(out)):
        assert out[i][1] - out[i - 1][1] == pytest.approx(FRAME_S, abs=1e-6)
    types = [e.type for e in all_events(out)]
    assert types == ["anchor"]
    assert clock.warm and abs(clock.drift_ms) < 1


def test_33bit_wraparound_is_continuous():
    clock, wall = make()
    start = PTS_WRAP_90K - 50 * FRAME_TICKS
    pts, out = feed(clock, wall, start, 100)
    # emulate the encoder's wrapped values
    clock2, wall2 = make()
    utcs, raw = [], start
    for _ in range(100):
        utc, ev = clock2.map(raw % PTS_WRAP_90K, FRAME_S)
        assert not any(e.type == "discontinuity" for e in ev)
        utcs.append(utc)
        raw += FRAME_TICKS
        wall2.t += FRAME_S
    assert clock2.wraps == 1
    for a, b in zip(utcs, utcs[1:]):
        assert b - a == pytest.approx(FRAME_S, abs=1e-6)


def test_backward_jump_reanchors_monotonic():
    """Looping test stream: PTS jumps back to the start each loop."""
    clock, wall = make()
    _, out1 = feed(clock, wall, 5000, 300)  # ~6.4 s
    last_end = out1[-1][1] + FRAME_S
    utc, ev = clock.map(5000, FRAME_S)  # loop restart
    assert [e.type for e in ev] == ["discontinuity"]
    assert ev[0].details["jump_s"] < 0
    assert utc >= last_end - 1e-9
    assert utc == pytest.approx(wall.t, abs=0.05)  # wall clock minus ~0 baseline
    assert clock.discontinuities == 1


def test_forward_jump_over_threshold_reanchors():
    clock, wall = make()
    pts, _ = feed(clock, wall, 0, 100)
    wall.t += 5.0  # signal loss upstream for 5 s
    utc, ev = clock.map(pts + 5 * 90_000, FRAME_S)
    assert [e.type for e in ev] == ["discontinuity"]
    assert utc == pytest.approx(wall.t, abs=0.05)


def test_small_forward_gap_is_not_discontinuity():
    clock, wall = make()
    pts, _ = feed(clock, wall, 0, 100)
    utc_before = clock.last_end_utc
    utc, ev = clock.map(pts + int(0.3 * 90_000), FRAME_S)  # 300 ms gap < 1 s threshold
    assert not ev
    assert utc == pytest.approx(utc_before + 0.3, abs=1e-6)


def test_startup_burst_is_corrected():
    """First 1 s decodes instantly (probe buffer), then real time: anchor shifts earlier by ~1 s."""
    cfg = ClockConfig(warmup_s=3.0)
    clock, wall = make(cfg)
    t0 = wall.t
    burst = int(1.0 / FRAME_S)
    pts, out_b = feed(clock, wall, 0, burst, real_time=False)
    pts, out_r = feed(clock, wall, pts, 200, real_time=True)
    ev = [e for e in all_events(out_b + out_r) if e.type == "startup_correction"]
    assert len(ev) == 1
    shift = ev[0].details["shift_s"]
    assert shift == pytest.approx(-burst * FRAME_S, abs=FRAME_S)
    # After correction, a frame's UTC equals its arrival (wall) time.
    utc, _ = clock.map(pts, FRAME_S)
    assert utc == pytest.approx(wall.t, abs=FRAME_S)
    # The first frame's corrected UTC is ~1 s before the moment it was decoded.
    assert t0 + shift == pytest.approx(t0 - 1.0, abs=FRAME_S)


def test_drift_detected_and_slewed_monotonically():
    cfg = ClockConfig(warmup_s=0.5, drift_threshold_ms=500, drift_sustain_s=5,
                      drift_target_ms=50, drift_slew_ms_per_s=20, drift_ema_tau_s=1)
    clock, wall = make(cfg)
    pts = 0
    utcs, events = [], []
    # Encoder clock 1% slow vs host: wall advances 1.01 × frame duration per frame.
    for _ in range(int(200 / FRAME_S)):
        utc, ev = clock.map(pts, FRAME_S)
        utcs.append(utc)
        events += ev
        pts += FRAME_TICKS
        wall.t += FRAME_S * 1.01
    assert any(e.type == "drift" for e in events)
    assert all(b > a for a, b in zip(utcs, utcs[1:]))
    # Slew capacity (20 ms/s) exceeds drift rate (10 ms/s) → drift held near threshold.
    assert abs(clock.drift_ms) < cfg.drift_threshold_ms + 100


def test_non_90k_time_base_wrap_period():
    clock = PtsClock(Fraction(1, 48_000), ClockConfig())
    assert clock.wrap_ticks == round(PTS_WRAP_90K / 90_000 * 48_000)


def test_reset_forgets_anchor():
    clock, wall = make()
    feed(clock, wall, 0, 10)
    clock.reset()
    wall.t += 100
    utc, ev = clock.map(123456, FRAME_S)
    assert ev[0].type == "anchor" and utc == wall.t


def test_sub_alert_offset_is_corrected_without_alert():
    """Sender runs 300 ms ahead after warmup (gradual initial burst): slewed back near zero."""
    cfg = ClockConfig(warmup_s=1.0, drift_sustain_s=3, drift_ema_tau_s=1)
    clock, wall = make(cfg)
    pts = 0
    events, utcs = [], []
    n_burst = int(5 / FRAME_S)
    for i in range(int(60 / FRAME_S)):
        utc, ev = clock.map(pts, FRAME_S)
        events += ev
        utcs.append(utc)
        pts += FRAME_TICKS
        # after warmup, frames arrive 6% early for 5 s (≈300 ms total), then real time
        early = 0.06 if int(1.0 / FRAME_S) < i < int(1.0 / FRAME_S) + n_burst else 0.0
        wall.t += FRAME_S * (1 - early)
    drift_ev = [e for e in events if e.type == "drift"]
    assert drift_ev and drift_ev[0].details["alert"] is False
    assert abs(clock.drift_ms) < 80
    assert all(b > a for a, b in zip(utcs, utcs[1:]))
