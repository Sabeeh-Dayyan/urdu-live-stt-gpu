"""ChannelPipeline driven by synthetic audio + energy VAD + in-memory bus (no PyAV/ONNX)."""

import numpy as np
import pytest
from hypothesis import given, settings as hsettings, strategies as st

from live_stt.bus.base import InMemoryBus, pcm16_to_f32
from live_stt.ingest.clock import ClockEvent
from live_stt.ingest.source import AudioChunk, SourceEvent
from live_stt.settings import ChannelConfig, Settings
from live_stt.worker import ChannelPipeline

SR = 16_000
T0 = 1_700_000_000.0


class EnergyVAD:
    def __init__(self):
        self.resets = 0

    def __call__(self, frame):
        return 0.95 if float(np.sqrt(np.mean(frame * frame))) > 0.05 else 0.02

    def reset(self):
        self.resets += 1


def settings(**asr):
    s = Settings()
    s.asr.refine_enabled = asr.get("refine", True)
    s.channels = [ChannelConfig(id="ch1", url="udp://x:1")]
    return s


def program(spec):
    """spec: list of (seconds, is_speech) → float32 audio."""
    parts = []
    rng = np.random.default_rng(1)
    for sec, speech in spec:
        n = int(sec * SR)
        if speech:
            t = np.arange(n) / SR
            parts.append((0.3 * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.standard_normal(n))
                         .astype(np.float32))
        else:
            parts.append((0.001 * rng.standard_normal(n)).astype(np.float32))
    return np.concatenate(parts)


def feed(pipe, audio, chunk_sizes, utc0=T0, warm=True):
    pos = 0
    i = 0
    while pos < len(audio):
        n = chunk_sizes[i % len(chunk_sizes)]
        pipe.on_chunk(AudioChunk(audio[pos:pos + n], utc0 + pos / SR), warm)
        pos += n
        i += 1


def fast(bus, s):
    return bus.messages(s.bus.stream_fast)


def test_segments_utc_and_audio_exact():
    s, bus = settings(), InMemoryBus()
    pipe = ChannelPipeline(s, s.channels[0], bus, EnergyVAD(), wall=lambda: T0 + 100)
    audio = program([(1, 0), (2, 1), (1.5, 0), (3, 1), (2, 0)])
    feed(pipe, audio, [1024, 777, 4000, 333])
    msgs = fast(bus, s)
    assert len(msgs) == 2
    for m, (sp_start, sp_dur) in zip(msgs, [(1.0, 2.0), (4.5, 3.0)]):
        pcm = pcm16_to_f32(m["pcm"])
        assert abs(len(pcm) / SR - m["duration_s"]) < 1e-6
        assert m["speech_start_utc"] == pytest.approx(T0 + sp_start, abs=0.04)
        assert m["speech_end_utc"] == pytest.approx(T0 + sp_start + sp_dur, abs=0.04)
        assert m["utc_start"] == pytest.approx(m["speech_start_utc"] - 0.2, abs=0.04)
        # Audio content matches the source exactly at the mapped position.
        off = int(round((m["utc_start"] - T0) * SR))
        np.testing.assert_allclose(pcm, audio[off:off + len(pcm)], atol=1 / 16000)
        assert m["segment_id"] == f"ch1:{int(round(m['utc_start'] * 1000))}"
    ref = bus.messages(s.bus.stream_refine)
    assert len(ref) == 2
    for r, m in zip(ref, msgs):
        assert r["segment_id"] == m["segment_id"]
        assert r["ctx_pre_s"] == pytest.approx(0.5, abs=1e-6)
        assert r["utc_start"] == pytest.approx(m["utc_start"] - 0.5, abs=1e-6)


def test_held_until_clock_warm_then_released():
    s, bus = settings(refine=False), InMemoryBus()
    pipe = ChannelPipeline(s, s.channels[0], bus, EnergyVAD())
    audio = program([(0.5, 0), (1, 1), (1, 0)])
    feed(pipe, audio, [800], warm=False)
    assert fast(bus, s) == []
    pipe.on_chunk(AudioChunk(np.zeros(800, np.float32), T0 + len(audio) / SR), True)
    assert len(fast(bus, s)) == 1


def test_startup_correction_shifts_held_segment_utc():
    s, bus = settings(refine=False), InMemoryBus()
    pipe = ChannelPipeline(s, s.channels[0], bus, EnergyVAD())
    pipe.on_source_event(SourceEvent("stream_up", T0))
    audio = program([(0.5, 0), (1, 1), (1, 0)])
    feed(pipe, audio, [800], warm=False)
    corr = ClockEvent("startup_correction", T0, {"shift_s": -0.75})
    pipe.on_chunk(AudioChunk(np.zeros(800, np.float32), T0 - 0.75 + len(audio) / SR, [corr]),
                  True)
    m = fast(bus, s)[0]
    assert m["speech_start_utc"] == pytest.approx(T0 + 0.5 - 0.75, abs=0.04)


def test_discontinuity_flushes_open_segment():
    s, bus = settings(refine=False), InMemoryBus()
    pipe = ChannelPipeline(s, s.channels[0], bus, EnergyVAD())
    feed(pipe, program([(0.5, 0), (2, 1)]), [512])
    assert fast(bus, s) == []
    disc = ClockEvent("discontinuity", T0 + 10, {"jump_s": -20.0})
    pipe.on_chunk(AudioChunk(program([(0.5, 1)]), T0 + 10, [disc]), True)
    msgs = fast(bus, s)
    assert len(msgs) == 1 and msgs[0]["cut_reason"] == "flush"
    assert msgs[0]["speech_end_utc"] <= T0 + 2.5 + 1e-6
    sys_events = [e["type"] for e in bus.messages(s.bus.stream_system)]
    assert "discontinuity" in sys_events


def test_stream_down_flushes_and_resets():
    s, bus = settings(), InMemoryBus()
    vad = EnergyVAD()
    pipe = ChannelPipeline(s, s.channels[0], bus, vad)
    pipe.on_source_event(SourceEvent("stream_up", T0))
    feed(pipe, program([(0.5, 0), (2, 1)]), [512])
    pipe.on_source_event(SourceEvent("stream_down", T0 + 3, {"reason": "eof"}))
    assert len(fast(bus, s)) == 1
    assert len(bus.messages(s.bus.stream_refine)) == 1  # forced out on flush
    assert vad.resets == 1 and not pipe.warm


def test_nan_inf_audio_does_not_crash_or_leak_nan():
    s, bus = settings(), InMemoryBus()
    pipe = ChannelPipeline(s, s.channels[0], bus, EnergyVAD())
    audio = program([(0.5, 0), (2, 1), (1, 0)])
    audio[5000:5100] = np.nan
    audio[9000] = np.inf
    feed(pipe, audio, [1000])
    for m in fast(bus, s):
        assert np.all(np.isfinite(pcm16_to_f32(m["pcm"])))


def test_ring_overrun_counts_lost_instead_of_crashing():
    s, bus = settings(refine=False), InMemoryBus()
    s.ingest.ring_buffer_s = 2.0
    s.streaming.max_segment_s = 30  # segment longer than the ring
    pipe = ChannelPipeline(s, s.channels[0], bus, EnergyVAD())
    feed(pipe, program([(0.5, 0), (5, 1), (1, 0)]), [1600])
    assert pipe.stats["lost"] >= 1 or all(m["duration_s"] <= 2.0 for m in fast(bus, s))


@hsettings(max_examples=40, deadline=None)
@given(st.lists(st.integers(min_value=1, max_value=20_000), min_size=1, max_size=8),
       st.lists(st.tuples(st.floats(0.05, 4), st.booleans()), min_size=1, max_size=10))
def test_chunking_invariance(chunks, spec):
    """Segment decisions must not depend on how audio is chunked."""
    audio = program(spec)
    results = []
    for sizes in (chunks, [len(audio) or 1]):
        s, bus = settings(refine=False), InMemoryBus()
        pipe = ChannelPipeline(s, s.channels[0], bus, EnergyVAD())
        feed(pipe, audio, sizes)
        pipe._flush()
        results.append([(m["utc_start"], m["duration_s"]) for m in fast(bus, s)])
    a, b = results
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x[0] == pytest.approx(y[0], abs=1e-6) and x[1] == pytest.approx(y[1], abs=1e-6)


def test_held_cuts_emitted_in_order_before_newer_ones():
    s, bus = settings(refine=False), InMemoryBus()
    pipe = ChannelPipeline(s, s.channels[0], bus, EnergyVAD())
    a1 = program([(0.5, 0), (1, 1), (1, 0)])
    feed(pipe, a1, [800], warm=False)
    a2 = program([(1, 1), (1, 0)])
    # warm arrives in the same chunk that completes the next segment
    pipe.on_chunk(AudioChunk(a2, T0 + len(a1) / SR), True)
    starts = [m["utc_start"] for m in fast(bus, s)]
    assert len(starts) == 2 and starts == sorted(starts)
    seqs = [m["seq"] for m in fast(bus, s)]
    assert seqs == sorted(seqs)
