import numpy as np
import pytest
from hypothesis import given, settings as hsettings, strategies as st

from live_stt.settings import StreamingConfig, VadConfig
from live_stt.streaming.segment_policy import SegmentPolicy

SR, F = 16_000, 512
FPS = SR / F  # frames per second


def run(probs, vad=None, stream=None, rms=None):
    p = SegmentPolicy(vad or VadConfig(), stream or StreamingConfig(), SR, F)
    cuts = []
    for i, pr in enumerate(probs):
        cuts += p.process(i * F, pr, (rms[i] if rms is not None else pr))
    return p, cuts


def frames(seconds, prob):
    return [prob] * int(round(seconds * FPS))


def test_single_utterance_with_padding():
    probs = frames(1, 0.0) + frames(2, 0.9) + frames(1, 0.0)
    _, cuts = run(probs)
    assert len(cuts) == 1
    c = cuts[0]
    assert c.kind == "final" and c.reason == "silence"
    speech_start = int(round(1 * FPS)) * F
    assert c.speech_start == speech_start
    assert c.start == speech_start - int(0.2 * SR)
    assert c.end == c.speech_end + int(0.2 * SR)


def test_short_blip_discarded():
    probs = frames(1, 0.0) + [0.9] * 3 + frames(1, 0.0)  # ~96 ms < 250 ms min speech
    assert run(probs)[1] == []


def test_short_dip_does_not_split():
    probs = frames(1, 0) + frames(1, 0.9) + frames(0.3, 0.0) + frames(1, 0.9) + frames(1, 0)
    _, cuts = run(probs)
    assert len(cuts) == 1


def test_hysteresis_between_neg_and_threshold_keeps_speech():
    probs = frames(1, 0) + frames(1, 0.9) + frames(2, 0.4) + frames(1, 0)  # 0.4 > neg 0.35
    _, cuts = run(probs)
    assert len(cuts) == 1
    assert cuts[0].speech_end - cuts[0].speech_start >= int(3 * SR) - F


def test_long_speech_force_cut_contiguous():
    probs = frames(0.5, 0) + frames(30, 0.95) + frames(1, 0)
    _, cuts = run(probs)
    assert len(cuts) >= 4
    for a, b in zip(cuts, cuts[1:]):
        assert a.end == b.start, "segments must be contiguous across forced cuts"
    for c in cuts:
        assert c.n <= int(8 * SR) + F
    assert [c.reason for c in cuts[:-1]] == ["max_len"] * (len(cuts) - 1)
    assert cuts[-1].reason == "silence"


def test_force_cut_prefers_quiet_frame():
    n_speech = int(10 * FPS)
    probs = frames(0.5, 0) + [0.95] * n_speech + frames(1, 0)
    rms = [0.5] * len(probs)
    # Quiet dip at 7.5 s into the speech (inside the last 1.5 s before the 8 s limit)
    dip = int(0.5 * FPS) + int(7.5 * FPS)
    rms[dip] = 0.01
    probs[dip] = 0.6
    _, cuts = run(probs, rms=rms)
    assert cuts[0].reason == "max_len"
    assert cuts[0].end == dip * F + F // 2


def test_partials_share_segment_key_with_final():
    probs = frames(0.5, 0) + frames(5, 0.9) + frames(1, 0)
    _, cuts = run(probs, stream=StreamingConfig(partials_enabled=True, partial_interval_s=2))
    partials = [c for c in cuts if c.kind == "partial"]
    finals = [c for c in cuts if c.kind == "final"]
    assert len(partials) == 2 and len(finals) == 1
    assert all(p.segment_key == finals[0].segment_key for p in partials)
    assert partials[0].end < partials[1].end <= finals[0].end
    seqs = [c.seq for c in cuts]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_flush_closes_open_segment():
    probs = frames(0.5, 0) + frames(3, 0.9)
    p, cuts = run(probs)
    assert cuts == []
    out = p.flush()
    assert len(out) == 1 and out[0].reason == "flush" and out[0].end == p.head
    assert p.flush() == []  # idempotent


def test_reset_drops_state_but_keeps_seq():
    p, _ = run(frames(0.5, 0) + frames(1, 0.9) + frames(1, 0))
    seq = p.seq
    p.reset()
    assert not p.triggered and p.seq == seq


def test_pad_never_overlaps_previous_segment():
    probs = frames(0.5, 0) + frames(1, 0.9) + frames(0.55, 0) + frames(1, 0.9) + frames(1, 0)
    _, cuts = run(probs, vad=VadConfig(min_silence_ms=500, speech_pad_ms=400))
    assert len(cuts) == 2
    assert cuts[1].start >= cuts[0].end


@hsettings(max_examples=300, deadline=None)
@given(
    st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=1, max_size=3000),
    st.floats(min_value=2, max_value=12),
    st.integers(min_value=0, max_value=600),
    st.integers(min_value=0, max_value=1000),
    st.booleans(),
)
def test_invariants_random(probs, max_seg, pad_ms, min_sil_ms, partials):
    vad = VadConfig(speech_pad_ms=pad_ms, min_silence_ms=min_sil_ms, min_speech_ms=100)
    sc = StreamingConfig(max_segment_s=max_seg, partials_enabled=partials, partial_interval_s=1)
    p, cuts = run(probs, vad=vad, stream=sc)
    cuts += p.flush()
    finals = [c for c in cuts if c.kind == "final"]
    head = len(probs) * F
    for c in cuts:
        assert 0 <= c.start < c.end <= head
        assert c.start <= c.speech_start <= c.speech_end <= c.end or c.kind == "partial"
    for c in finals:
        assert c.n <= p.max_seg + F
    for a, b in zip(finals, finals[1:]):
        assert a.end <= b.start, "final segments must never overlap"
    seqs = [c.seq for c in cuts]
    assert seqs == sorted(seqs)
