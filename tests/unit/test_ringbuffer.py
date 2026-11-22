import numpy as np
import pytest

from live_stt.ingest.ringbuffer import OverwrittenError, RingBuffer

SR = 16_000
T0 = 1_700_000_000.0


def test_write_read_and_wrap():
    rb = RingBuffer(1.0, SR)
    data = np.arange(int(2.5 * SR), dtype=np.float32)
    for i in range(0, len(data), 1000):
        rb.write(data[i : i + 1000], T0 + i / SR)
    assert rb.total == len(data)
    assert rb.start == len(data) - SR
    np.testing.assert_array_equal(rb.read(rb.start, rb.end), data[rb.start :])
    with pytest.raises(OverwrittenError):
        rb.read(0, 10)
    with pytest.raises(IndexError):
        rb.read(rb.end - 5, rb.end + 5)


def test_oversized_write_keeps_tail():
    rb = RingBuffer(0.5, SR)
    data = np.arange(SR, dtype=np.float32)
    first = rb.write(data, T0)
    assert first == 0 and rb.total == SR
    np.testing.assert_array_equal(rb.read(rb.start, rb.end), data[-SR // 2 :])
    assert rb.index_to_utc(SR // 2) == pytest.approx(T0 + 0.5)


def test_continuous_writes_single_anchor():
    rb = RingBuffer(10, SR)
    for k in range(50):
        rb.write(np.zeros(320, np.float32), T0 + k * 0.02)
    assert len(rb.anchors) == 1
    assert rb.index_to_utc(16_000) == pytest.approx(T0 + 1.0)


def test_reanchor_and_mapping_across_anchors():
    rb = RingBuffer(10, SR)
    rb.write(np.zeros(SR, np.float32), T0)  # [0,16000) → T0..T0+1
    rb.write(np.zeros(SR, np.float32), T0 + 5.0)  # jump: new anchor at 16000
    assert len(rb.anchors) == 2
    assert rb.index_to_utc(8000) == pytest.approx(T0 + 0.5)
    assert rb.index_to_utc(24_000) == pytest.approx(T0 + 5.5)
    assert rb.utc_to_index(T0 + 0.5) == 8000
    assert rb.utc_to_index(T0 + 5.5) == 24_000
    # A UTC inside the gap maps to the boundary.
    assert rb.utc_to_index(T0 + 3.0) == 16_000


def test_sub_tolerance_jitter_does_not_create_anchors():
    rb = RingBuffer(10, SR, anchor_tolerance_s=0.001)
    rb.write(np.zeros(320, np.float32), T0)
    rb.write(np.zeros(320, np.float32), T0 + 0.02 + 0.0004)
    assert len(rb.anchors) == 1


def test_prune_keeps_covering_anchor():
    rb = RingBuffer(1.0, SR)
    rb.write(np.zeros(SR, np.float32), T0)
    rb.write(np.zeros(SR // 2, np.float32), T0 + 10)
    rb.write(np.zeros(SR // 2, np.float32), T0 + 20)
    # start = 16000 → anchor at 16000 (T0+10) covers it; the T0 anchor is pruned.
    assert rb.anchors[0] == (16_000, T0 + 10)
    assert rb.index_to_utc(rb.start) == pytest.approx(T0 + 10)


def test_shift_utc():
    rb = RingBuffer(10, SR)
    rb.write(np.zeros(SR, np.float32), T0)
    rb.shift_utc(-0.75)
    assert rb.index_to_utc(0) == pytest.approx(T0 - 0.75)
    # continuation after the shift does not create a spurious anchor
    rb.write(np.zeros(10, np.float32), T0 - 0.75 + 1.0)
    assert len(rb.anchors) == 1
