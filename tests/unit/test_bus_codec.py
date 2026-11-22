import numpy as np

from live_stt.bus.base import InMemoryBus, decode, encode, f32_to_pcm16, pcm16_to_f32


def test_roundtrip_with_pcm_and_unicode():
    m = {"text": "پاکستان ✓", "n": 1.5, "l": [1, 2], "pcm": b"\x00\x01\x02\x03"}
    assert decode(encode(m)) == m


def test_pcm_conversion_robust():
    x = np.array([0.0, 0.5, -0.5, 1.5, -2.0, np.nan, np.inf, -np.inf], np.float32)
    y = pcm16_to_f32(f32_to_pcm16(x))
    assert np.all(np.isfinite(y)) and np.all(np.abs(y) <= 1.0)
    assert abs(y[1] - 0.5) < 1e-3 and y[5] == 0.0
    assert len(pcm16_to_f32(b"\x01\x02\x03")) == 1  # odd byte count tolerated
    assert len(pcm16_to_f32(b"")) == 0


def test_inmemory_semantics_pending_redelivery():
    b = InMemoryBus()
    ids = [b.publish("s", {"i": i}) for i in range(3)]
    got = b.consume("s", "g", "c1", count=2)
    assert [m["i"] for _, m in got] == [0, 1]
    b.ack("s", "g", [got[0][0]])
    assert b.depth("s", "g") == 2  # 1 pending + 1 undelivered
    b.simulate_restart("s", "g", "c1")
    again = b.consume("s", "g", "c1", count=10)
    assert [m["i"] for _, m in again] == [1]
    rest = b.consume("s", "g", "c1", count=10)
    assert [m["i"] for _, m in rest] == [2]
    assert b.consume("s", "g", "c1", count=10, block_ms=10) == []
    assert len(ids) == 3
