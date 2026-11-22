"""The same contract tests run against every Bus implementation."""

import inspect

import pytest

from live_stt.bus.base import Bus, InMemoryBus
from live_stt.bus.kafka import KafkaBus
from live_stt.bus.redis_streams import RedisStreamsBus
from tests.integration.conftest import REDIS, _redis_ok


def make(kind):
    if kind == "memory":
        return InMemoryBus()
    if not _redis_ok():
        pytest.skip("redis not running")
    return RedisStreamsBus(REDIS)


@pytest.mark.parametrize("kind", ["memory", "redis"])
def test_contract_delivery_and_ack(kind, settings):
    b = make(kind)
    s = settings.bus.stream_fast
    for i in range(5):
        b.publish(s, {"i": i, "pcm": b"\x01\x02"})
    seen = []
    for _ in range(10):
        got = b.consume(s, "g", "c", count=2, block_ms=50)
        if not got:
            break
        seen += [m["i"] for _, m in got]
        assert all(m["pcm"] == b"\x01\x02" for _, m in got)
        b.ack(s, "g", [i for i, _ in got])
    assert seen == [0, 1, 2, 3, 4]
    assert b.depth(s, "g") == 0
    # second group sees everything independently
    assert [m["i"] for _, m in b.consume(s, "g2", "c", count=10, block_ms=50)] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("cls", [InMemoryBus, RedisStreamsBus, KafkaBus])
def test_all_implementations_cover_the_interface(cls):
    abstract = {n for n, v in inspect.getmembers(Bus) if getattr(v, "__isabstractmethod__", False)}
    assert abstract <= set(cls.__dict__) | {n for b in cls.__mro__[1:-1] if b is not Bus
                                             for n in b.__dict__}


def test_kafka_stub_refuses_to_construct():
    with pytest.raises(NotImplementedError):
        KafkaBus("localhost:9092")
