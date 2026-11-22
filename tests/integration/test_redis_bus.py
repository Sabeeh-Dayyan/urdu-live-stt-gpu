import threading
import time

import redis

from live_stt.bus.redis_streams import RedisStreamsBus
from tests.integration.conftest import REDIS, requires_redis

pytestmark = requires_redis


def bus(**kw):
    return RedisStreamsBus(REDIS, **kw)


def test_publish_consume_ack_delete(settings):
    b = bus(delete_on_ack=(settings.bus.stream_fast,))
    s = settings.bus.stream_fast
    ids = [b.publish(s, {"i": i, "pcm": bytes([i]) * 10}) for i in range(5)]
    got = b.consume(s, "g", "c1", count=3, block_ms=100)
    assert [m["i"] for _, m in got] == [0, 1, 2] and got[0][1]["pcm"] == b"\x00" * 10
    b.ack(s, "g", [i for i, _ in got])
    assert b.r.xlen(s) == 2  # acked entries deleted (bounded memory for audio)
    assert b.depth(s, "g") == 2
    assert len(ids) == 5


def test_pending_redelivered_after_restart(settings):
    s = settings.bus.stream_fast
    b1 = bus()
    for i in range(3):
        b1.publish(s, {"i": i})
    got = b1.consume(s, "g", "worker", count=2)
    b1.ack(s, "g", [got[0][0]])  # crash before acking the second
    b2 = bus()  # "restarted" process with the same consumer name
    again = b2.consume(s, "g", "worker", count=10)
    assert [m["i"] for _, m in again] == [1]
    nxt = b2.consume(s, "g", "worker", count=10)
    assert [m["i"] for _, m in nxt] == [2]


def test_claim_stale_from_dead_consumer(settings):
    s = settings.bus.stream_fast
    b = bus()
    b.publish(s, {"i": 1})
    b.consume(s, "g", "dead", count=1)
    time.sleep(0.05)
    claimed = b.claim_stale(s, "g", "alive", min_idle_ms=10)
    assert [m["i"] for _, m in claimed] == [1]


def test_blocking_consume_wakes_on_publish(settings):
    s = settings.bus.stream_fast
    b = bus()
    b.ensure_group(s, "g")
    threading.Timer(0.2, lambda: bus().publish(s, {"x": 1})).start()
    t = time.time()
    got = b.consume(s, "g", "c", count=1, block_ms=3000)
    assert got and time.time() - t < 2.5


def test_survives_stream_deletion_nogroup(settings):
    s = settings.bus.stream_fast
    b = bus()
    b.publish(s, {"i": 1})
    b.consume(s, "g", "c", count=1)
    redis.Redis.from_url(REDIS).delete(s)  # e.g. FLUSHDB / data loss on the server
    b._pending_done.add((s, "g", "c"))
    assert b.consume(s, "g", "c", count=1, block_ms=10) == []
    b.publish(s, {"i": 2})
    assert [m["i"] for _, m in b.consume(s, "g", "c", count=1, block_ms=100)] == [2]


def test_maxlen_cap(settings):
    s = settings.bus.stream_system
    b = bus(maxlen=50)
    for i in range(2000):
        b.publish(s, {"i": i})
    assert b.r.xlen(s) < 300  # approximate trimming


def test_kv_and_broadcast(settings):
    b = bus()
    b.set_value("test:kv", {"a": "پاکستان"}, ttl_s=5)
    assert b.get_value("test:kv") == {"a": "پاکستان"}
    assert b.get_value("test:missing") is None
    ps = redis.Redis.from_url(REDIS).pubsub(ignore_subscribe_messages=True)
    ps.subscribe(settings.bus.pubsub_live)
    time.sleep(0.1)
    b.broadcast(settings.bus.pubsub_live, {"t": 1, "pcm": b"x"})
    m = None
    for _ in range(20):
        m = ps.get_message(timeout=0.2)
        if m:
            break
    assert m and b"pcm" not in m["data"]


def test_retry_gives_up_on_dead_server():
    b = RedisStreamsBus("redis://localhost:6399/0", max_retry_s=0.5)  # nothing listens
    t = time.time()
    try:
        b.publish("x", {"a": 1})
        raised = False
    except Exception:
        raised = True
    assert raised and time.time() - t < 10
    assert b.ping() is False
