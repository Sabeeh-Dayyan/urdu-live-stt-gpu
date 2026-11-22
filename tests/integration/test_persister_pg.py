import asyncio
import time

import asyncpg
import pytest

from live_stt.bus.redis_streams import RedisStreamsBus
from live_stt.store.db import migrate
from live_stt.store.persister import Persister
from tests.integration.conftest import requires_pg, requires_redis

pytestmark = [requires_pg, requires_redis]
T0 = 1_763_000_000.0  # Nov 2025


def ev(ch, t0, typ="final", text="سلام دنیا", **kw):
    sid = f"{ch}:{int(round(t0 * 1000))}"
    return {"type": typ, "channel_id": ch, "segment_id": sid, "start_utc": t0, "end_utc": t0 + 2,
            "text_raw": text, "text_norm": text, "words": [{"w": "سلام", "s": t0, "e": t0 + .5}],
            "flags": kw.get("flags", []), "model": kw.get("model", "m"),
            "timings": {"latency_s": 1.0}}


async def run_persister(settings, seconds=1.5):
    p = Persister(settings)
    task = asyncio.create_task(p.run())
    await asyncio.sleep(seconds)
    p.stop()
    await asyncio.wait_for(task, 10)
    return p


async def rows(dsn, sql, *a):
    c = await asyncpg.connect(dsn)
    try:
        return await c.fetch(sql, *a)
    finally:
        await c.close()


def test_migrate_idempotent_and_partitions(fresh_db):
    first = asyncio.run(migrate(fresh_db))
    second = asyncio.run(migrate(fresh_db))
    from live_stt.store.db import MIGRATIONS

    expected = sorted(f.stem for f in MIGRATIONS.glob("*.sql"))
    assert first == expected and first[0] == "001_init" and second == []
    parts = asyncio.run(rows(fresh_db, "SELECT relname FROM pg_class WHERE relname LIKE "
                                       "'asr_segments_20%' ORDER BY 1"))
    assert len(parts) >= 2


def test_concurrent_migrations_do_not_conflict(fresh_db):
    async def go():
        return await asyncio.gather(*[migrate(fresh_db) for _ in range(4)])

    res = asyncio.run(go())
    from live_stt.store.db import MIGRATIONS

    n = len(list(MIGRATIONS.glob("*.sql")))
    assert sum(len(r) for r in res) == n  # each migration applied exactly once across 4 racers


def test_upsert_semantics_and_events(settings, fresh_db):
    b = RedisStreamsBus(settings.bus.redis_url)
    ch = settings.channels[0].id
    tr, sysst = settings.bus.stream_transcripts, settings.bus.stream_system
    b.publish(tr, ev(ch, T0, text="fast text"))
    b.publish(tr, ev(ch, T0, "refined", text="refined text", model="large-v3"))
    b.publish(tr, ev(ch, T0, text="late fast must not win"))  # replay / late duplicate
    b.publish(tr, ev(ch, T0 + 10, text="second"))
    b.publish(tr, {"type": "refine_skipped", "channel_id": ch,
                   "segment_id": f"{ch}:{int(round((T0 + 10) * 1000))}",
                   "flags": ["refine_skipped:refine_lag"]})
    b.publish(tr, ev(ch, T0 + 20, "partial", text="partials are not stored"))
    b.publish(tr, ev(ch, T0 + 30, text=""))  # guard-dropped FAST → not stored
    b.publish(tr, {"type": "final", "garbage": True})  # malformed → skipped, not crash
    b.publish(sysst, {"type": "discontinuity", "channel_id": ch, "at_utc": T0,
                      "details": {"jump_s": -24.4}})
    b.publish(sysst, {"type": "broken"})  # malformed system event
    asyncio.run(run_persister(settings))
    r = asyncio.run(rows(fresh_db, "SELECT segment_id, pass, text_raw, flags, model FROM "
                                   "asr_segments ORDER BY start_utc"))
    assert [(x["pass"], x["text_raw"]) for x in r] == [("refined", "refined text"),
                                                       ("fast", "second")]
    assert r[0]["model"] == "large-v3"
    assert r[1]["flags"] == ["refine_skipped:refine_lag"]
    e = asyncio.run(rows(fresh_db, "SELECT type, details FROM stream_events"))
    assert [x["type"] for x in e] == ["discontinuity"]
    # everything acked
    pend = b.r.xpending(tr, "persister")
    assert pend["pending"] == 0


def test_replay_is_idempotent(settings, fresh_db):
    b = RedisStreamsBus(settings.bus.redis_url)
    ch = settings.channels[0].id
    for _ in range(3):
        b.publish(settings.bus.stream_transcripts, ev(ch, T0, text="same"))
    asyncio.run(run_persister(settings))
    r = asyncio.run(rows(fresh_db, "SELECT count(*) AS n FROM asr_segments"))
    assert r[0]["n"] == 1


def test_bulk_throughput(settings, fresh_db):
    b = RedisStreamsBus(settings.bus.redis_url)
    ch = settings.channels[0].id
    pipe = b.r.pipeline()
    from live_stt.bus.base import encode
    for i in range(3000):
        pipe.xadd(settings.bus.stream_transcripts, encode(ev(ch, T0 + i * 3, text=f"جملہ {i}")))
    pipe.execute()
    t = time.time()
    asyncio.run(run_persister(settings, seconds=6))
    r = asyncio.run(rows(fresh_db, "SELECT count(*) AS n FROM asr_segments"))
    assert r[0]["n"] == 3000, f"only {r[0]['n']} rows in {time.time() - t:.1f}s"


def test_trigram_search_index_used(settings, fresh_db):
    asyncio.run(migrate(fresh_db))
    async def explain():
        c = await asyncpg.connect(fresh_db)
        try:
            await c.execute("SET enable_seqscan = off")
            return await c.fetch("EXPLAIN SELECT * FROM asr_segments WHERE text_norm ILIKE '%پاکستان%'")
        finally:
            await c.close()

    plan = asyncio.run(explain())
    assert "trgm" in " ".join(x[0] for x in plan) or "Bitmap" in " ".join(x[0] for x in plan)


def test_poison_events_do_not_wedge(settings, fresh_db):
    b = RedisStreamsBus(settings.bus.redis_url)
    ch = settings.channels[0].id
    tr = settings.bus.stream_transcripts
    bad = [
        ev(ch, T0, text="nul\x00inside"),
        dict(ev(ch, T0 + 3), start_utc=1e20),
        dict(ev(ch, T0 + 6), start_utc="not a number", segment_id="x"),
        dict(ev(ch, T0 + 9), words=[{"w": "x\x00", "s": float("nan")}]),
        dict(ev(ch, T0 + 12), avg_logprob="NaN", timings={"latency_s": "slow"}),
        dict(ev(ch, T0 + 15), flags=None, words="not a list"),
    ]
    for e in bad:
        b.publish(tr, e)
    b.publish(tr, ev(ch, T0 + 100, text="healthy row after poison"))
    asyncio.run(run_persister(settings, seconds=2))
    r = asyncio.run(rows(fresh_db, "SELECT text_raw FROM asr_segments ORDER BY start_utc"))
    texts = [x["text_raw"] for x in r]
    assert "healthy row after poison" in texts
    assert "nulinside" in texts
    assert b.r.xpending(tr, "persister")["pending"] == 0
