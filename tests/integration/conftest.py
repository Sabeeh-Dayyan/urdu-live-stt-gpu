import asyncio
import os
import uuid

import pytest

from live_stt.settings import ChannelConfig, Settings

REDIS = os.environ.get("STT_TEST_REDIS", "redis://localhost:6379/15")  # dedicated test DB index
PG_ADMIN = os.environ.get("STT_TEST_PG_ADMIN", "postgresql://stt:stt@localhost:5432/stt")
PG_TEST = os.environ.get("STT_TEST_PG", "postgresql://stt:stt@localhost:5432/stt_test")


def _redis_ok():
    try:
        import redis

        return redis.Redis.from_url(REDIS, socket_connect_timeout=1).ping()
    except Exception:
        return False


def _pg_ok():
    import asyncpg

    async def go():
        c = await asyncpg.connect(PG_ADMIN, timeout=2)
        if not await c.fetchval("SELECT 1 FROM pg_database WHERE datname='stt_test'"):
            await c.execute("CREATE DATABASE stt_test")
        await c.close()

    try:
        asyncio.run(go())
        return True
    except Exception:
        return False


requires_redis = pytest.mark.skipif(not _redis_ok(), reason="redis not running (make infra)")
requires_pg = pytest.mark.skipif(not _pg_ok(), reason="postgres not running (make infra)")


@pytest.fixture
def settings():
    s = Settings()
    tag = uuid.uuid4().hex[:8]
    s.bus.redis_url = REDIS
    for f in ("stream_fast", "stream_refine", "stream_transcripts", "stream_system", "pubsub_live"):
        setattr(s.bus, f, f"test.{tag}.{getattr(s.bus, f)}")
    s.db.dsn = PG_TEST
    s.channels = [ChannelConfig(id=f"t{tag}", url="udp://127.0.0.1:1")]
    return s


@pytest.fixture
def fresh_db():
    import asyncpg

    async def reset():
        c = await asyncpg.connect(PG_TEST)
        await c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        await c.close()

    asyncio.run(reset())
    return PG_TEST
