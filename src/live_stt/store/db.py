"""PostgreSQL access (asyncpg 0.32.0) and the plain-SQL migration runner.

Migrations: ``store/migrations/NNN_name.sql`` applied in order, each in a transaction, recorded
in ``schema_migrations``. A session-level advisory lock serialises concurrent runners.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import asyncpg

from live_stt.logging import get_logger
from live_stt.settings import ChannelConfig

log = get_logger(__name__)
MIGRATIONS = Path(__file__).parent / "migrations"
_LOCK_KEY = 0x5354_5401  # arbitrary constant


async def _init_conn(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec("jsonb", encoder=lambda v: json.dumps(v, ensure_ascii=False),
                              decoder=json.loads, schema="pg_catalog")


async def connect_pool(dsn: str, min_size: int = 1, max_size: int = 5) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size, init=_init_conn,
                                     command_timeout=30)


async def migrate(dsn: str) -> list[str]:
    applied: list[str] = []
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("SELECT pg_advisory_lock($1)", _LOCK_KEY)
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {r["version"] for r in await conn.fetch("SELECT version FROM schema_migrations")}
        for f in sorted(MIGRATIONS.glob("*.sql")):
            if f.stem in done:
                continue
            async with conn.transaction():
                await conn.execute(f.read_text(encoding="utf-8"))
                await conn.execute("INSERT INTO schema_migrations(version) VALUES ($1)", f.stem)
            applied.append(f.stem)
            log.info("migration_applied", version=f.stem)
        now = datetime.now(timezone.utc)
        await conn.execute("SELECT ensure_asr_partition($1)", now)
        nxt = now.replace(day=28)  # day 28 + 5 days is always next month
        from datetime import timedelta
        await conn.execute("SELECT ensure_asr_partition($1)", nxt + timedelta(days=5))
    finally:
        try:
            await conn.execute("SELECT pg_advisory_unlock($1)", _LOCK_KEY)
        finally:
            await conn.close()
    return applied


async def upsert_channels(pool: asyncpg.Pool, channels: list[ChannelConfig]) -> None:
    async with pool.acquire() as conn:
        for c in channels:
            await conn.execute(
                """INSERT INTO channels (id, name, url, audio_track, language, glossary, enabled)
                   VALUES ($1,$2,$3,$4,$5,$6,$7)
                   ON CONFLICT (id) DO NOTHING""",
                c.id, c.name, c.url, c.audio_track.model_dump(), c.language, c.glossary_path,
                c.enabled,
            )
