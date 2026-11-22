"""Persister: transcripts.events + system.events (Redis Streams) → PostgreSQL.

* Batches rows; flushes every ``flush_ms`` or ``flush_rows``.
* Upsert on (channel_id, segment_id, start_utc). ``refined`` supersedes ``fast``; a late ``fast``
  never overwrites a ``refined`` row. ``refine_skipped`` appends a flag to the fast row.
* Partials are not persisted (they are superseded by finals within seconds).
* Acks only after the DB transaction commits (at-least-once; upserts make replays idempotent).
* DB outage: retries with backoff while holding the un-acked batch; Redis keeps the rest.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import time
from datetime import datetime, timezone

import asyncpg

from live_stt import metrics as M
from live_stt.bus.redis_streams import AsyncRedisStreams
from live_stt.logging import get_logger
from live_stt.settings import Settings
from live_stt.store.db import connect_pool, migrate, upsert_channels
from live_stt.text.confidence import mean_confidence

log = get_logger(__name__)
GROUP = "persister"

UPSERT_SQL = """
INSERT INTO asr_segments (channel_id, segment_id, start_utc, end_utc, pass, model, text_raw,
                          text_norm, words, avg_logprob, no_speech_prob, flags, latency_s,
                          confidence)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
ON CONFLICT (channel_id, segment_id, start_utc) DO UPDATE SET
    end_utc = EXCLUDED.end_utc, pass = EXCLUDED.pass, model = EXCLUDED.model,
    text_raw = EXCLUDED.text_raw, text_norm = EXCLUDED.text_norm, words = EXCLUDED.words,
    avg_logprob = EXCLUDED.avg_logprob, no_speech_prob = EXCLUDED.no_speech_prob,
    confidence = EXCLUDED.confidence,
    flags = COALESCE((SELECT array_agg(DISTINCT f ORDER BY f)
                     FROM unnest(asr_segments.flags || EXCLUDED.flags) f), '{}'),
    latency_s = COALESCE(asr_segments.latency_s, EXCLUDED.latency_s),
    updated_at = now()
WHERE asr_segments.pass <> 'refined' OR EXCLUDED.pass = 'refined'
"""

FLAG_SQL = """
UPDATE asr_segments SET flags = array_append(flags, $3), updated_at = now()
WHERE channel_id = $1 AND segment_id = $2 AND NOT ($3 = ANY(flags))
"""

EVENT_SQL = "INSERT INTO stream_events (channel_id, type, at_utc, details) VALUES ($1,$2,$3,$4)"


def _ts(x) -> datetime:
    v = float(x)
    if not (0 < v < 32503680000):  # 1970 … year 3000
        raise ValueError(f"timestamp out of range: {x!r}")
    return datetime.fromtimestamp(v, tz=timezone.utc)


def segment_row(ev: dict) -> tuple | None:
    """Map a transcript event to an asr_segments row tuple (None → not persisted)."""
    t = ev.get("type")
    if t not in ("final", "refined"):
        return None
    text = ev.get("text_raw") or ""
    if t == "final" and not text.strip():
        return None  # guards dropped everything; nothing to store for FAST
    segment_id = ev.get("segment_id")
    try:
        start = _ts(_segment_start(segment_id, ev["start_utc"]))
        end = _ts(ev.get("end_utc", ev["start_utc"]))
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return None
    lat = (ev.get("timings") or {}).get("latency_s")
    if not isinstance(lat, (int, float)):
        lat = None
    words = ev.get("words") if isinstance(ev.get("words"), list) else []
    return (
        _clean(ev["channel_id"]), _clean(segment_id), start, max(end, start),
        "refined" if t == "refined" else "fast", _clean(ev.get("model")), _clean(text),
        _clean(ev.get("text_norm") or ""), _clean_obj(words), _num(ev.get("avg_logprob")),
        _num(ev.get("no_speech_prob")), [_clean(str(f)) for f in (ev.get("flags") or [])],
        float(lat) if lat is not None else None,
        mean_confidence(words),
    )


def _clean(x):
    """PostgreSQL text/jsonb cannot hold NUL characters."""
    return x.replace("\x00", "") if isinstance(x, str) else x


def _clean_obj(o):
    if isinstance(o, str):
        return _clean(o)
    if isinstance(o, list):
        return [_clean_obj(v) for v in o]
    if isinstance(o, dict):
        return {_clean(str(k)): _clean_obj(v) for k, v in o.items()}
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None  # NaN/inf are not valid JSON
    return o


def _num(x):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _segment_start(segment_id: str | None, fallback: float) -> float:
    """The row key uses the segment's FAST start (encoded in segment_id as ms), so the refined
    event (whose audio includes context) upserts the same row."""
    try:
        return int(str(segment_id).rsplit(":", 1)[1]) / 1000.0
    except (IndexError, ValueError):
        return float(fallback)


class Persister:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.consumer = f"persister-{socket.gethostname()}"
        self.streams = [settings.bus.stream_transcripts, settings.bus.stream_system]
        self._stop = asyncio.Event()
        self.written = 0

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        await self._retry(lambda: migrate(self.s.db.dsn), "migrate")
        pool = await self._retry(lambda: connect_pool(self.s.db.dsn), "connect")
        await upsert_channels(pool, self.s.channels)
        bus = AsyncRedisStreams(self.s.bus.redis_url)
        last_part = 0.0
        try:
            while not self._stop.is_set():
                try:
                    batch = await self._collect(bus)
                    if batch:
                        await self._write(pool, batch)
                        for stream in self.streams:
                            ids = [i for s, i, _ in batch if s == stream]
                            await bus.ack(stream, GROUP, ids)
                    if time.time() - last_part > 3600:
                        last_part = time.time()
                        async with pool.acquire() as c:
                            await c.execute("SELECT ensure_asr_partition(now() + interval '7 days')")
                except (OSError, asyncpg.PostgresError, asyncpg.InterfaceError,
                        ConnectionError) as e:
                    log.error("persister_error", error=str(e))
                    bus.reset_groups()  # also re-read our pending after a Redis restart
                    await asyncio.sleep(1.0)
                except Exception as e:  # noqa: BLE001 - redis errors etc.
                    log.error("persister_error", error=f"{type(e).__name__}: {e}")
                    bus.reset_groups()
                    await asyncio.sleep(1.0)
        finally:
            await bus.close()
            await pool.close()

    async def _retry(self, fn, what: str):
        delay = 0.5
        while True:
            try:
                return await fn()
            except Exception as e:  # noqa: BLE001
                if self._stop.is_set():
                    raise
                log.warning("persister_retry", what=what, error=str(e), wait_s=delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10)

    async def _collect(self, bus: AsyncRedisStreams) -> list:
        deadline = time.monotonic() + self.s.db.flush_ms / 1000
        batch: list = []
        while len(batch) < self.s.db.flush_rows:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            got = await bus.consume(self.streams, GROUP, self.consumer,
                                    count=self.s.db.flush_rows - len(batch),
                                    block_ms=max(1, int(left * 1000)))
            if not got and batch:
                break
            batch += got
        return batch

    async def _write_one_by_one(self, pool, rows, flags, events) -> None:
        async with pool.acquire() as conn:
            for r in rows:
                await _safe_exec(conn, UPSERT_SQL, r)
            for f in flags:
                await _safe_exec(conn, FLAG_SQL, f)
            for e in events:
                await _safe_exec(conn, EVENT_SQL, e)

    async def _write(self, pool: asyncpg.Pool, batch: list) -> None:
        rows, flags, events = [], [], []
        for stream, _id, ev in batch:
            if stream == self.s.bus.stream_system:
                try:
                    events.append((_clean(ev["channel_id"]), _clean(ev["type"]),
                                   _ts(ev["at_utc"]), _clean_obj(ev.get("details") or {})))
                except (KeyError, TypeError, ValueError, OverflowError, OSError):
                    log.warning("bad_system_event", payload=str(ev)[:200])
                continue
            if ev.get("type") == "refine_skipped":
                for f in ev.get("flags") or ["refine_skipped"]:
                    flags.append((ev.get("channel_id"), ev.get("segment_id"), f))
                continue
            r = segment_row(ev)
            if r:
                rows.append(r)
        # Coalesce: last write per key wins, refined beats fast within the batch.
        dedup: dict[tuple, tuple] = {}
        for r in rows:
            k = (r[0], r[1], r[2])
            if k not in dedup or not (dedup[k][4] == "refined" and r[4] == "fast"):
                dedup[k] = r
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    if dedup:
                        await conn.executemany(UPSERT_SQL, list(dedup.values()))
                    if flags:
                        await conn.executemany(FLAG_SQL, flags)
                    if events:
                        await conn.executemany(EVENT_SQL, events)
        except (asyncpg.DataError, asyncpg.IntegrityConstraintViolationError,
                asyncpg.SyntaxOrAccessError, ValueError, TypeError) as e:
            # A bad row must not wedge the pipeline: write row by row, skip and log offenders.
            log.error("persister_batch_rejected", error=str(e), rows=len(dedup))
            await self._write_one_by_one(pool, list(dedup.values()), flags, events)
        self.written += len(dedup)
        M.ROWS_WRITTEN.labels("asr_segments").inc(len(dedup))
        M.ROWS_WRITTEN.labels("stream_events").inc(len(events))


async def _safe_exec(conn, sql: str, args: tuple) -> bool:
    try:
        async with conn.transaction():
            await conn.execute(sql, *args)
        return True
    except (asyncpg.DataError, asyncpg.IntegrityConstraintViolationError,
            asyncpg.SyntaxOrAccessError, ValueError, TypeError) as e:
        log.error("persister_row_skipped", error=str(e), key=str(args[:2])[:200])
        return False


def run_persister(settings: Settings) -> None:
    p = Persister(settings)

    async def main():
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, p.stop)
            except (NotImplementedError, RuntimeError):
                pass
        await p.run()

    log.info("persister_start", pid=os.getpid())
    asyncio.run(main())
