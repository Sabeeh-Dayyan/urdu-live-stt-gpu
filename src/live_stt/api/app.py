"""FastAPI app: REST + WebSocket live captions + static UI (fastapi 0.165.0, uvicorn 0.54.0).

The API never does CPU-heavy work. PostgreSQL and Redis are optional at start-up: endpoints that
need an unavailable dependency return 503 and the connection is retried on the next request.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import asyncpg
import redis.asyncio as aioredis
from fastapi import (Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
from pydantic import BaseModel

from live_stt.logging import get_logger
from live_stt.settings import AudioTrack, ChannelConfig, Settings, resolve_path
from live_stt.store.db import connect_pool
from live_stt.text.sentences import (FORMATS, assemble, get_tz, keep, render,
                                     split_sentences)
from live_stt.text.urdu_normalize import normalize

log = get_logger(__name__)
WS_CLIENTS = Gauge("stt_api_ws_clients", "Connected WebSocket clients")
WS_DROPPED = Counter("stt_api_ws_dropped_total", "WebSocket clients dropped (slow/broken)")
MAX_ROWS = 1000
MAX_EXPORT_ROWS = 50_000
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
HLS_FILE = re.compile(r"^(index\.m3u8|seg_\d{8}\.ts)$")
SEG_COLS = """channel_id, segment_id, start_utc, end_utc, pass, model, text_raw, text_norm,
              words, avg_logprob, no_speech_prob, flags, latency_s, confidence"""


def parse_time(v: str | None) -> datetime | None:
    if v is None or v == "":
        return None
    try:
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    except ValueError:
        pass
    try:
        d = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(400, f"invalid time {v!r} (ISO-8601 or epoch seconds)")
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


class ChannelPatch(BaseModel):
    model_config = {"extra": "forbid"}
    name: str | None = None
    url: str | None = None
    audio_track: AudioTrack | None = None
    language: str | None = None
    glossary_path: str | None = None
    enabled: bool | None = None


class State:
    def __init__(self, s: Settings) -> None:
        self.s = s
        self.pool: asyncpg.Pool | None = None
        self.redis = aioredis.Redis.from_url(s.bus.redis_url, socket_connect_timeout=3)
        self._pool_lock = asyncio.Lock()
        self._last_try = 0.0

    async def db(self) -> asyncpg.Pool:
        if self.pool is not None:
            return self.pool
        async with self._pool_lock:
            if self.pool is None:
                if time.time() - self._last_try < 2:
                    raise HTTPException(503, "database unavailable")
                self._last_try = time.time()
                try:
                    self.pool = await asyncio.wait_for(connect_pool(self.s.db.dsn), 5)
                except Exception as e:  # noqa: BLE001
                    log.warning("db_unavailable", error=str(e))
                    raise HTTPException(503, "database unavailable") from e
        return self.pool


def _row(r) -> dict:
    d = dict(r)
    for k in ("start_utc", "end_utc", "at_utc", "created_at", "updated_at"):
        if isinstance(d.get(k), datetime):
            d[k] = d[k].timestamp()
    if "id" in d and not isinstance(d["id"], (str, int)):
        d["id"] = str(d["id"])
    for k in ("words", "audio_track", "details"):
        if isinstance(d.get(k), str):
            d[k] = json.loads(d[k])
    return d


def create_app(s: Settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.st = State(s)
        yield
        st: State = app.state.st
        if st.pool:
            await st.pool.close()
        await st.redis.aclose()

    app = FastAPI(title="live-stt", version="0.1.0", lifespan=lifespan)

    def st(req) -> State:
        return req.app.state.st

    def require_token(authorization: str | None = Header(None)) -> None:
        """Write endpoints: Bearer token when api.token is configured (constant-time compare)."""
        want = s.api.token
        if not want:
            return
        got = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(got.encode(), want.encode()):
            raise HTTPException(401, "missing or invalid API token",
                                headers={"WWW-Authenticate": "Bearer"})

    async def q(req: Request, sql: str, *args):
        pool = await st(req).db()
        try:
            return await pool.fetch(sql, *args)
        except (OSError, asyncpg.InterfaceError, asyncpg.CannotConnectNowError) as e:
            st(req).pool = None
            raise HTTPException(503, "database unavailable") from e

    # ------------------------------------------------------------------ health
    @app.get("/health")
    async def health(req: Request):
        out: dict = {"t": time.time()}
        r = st(req).redis
        try:
            out["redis"] = bool(await asyncio.wait_for(r.ping(), 2))
            keys = [k async for k in r.scan_iter("stt:health:*", count=200)]
            vals = await r.mget(keys) if keys else []
            out["components"] = {k.decode().removeprefix("stt:health:"): json.loads(v)
                                 for k, v in zip(keys, vals) if v}
        except Exception as e:  # noqa: BLE001
            out["redis"] = False
            out["redis_error"] = str(e)
        try:
            pool = await st(req).db()
            out["db"] = (await asyncio.wait_for(pool.fetchval("SELECT 1"), 2)) == 1
        except Exception:  # noqa: BLE001
            out["db"] = False
            st(req).pool = None
        ok = out["redis"] and out["db"]
        out["status"] = "ok" if ok else "degraded"
        return JSONResponse(out, status_code=200 if ok else 503)

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ---------------------------------------------------------------- channels
    @app.get("/channels")
    async def list_channels(req: Request):
        rows = await q(req, "SELECT * FROM channels ORDER BY id")
        return [_row(r) for r in rows]

    @app.post("/channels", status_code=201, dependencies=[Depends(require_token)])
    async def create_channel(req: Request, ch: ChannelConfig):
        pool = await st(req).db()
        try:
            r = await pool.fetchrow(
                """INSERT INTO channels (id, name, url, audio_track, language, glossary, enabled)
                   VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING *""",
                ch.id, ch.name, ch.url, ch.audio_track.model_dump(), ch.language,
                ch.glossary_path, ch.enabled)
        except asyncpg.UniqueViolationError:
            raise HTTPException(409, f"channel {ch.id!r} exists")
        return _row(r)

    @app.patch("/channels/{cid}", dependencies=[Depends(require_token)])
    async def patch_channel(req: Request, cid: str, p: ChannelPatch):
        fields = p.model_dump(exclude_unset=True)
        if "glossary_path" in fields:
            fields["glossary"] = fields.pop("glossary_path")
        if "audio_track" in fields and fields["audio_track"] is not None:
            fields["audio_track"] = AudioTrack.model_validate(fields["audio_track"]).model_dump()
        if not fields:
            raise HTTPException(400, "no fields to update")
        cols = list(fields)
        sets = ", ".join(f"{c} = ${i + 2}" for i, c in enumerate(cols))
        pool = await st(req).db()
        r = await pool.fetchrow(
            f"UPDATE channels SET {sets}, updated_at = now() WHERE id = $1 RETURNING *",
            cid, *[fields[c] for c in cols])
        if r is None:
            raise HTTPException(404, f"channel {cid!r} not found")
        return _row(r)

    # ------------------------------------------------------------- transcripts
    @app.get("/channels/{cid}/transcripts")
    async def transcripts(req: Request, cid: str, from_: str | None = Query(None, alias="from"),
                          to: str | None = None, pass_: str = Query("any", alias="pass"),
                          limit: int = Query(200, ge=1, le=MAX_ROWS),
                          min_conf: float = Query(0.0, ge=0.0, le=1.0)):
        if pass_ not in ("any", "fast", "refined"):
            raise HTTPException(400, "pass must be any|fast|refined")
        rows = await _segments(req, cid, from_, to, pass_, min_conf, limit)
        out = []
        for d in rows:
            d["sentences"] = split_sentences(d)
            out.append(d)
        return out

    async def _segments(req, cid, from_, to, pass_, min_conf, limit) -> list[dict]:
        t0, t1 = parse_time(from_), parse_time(to)
        rows = await q(
            req,
            f"""SELECT {SEG_COLS}
               FROM asr_segments
               WHERE channel_id = $1
                 AND ($2::timestamptz IS NULL OR start_utc >= $2)
                 AND ($3::timestamptz IS NULL OR start_utc < $3)
                 AND ($4 = 'any' OR pass = $4)
                 AND ($5::float8 <= 0 OR confidence >= $5::float8)
               ORDER BY start_utc DESC LIMIT $6""",
            cid, t0, t1, pass_, min_conf, limit)
        return [_row(r) for r in reversed(rows)]

    @app.get("/channels/{cid}/export")
    async def export(req: Request, cid: str, format: str = "srt",
                     from_: str | None = Query(None, alias="from"), to: str | None = None,
                     pass_: str = Query("any", alias="pass"),
                     min_conf: float = Query(0.0, ge=0.0, le=1.0), tz: str = "UTC"):
        """Sentence-level transcript download: srt | vtt | csv | txt | json.

        ``min_conf`` filters *sentences* by model confidence (mean word probability)."""
        if format not in FORMATS:
            raise HTTPException(400, f"format must be one of {sorted(FORMATS)}")
        if pass_ not in ("any", "fast", "refined"):
            raise HTTPException(400, "pass must be any|fast|refined")
        try:
            zone = get_tz(tz)
        except ValueError as e:
            raise HTTPException(400, str(e))
        rows = await _segments(req, cid, from_, to, pass_, 0.0, MAX_EXPORT_ROWS)
        sents = [x for x in assemble(rows) if keep(x, min_conf)]  # across segment cuts
        t0 = parse_time(from_)
        body = render(sents, format, zone, origin=t0.timestamp() if t0 else None)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        safe = cid if SAFE_ID.match(cid) else "channel"
        return Response(body.encode("utf-8"),
                        media_type=FORMATS[format] + "; charset=utf-8",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{safe}_{stamp}.{format}"',
                                 "X-Sentences": str(len(sents))})

    @app.get("/accuracy")
    async def accuracy(req: Request, dataset: str | None = None):
        """Measured accuracy results (from `stt record-accuracy`)."""
        rows = await q(req, """SELECT recorded_at, source_file, dataset, kind, config, metric,
                                      threshold, value, n FROM accuracy_results
                               WHERE ($1::text IS NULL OR dataset = $1)
                               ORDER BY kind, config, metric, threshold""", dataset)
        return [_row(r) | {"recorded_at": r["recorded_at"].timestamp()} for r in rows]

    # ---------------------------------------------------------------- video
    @app.get("/hls/{cid}/{name}")
    async def hls(cid: str, name: str):
        """Rolling HLS copy of the channel written by its worker (relay.enabled)."""
        if not SAFE_ID.match(cid) or not HLS_FILE.match(name):
            raise HTTPException(404, "not found")
        path = resolve_path(s.relay.dir) / cid / name
        if not path.is_file():
            raise HTTPException(404, "no video for this channel (relay disabled or not started)")
        if name.endswith(".m3u8"):
            return Response(path.read_bytes(), media_type="application/vnd.apple.mpegurl",
                            headers={"Cache-Control": "no-cache"})
        return FileResponse(path, media_type="video/mp2t")

    @app.get("/search")
    async def search(req: Request, q_: str = Query(..., alias="q", min_length=1, max_length=200),
                     channel: str | None = None, from_: str | None = Query(None, alias="from"),
                     to: str | None = None, limit: int = Query(100, ge=1, le=MAX_ROWS)):
        nq = normalize(q_)
        if not nq:
            raise HTTPException(400, "query is empty after normalisation")
        like = "%" + nq.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        rows = await q(
            req,
            """SELECT channel_id, segment_id, start_utc, end_utc, pass, text_raw, flags
               FROM asr_segments
               WHERE text_norm ILIKE $1
                 AND ($2::text IS NULL OR channel_id = $2)
                 AND ($3::timestamptz IS NULL OR start_utc >= $3)
                 AND ($4::timestamptz IS NULL OR start_utc < $4)
               ORDER BY start_utc DESC LIMIT $5""",
            like, channel, parse_time(from_), parse_time(to), limit)
        return {"query": q_, "normalized": nq, "results": [_row(r) for r in rows]}

    @app.get("/channels/{cid}/events")
    async def stream_events(req: Request, cid: str, limit: int = Query(100, ge=1, le=MAX_ROWS)):
        rows = await q(req, "SELECT * FROM stream_events WHERE channel_id=$1 "
                            "ORDER BY at_utc DESC LIMIT $2", cid, limit)
        return [_row(r) for r in rows]

    # --------------------------------------------------------------- websocket
    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket, channels: str | None = None):
        await ws.accept()
        want = {c for c in (channels or "").split(",") if c}
        WS_CLIENTS.inc()
        r = aioredis.Redis.from_url(s.bus.redis_url, socket_connect_timeout=3)
        ps = r.pubsub(ignore_subscribe_messages=True)
        try:
            await ps.subscribe(s.bus.pubsub_live)
            recv = asyncio.create_task(_drain_client(ws))
            while not recv.done():
                msg = await ps.get_message(timeout=1.0)
                if msg is None:
                    continue
                data = msg["data"]
                if want:
                    try:
                        if json.loads(data).get("channel_id") not in want:
                            continue
                    except (ValueError, AttributeError):
                        continue
                try:  # a client that cannot keep up for 5 s is dropped
                    await asyncio.wait_for(ws.send_text(
                        data.decode() if isinstance(data, bytes) else data), 5)
                except (asyncio.TimeoutError, RuntimeError):
                    WS_DROPPED.inc()
                    break
            recv.cancel()
        except WebSocketDisconnect:
            pass
        except Exception as e:  # noqa: BLE001
            log.warning("ws_error", error=str(e))
            try:
                await ws.close(code=1011)
            except Exception:  # noqa: BLE001
                pass
        finally:
            WS_CLIENTS.dec()
            try:
                await ps.aclose()
                await r.aclose()
            except Exception:  # noqa: BLE001
                pass

    # -------------------------------------------------------------------- UI
    @app.get("/")
    async def index():
        return FileResponse(resolve_path("ui/index.html"))

    @app.get("/ui/vendor/{name}")
    async def vendor(name: str):
        if name not in ("hls.min.js",):
            raise HTTPException(404, "not found")
        return FileResponse(resolve_path("ui/vendor") / name, media_type="text/javascript")

    return app


async def _drain_client(ws: WebSocket) -> None:
    """Consume client frames so disconnects are noticed."""
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        return


def run_api(s: Settings) -> None:
    import uvicorn

    uvicorn.run(create_app(s), host=s.api.host, port=s.api.port, log_level="warning",
                ws_ping_interval=20, ws_ping_timeout=20)
