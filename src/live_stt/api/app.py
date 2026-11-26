"""FastAPI app: REST + WebSocket live captions + static UI (fastapi 0.165.0, uvicorn 0.54.0).

The API never does CPU-heavy work. PostgreSQL and Redis are optional at start-up: endpoints that
need an unavailable dependency return 503 and the connection is retried on the next request.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
import redis.asyncio as aioredis
from fastapi import (Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
from pydantic import BaseModel

from live_stt.logging import get_logger
from live_stt.settings import AudioTrack, ChannelConfig, Settings, resolve_path
from live_stt.store.db import connect_pool
from live_stt.text.sentences import (FORMATS, assemble, get_tz, keep, render,
                                     split_sentences)
from live_stt.text.urdu_normalize import normalize
from live_stt import reports as R
from live_stt.ingest import archive as A
from live_stt import uploads as U

log = get_logger(__name__)
WS_CLIENTS = Gauge("stt_api_ws_clients", "Connected WebSocket clients")
WS_DROPPED = Counter("stt_api_ws_dropped_total", "WebSocket clients dropped (slow/broken)")
MAX_ROWS = 1000
MAX_EXPORT_ROWS = 50_000
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
HLS_FILE = re.compile(r"^(index\.m3u8|seg_\d{8}\.ts)$")
UPLOAD_ID = re.compile(r"^up-[0-9a-f]{10}$")
# UI files: browsers revalidate every load (ETag → 304), so an update is never hidden by a cache.
NO_CACHE = {"Cache-Control": "no-cache"}
URL_SCHEMES = ("udp", "rtp", "srt", "rtsp", "rtmp", "rtmps", "http", "https")
UI_PAGES = {"live": "index.html", "archive": "archive.html", "reports": "reports.html",
            "uploads": "uploads.html", "channels": "channels.html", "system": "system.html"}
UI_ASSETS = {"common.js": "text/javascript", "common.css": "text/css"}
SEG_COLS = """channel_id, segment_id, start_utc, end_utc, pass, model, text_raw, text_norm,
              words, avg_logprob, no_speech_prob, flags, latency_s, confidence"""


def bench_label(s) -> str:
    """The `stt bench` label of the live FAST configuration (bench/run_bench.py:Cfg.label), so the
    UI can show the accuracy measured for exactly the configuration that is running."""
    a, pack = s.asr, f" pack{s.asr.pack_s:g}" if s.asr.pack_s else ""
    return (f"{a.fast.model} {a.compute_type} b{a.fast.beam_size} seg{s.streaming.max_segment_s:g} "
            f"t{a.cpu_threads}{pack}")


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


def check_url(url: str) -> str:
    """Channel source URLs set through the API: network stream protocols only (no local files)."""
    from urllib.parse import urlsplit

    u = (url or "").strip()
    scheme = urlsplit(u).scheme.lower()
    if scheme not in URL_SCHEMES or len(u) > 2000:
        raise HTTPException(400, f"url must start with one of {', '.join(s + '://' for s in URL_SCHEMES)}")
    return u


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
        try:  # clip temp files left by a crash during a download
            for f in resolve_path(s.relay.dir).glob(".clip_*.mp4"):
                if time.time() - f.stat().st_mtime > 3600:
                    f.unlink(missing_ok=True)
        except OSError:
            pass
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
        """Redis, database and every process heartbeat (503 when degraded)."""
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
        """Prometheus metrics of the API process."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ---------------------------------------------------------------- channels
    @app.get("/channels")
    async def list_channels(req: Request):
        """All channels (YAML ones are copied into the database at start)."""
        rows = await q(req, "SELECT * FROM channels ORDER BY id")
        return [_row(r) for r in rows]

    @app.post("/channels", status_code=201, dependencies=[Depends(require_token)])
    async def create_channel(req: Request, ch: ChannelConfig):
        """Add a live channel (id, name, url, audio_track, language, enabled); started within ~10 s."""
        if not SAFE_ID.match(ch.id) or ch.id.startswith("up-") or ch.id in (".", ".."):
            raise HTTPException(400, "id: 1-64 of A-Z a-z 0-9 _ . - (not starting with 'up-')")
        ch.url = check_url(ch.url)
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
        """Change a channel (URL, name, audio track, enabled …); its worker restarts within ~10 s."""
        fields = p.model_dump(exclude_unset=True)
        if fields.get("url") is not None:
            fields["url"] = check_url(fields["url"])
        elif "url" in fields:
            raise HTTPException(400, "url cannot be null")
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

    @app.delete("/channels/{cid}", status_code=204, dependencies=[Depends(require_token)])
    async def delete_channel(req: Request, cid: str):
        """Remove an API-managed channel (its worker stops within ~10 s). Transcripts are kept.
        A channel also defined in the YAML config comes back from there."""
        pool = await st(req).db()
        r = await pool.execute("DELETE FROM channels WHERE id = $1", cid)
        if r.endswith(" 0"):
            raise HTTPException(404, f"channel {cid!r} not found")
        return Response(status_code=204)

    # ------------------------------------------------------------- transcripts
    @app.get("/channels/{cid}/transcripts")
    async def transcripts(req: Request, cid: str, from_: str | None = Query(None, alias="from"),
                          to: str | None = None, pass_: str = Query("any", alias="pass"),
                          limit: int = Query(200, ge=1, le=MAX_ROWS),
                          min_conf: float = Query(0.0, ge=0.0, le=1.0)):
        """Transcript segments (newest ``limit`` in the range) with words and sentence times."""
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
    def chan_dir(cid: str) -> Path:
        if not SAFE_ID.match(cid) or cid in (".", ".."):
            raise HTTPException(404, "not found")
        return resolve_path(s.relay.dir) / cid

    def time_range(from_, to, default_s: float = 3600.0) -> tuple[float, float]:
        t1 = parse_time(to)
        t0 = parse_time(from_)
        b = t1.timestamp() if t1 else time.time()
        a = t0.timestamp() if t0 else b - default_s
        if b <= a:
            raise HTTPException(400, "'to' must be after 'from'")
        if b - a > s.relay.max_clip_s:
            raise HTTPException(400, f"range longer than relay.max_clip_s = {s.relay.max_clip_s:g} s")
        return a, b

    @app.get("/hls/{cid}/{name:path}")
    async def hls(cid: str, name: str, from_: str | None = Query(None, alias="from"),
                  to: str | None = None):
        """Live playlist (``index.m3u8``), archive playlist for a time range
        (``archive.m3u8?from&to``, VOD with PROGRAM-DATE-TIME) and the segments."""
        d = chan_dir(cid)
        hdr = {"Cache-Control": "no-cache"}
        if name == "archive.m3u8":
            a, b = time_range(from_, to)
            entries = await asyncio.to_thread(A.ArchiveIndex(d).read, a, b)
            if not entries:
                raise HTTPException(404, "no archived video in this range (archive disabled, "
                                         "stream down, or older than relay.archive_hours)")
            return Response(A.vod_playlist(entries), media_type="application/vnd.apple.mpegurl",
                            headers=hdr)
        if not (HLS_FILE.match(name) or A.SEG_REL.match(name)):
            raise HTTPException(404, "not found")
        path = d / name
        if not path.is_file():
            raise HTTPException(404, "no video for this channel (relay disabled or not started)")
        if name.endswith(".m3u8"):
            return Response(A.rewrite_live_playlist(path.read_text(), d),
                            media_type="application/vnd.apple.mpegurl", headers=hdr)
        return FileResponse(path, media_type="video/mp2t")

    @app.get("/channels/{cid}/video/coverage")
    async def video_coverage(cid: str, from_: str | None = Query(None, alias="from"),
                             to: str | None = None):
        """Time spans for which archived video exists (default: the last relay.archive_hours)."""
        d = chan_dir(cid)
        t1 = parse_time(to)
        b = t1.timestamp() if t1 else time.time()
        t0 = parse_time(from_)
        a = t0.timestamp() if t0 else b - max(s.relay.archive_hours, s.relay.window_s / 3600) * 3600
        entries = await asyncio.to_thread(A.ArchiveIndex(d).read, a, b)
        spans = A.coverage(entries)
        return {"channel_id": cid, "archive_hours": s.relay.archive_hours,
                "segments": len(entries), "seconds": round(sum(e.d for e in entries), 1),
                "spans": [{"from_utc": x, "to_utc": y} for x, y in spans]}

    @app.get("/channels/{cid}/clip")
    async def clip(cid: str, from_: str = Query(..., alias="from"), to: str = Query(...),
                   format: str = "mp4"):
        """Download the archived video of a time range: mp4 (stream copy, falls back to ts when
        the codecs do not fit MP4) or ts. No re-encoding."""
        if format not in ("mp4", "ts"):
            raise HTTPException(400, "format must be mp4|ts")
        d = chan_dir(cid)
        a, b = time_range(from_, to)
        entries = await asyncio.to_thread(A.ArchiveIndex(d).read, a, b)
        if not entries:
            raise HTTPException(404, "no archived video in this range")
        files = [d / e.uri for e in entries]
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(entries[0].t))
        name = f"{cid}_{stamp}_{round(entries[-1].end - entries[0].t)}s"
        if format == "mp4":
            import tempfile

            tmpdir = resolve_path(s.relay.dir)  # same disk as the archive (clips can be GBs)
            tmpdir.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix=".clip_", dir=tmpdir)
            os.close(fd)
            try:
                await asyncio.to_thread(A.remux_mp4, files, tmp)
                return FileResponse(tmp, media_type="video/mp4", filename=name + ".mp4",
                                    background=BackgroundTask(os.unlink, tmp))
            except Exception as e:  # noqa: BLE001 - e.g. MP2 audio does not fit MP4
                os.unlink(tmp)
                log.info("clip_mp4_fallback_ts", error=str(e))
        return StreamingResponse(A.iter_ts_bytes(files), media_type="video/mp2t",
                                 headers={"Content-Disposition": f'attachment; filename="{name}.ts"'})

    @app.get("/topics")
    async def topics():
        """Named keyword groups (config/topics/topics.yaml) for topic search and reports."""
        return await asyncio.to_thread(R.load_topics, resolve_path(s.api.topics_path))

    @app.get("/search")
    async def search(req: Request, q_: str | None = Query(None, alias="q", max_length=500),
                     topic: str | None = None, channel: str | None = None,
                     from_: str | None = Query(None, alias="from"), to: str | None = None,
                     limit: int = Query(100, ge=1, le=MAX_ROWS),
                     gap_s: float = Query(180.0, ge=10, le=3600)):
        """Full-text search. ``q``: one term or several separated by ``|`` or ``,`` (any matches);
        ``topic``: a named group from /topics (its terms are added). ``clusters`` groups hits of a
        channel less than ``gap_s`` apart (≈ one news item / speech)."""
        terms = R.split_terms(q_)
        if topic:
            t = next((x for x in R.load_topics(resolve_path(s.api.topics_path))
                      if x["name"].lower() == topic.lower()), None)
            if t is None:
                raise HTTPException(404, f"unknown topic {topic!r}")
            terms += t["terms"]
        pats = R.like_patterns(terms)
        if not pats:
            raise HTTPException(400, "give q (search terms) or topic")
        rows = await q(
            req,
            """SELECT channel_id, segment_id, start_utc, end_utc, pass, text_raw, flags, confidence
               FROM asr_segments
               WHERE text_norm ILIKE ANY($1::text[])
                 AND ($2::text IS NULL OR channel_id = $2)
                 AND ($3::timestamptz IS NULL OR start_utc >= $3)
                 AND ($4::timestamptz IS NULL OR start_utc < $4)
               ORDER BY start_utc DESC LIMIT $5""",
            pats, channel, parse_time(from_), parse_time(to), limit)
        res = [_row(r) for r in rows]
        return {"query": q_, "topic": topic, "terms": terms,
                "normalized": normalize(terms[0]) if len(terms) == 1 else [normalize(t) for t in terms],
                "results": res, "clusters": R.cluster_hits(res, gap_s),
                "truncated": len(res) >= limit}

    # ---------------------------------------------------------------- reports
    @app.get("/reports")
    async def report(req: Request, channel: str | None = None,
                     from_: str | None = Query(None, alias="from"), to: str | None = None,
                     format: str = "html", tz: str = "UTC", topics: str | None = None,
                     transcript: bool = False):
        """Transcription report. ``channel``: one id, comma-separated ids, or empty = all live
        channels; an upload id reports on that file. No from/to = overall (all stored data).
        ``topics``: extra terms (``a|b``) reported as a custom topic beside config/topics/topics.yaml."""
        if format not in R.FORMATS:
            raise HTTPException(400, f"format must be one of {sorted(R.FORMATS)}")
        try:
            zone = get_tz(tz)
        except ValueError as e:
            raise HTTPException(400, str(e))
        pool = await st(req).db()
        names = {r["id"]: r["name"] for r in await q(req, "SELECT id, name FROM channels")}
        for c in s.channels:
            names.setdefault(c.id, c.name)
        if channel:
            ids = [c for c in channel.split(",") if c]
            ups = {r["id"]: r["filename"] for r in await q(
                req, "SELECT id, filename FROM uploads WHERE id = ANY($1::text[])", ids)} \
                if any(c.startswith("up-") for c in ids) else {}
            chans = {c: names.get(c) or ups.get(c, "") for c in ids}
        else:  # every live channel: configured ones + any that has transcripts in the period
            seen = await q(req, """SELECT DISTINCT channel_id FROM asr_segments
                                   WHERE channel_id NOT LIKE 'up-%'
                                     AND ($1::timestamptz IS NULL OR start_utc >= $1)
                                     AND ($2::timestamptz IS NULL OR start_utc < $2)""",
                           parse_time(from_), parse_time(to))
            for r in seen:
                names.setdefault(r["channel_id"], "")
            chans = dict(sorted(names.items()))
        tps = await asyncio.to_thread(R.load_topics, resolve_path(s.api.topics_path))
        if topics:
            tps = [{"name": "Custom: " + topics[:60], "terms": R.split_terms(topics)}] + tps
        arch = {}
        if s.relay.archive_hours > 0:
            t1 = parse_time(to)
            b = t1.timestamp() if t1 else time.time()
            t0 = parse_time(from_)
            a = t0.timestamp() if t0 else b - s.relay.archive_hours * 3600
            for cid in chans:
                if SAFE_ID.match(cid):
                    es = await asyncio.to_thread(A.ArchiveIndex(chan_dir(cid)).read, a, b)
                    if es:
                        arch[cid] = {"hours": sum(e.d for e in es) / 3600,
                                     "spans": len(A.coverage(es))}
        acc = [r for r in await accuracy(req) if r["kind"] == "bench" and r["metric"] in ("wer", "cer")]
        async with pool.acquire() as conn:
            data = await R.build_report(conn, chans, parse_time(from_), parse_time(to),
                                        tz if tz.upper() != "UTC" else "UTC", tps,
                                        include_transcript=transcript,
                                        max_rows=MAX_EXPORT_ROWS, archive=arch, accuracy=acc)
        body = R.render_report(data, format, zone)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        hdr = {}
        if req.query_params.get("download") == "1":
            hdr["Content-Disposition"] = f'attachment; filename="report_{stamp}.{format}"'
        return Response(body.encode("utf-8"), media_type=R.FORMATS[format] + "; charset=utf-8",
                        headers=hdr)

    # ---------------------------------------------------------------- uploads
    def up_row(r) -> dict:
        d = _row(r)
        for k in ("base_utc", "started_at", "finished_at"):
            if isinstance(d.get(k), datetime):
                d[k] = d[k].timestamp()
        d["media_url"] = f"/uploads/{d['id']}/media" if d.get("media_file") else None
        dur = d.get("duration_s") or 0
        d["progress"] = (1.0 if d["status"] == "done" else
                         round(min(1.0, 0.5 * (d.get("decoded_s") or 0) / dur
                                   + 0.5 * (d.get("segments_done") or 0)
                                   / max(1, d.get("segments_sent") or 0)), 3) if dur else 0.0)
        return d

    @app.put("/uploads", status_code=201, dependencies=[Depends(require_token)])
    async def upload(req: Request, filename: str = Query(..., min_length=1, max_length=255),
                     language: str = Query("ur", pattern=r"^[a-z]{2,3}$")):
        """Upload a video/audio file as the raw request body (``curl -T file.mp4``); it is
        transcribed in the background at low priority. Poll ``GET /uploads/{id}``."""
        if not s.uploads.enabled:
            raise HTTPException(503, "uploads are disabled (uploads.enabled)")
        ext = U.safe_ext(filename)
        if ext is None:
            raise HTTPException(400, f"unsupported file type; allowed: {' '.join(sorted(U.ALLOWED_EXT))}")
        limit = s.uploads.max_mb * 1024 * 1024
        cl = req.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > limit:
            raise HTTPException(413, f"file larger than uploads.max_mb = {s.uploads.max_mb}")
        pool = await st(req).db()
        uid = U.new_upload_id()
        root = U.uploads_dir(s) / uid
        root.mkdir(parents=True, exist_ok=True)
        dst = root / f"source{ext}"
        size = 0
        try:
            with dst.open("wb") as fh:
                async for chunk in req.stream():
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"file larger than uploads.max_mb = {s.uploads.max_mb}")
                    await asyncio.to_thread(fh.write, chunk)
            if size == 0:
                raise HTTPException(400, "empty body (send the file as the request body)")
            try:
                await asyncio.to_thread(U.probe, dst)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(400, f"not a readable media file with audio: {e}")
            r = await pool.fetchrow(
                """INSERT INTO uploads (id, filename, size_bytes, language, source_file, base_utc)
                   VALUES ($1,$2,$3,$4,$5,$6) RETURNING *""",
                uid, Path(filename).name[:255], size, language, dst.name, U.base_utc_now())
        except BaseException:
            A.remove_tree(root)
            raise
        return up_row(r)

    @app.get("/uploads")
    async def list_uploads(req: Request, limit: int = Query(100, ge=1, le=1000)):
        """Uploaded files with status and progress (newest first)."""
        return [up_row(r) for r in await q(req, "SELECT * FROM uploads ORDER BY created_at DESC "
                                               "LIMIT $1", limit)]

    @app.get("/uploads/{uid}")
    async def get_upload(req: Request, uid: str):
        """One upload: status, progress, duration, base_utc (= media time 0), media_url."""
        rows = await q(req, "SELECT * FROM uploads WHERE id = $1", uid)
        if not rows:
            raise HTTPException(404, "upload not found")
        return up_row(rows[0])

    @app.get("/uploads/{uid}/media")
    async def upload_media(req: Request, uid: str):
        """The uploaded file (or its browser-playable copy) for the viewer; supports Range."""
        if not UPLOAD_ID.match(uid):
            raise HTTPException(404, "not found")
        rows = await q(req, "SELECT media_file FROM uploads WHERE id = $1", uid)
        if not rows or not rows[0]["media_file"]:
            raise HTTPException(404, "no browser-playable media for this upload")
        path = U.uploads_dir(s) / uid / Path(rows[0]["media_file"]).name
        if not path.is_file():
            raise HTTPException(404, "media file missing")
        return FileResponse(path)

    @app.delete("/uploads/{uid}", status_code=204, dependencies=[Depends(require_token)])
    async def delete_upload(req: Request, uid: str):
        """Delete the file, its transcripts and its record. A file being processed is stopped
        when the runner next checks (its record disappears)."""
        if not UPLOAD_ID.match(uid):
            raise HTTPException(404, "not found")
        pool = await st(req).db()
        r = await pool.execute("DELETE FROM uploads WHERE id = $1", uid)
        if r.endswith(" 0"):
            raise HTTPException(404, "upload not found")
        await pool.execute("DELETE FROM asr_segments WHERE channel_id = $1", uid)
        await pool.execute("DELETE FROM stream_events WHERE channel_id = $1", uid)
        A.remove_tree(U.uploads_dir(s) / uid)
        return Response(status_code=204)

    @app.get("/channels/{cid}/events")
    async def stream_events(req: Request, cid: str, limit: int = Query(100, ge=1, le=MAX_ROWS)):
        """Stream up/down, discontinuity and clock events of a channel."""
        rows = await q(req, "SELECT * FROM stream_events WHERE channel_id=$1 "
                            "ORDER BY at_utc DESC LIMIT $2", cid, limit)
        return [_row(r) for r in rows]

    # --------------------------------------------------------------- websocket
    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket, channels: str | None = None):
        """Live transcript events (WebSocket)."""
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
        """Live captions page."""
        return FileResponse(resolve_path("ui/index.html"), headers=NO_CACHE)

    @app.get("/ui/config")
    async def ui_config():
        """What the UI needs to know about this deployment (links, limits, features)."""
        return {"links": s.api.links.model_dump(), "token_required": bool(s.api.token),
                "environment": s.environment,
                "relay": {"enabled": s.relay.enabled, "window_s": s.relay.window_s,
                          "archive_hours": s.relay.archive_hours, "max_clip_s": s.relay.max_clip_s},
                "uploads": {"enabled": s.uploads.enabled, "max_mb": s.uploads.max_mb,
                            "pass": s.uploads.pass_,
                            "extensions": sorted(U.ALLOWED_EXT)},
                "asr": {"device": s.asr.device, "fast": s.asr.fast.model,
                        "refine": s.asr.refine.model if s.asr.refine_enabled else None,
                        "bench_label": bench_label(s)},
                "url_schemes": list(URL_SCHEMES)}

    @app.get("/ui/{name}")
    async def ui_page(name: str):
        """UI pages: archive, reports, uploads, channels, system."""
        if name in UI_ASSETS:
            return FileResponse(resolve_path("ui") / name, media_type=UI_ASSETS[name],
                                headers=NO_CACHE)
        page = UI_PAGES.get(name.removesuffix(".html"))
        if page is None:
            raise HTTPException(404, "not found")
        return FileResponse(resolve_path("ui") / page, headers=NO_CACHE)

    @app.get("/ui/vendor/{name}")
    async def vendor(name: str):
        """Vendored browser libraries (hls.js)."""
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
