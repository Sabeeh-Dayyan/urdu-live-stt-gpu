"""Uploaded file transcription (supervisor child ``uploads``).

An upload is processed like a channel whose source is a file: PyAV decode → the same VAD and
segment policy as live channels → ASR requests on the shared bus → transcripts persisted under
``channel_id = upload id`` with ``utc = base_utc + media time``.

Live channels keep priority:
* requests go to the REFINE stream by default (``uploads.pass: refine`` — large-v3 / beam 5, the
  most accurate decoder), which the ASR service serves only when no FAST work is queued;
* at most ``uploads.max_inflight`` segments of an upload are queued at any time (backpressure on
  the decoder, tracked through the live event fan-out);
* upload requests carry ``no_expire`` so the scheduler never drops them for being old.

Crash safety: a ``processing`` upload found at start-up is re-queued. Re-processing reuses
``base_utc``, so segment ids are identical and the persister's upsert makes it idempotent.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from live_stt.logging import get_logger
from live_stt.settings import ChannelConfig, Settings, resolve_path

log = get_logger(__name__)
TERMINAL = {"final", "refined", "refine_skipped", "error"}
PLAYABLE_EXT = {".mp4", ".m4v", ".mov", ".webm", ".mp3", ".m4a", ".aac", ".wav", ".ogg", ".oga",
                ".opus", ".flac"}
ALLOWED_EXT = PLAYABLE_EXT | {".ts", ".mts", ".m2ts", ".mkv", ".avi", ".flv", ".wmv", ".mpg",
                              ".mpeg", ".3gp", ".wma", ".amr"}


class Cancelled(Exception):
    """The upload's record was deleted while it was being processed."""


def new_upload_id() -> str:
    return "up-" + secrets.token_hex(5)


def safe_ext(filename: str) -> str | None:
    ext = Path(filename or "").suffix.lower()
    return ext if ext in ALLOWED_EXT else None


def uploads_dir(s: Settings) -> Path:
    return resolve_path(s.uploads.dir)


# --------------------------------------------------------------------------- media helpers
def probe(path: Path) -> dict:
    import av

    with av.open(str(path)) as c:
        audio = [s for s in c.streams if s.type == "audio"]
        video = [s for s in c.streams if s.type == "video"]
        if not audio:
            raise ValueError("no audio stream in file")
        dur = c.duration / 1e6 if c.duration else None
        if dur is None and audio[0].duration and audio[0].time_base:
            dur = float(audio[0].duration * audio[0].time_base)
        return {"duration_s": dur, "has_video": bool(video),
                "video_codec": video[0].codec_context.name if video else None,
                "audio_codec": audio[0].codec_context.name}


def make_playable(src: Path, out: Path, info: dict) -> Path | None:
    """Browser-playable copy for the viewer. Containers browsers play are served as-is; others
    are stream-copied into MP4 when the codecs fit (H.264/HEVC + AAC/MP3). No re-encoding."""
    if src.suffix.lower() in PLAYABLE_EXT:
        return src
    if info.get("audio_codec") not in ("aac", "mp3") or (
            info.get("has_video") and info.get("video_codec") not in ("h264", "hevc")):
        return None
    import av

    try:
        with av.open(str(src)) as inp, av.open(str(out), "w", format="mp4",
                                                options={"movflags": "+faststart"}) as o:
            streams = ([s for s in inp.streams if s.type == "video"][:1]
                       + [s for s in inp.streams if s.type == "audio"][:1])
            omap = {s.index: o.add_stream_from_template(s) for s in streams}
            first: dict[int, int] = {}
            for p in inp.demux(streams):
                if p.dts is None or p.size == 0:
                    continue
                first.setdefault(p.stream.index, p.dts)
                p.stream = omap[p.stream.index]
                o.mux(p)
        return out
    except Exception as e:  # noqa: BLE001 - viewer then shows "no playable video"
        log.warning("upload_remux_failed", error=str(e), file=src.name)
        out.unlink(missing_ok=True)
        return None


# --------------------------------------------------------------------------- bus adapter
class UploadBus:
    """Wraps the real bus for one upload's ``ChannelPipeline``: final segments are re-routed to the
    configured pass with ``no_expire``; partials/refines of the pipeline itself are not used.
    ``publish`` blocks while ``max_inflight`` segments are waiting (decoder backpressure)."""

    def __init__(self, bus, s: Settings, upload_id: str) -> None:
        self.bus = bus
        self.s = s
        self.uid = upload_id
        self.pass_ = s.uploads.pass_
        self.sent: set[str] = set()
        self.done: set[str] = set()
        self.cv = threading.Condition()
        self.last_progress = time.time()
        self.stopped = False

    # called from the pub/sub listener thread
    def on_event(self, ev: dict) -> None:
        if ev.get("channel_id") != self.uid or ev.get("type") not in TERMINAL:
            return
        if self.pass_ == "refine" and ev.get("type") == "final":
            return
        with self.cv:
            sid = ev.get("segment_id")
            if sid in self.sent and sid not in self.done:
                self.done.add(sid)
                self.last_progress = time.time()
                self.cv.notify_all()

    def inflight(self) -> int:
        with self.cv:
            return len(self.sent) - len(self.done)

    def wait_slot(self, limit: int) -> None:
        with self.cv:
            while not self.stopped and len(self.sent) - len(self.done) >= limit:
                self.cv.wait(1.0)
                if time.time() - self.last_progress > self.s.uploads.stall_timeout_s:
                    raise TimeoutError("no transcription progress (is the ASR service running?)")

    def publish(self, stream: str, msg: dict) -> str:
        if stream == self.s.bus.stream_fast:
            if msg.get("kind") != "final":
                return ""
            self.wait_slot(self.s.uploads.max_inflight)
            msg = dict(msg, no_expire=True, t_enqueued=time.time())
            if self.pass_ == "refine":
                msg.update({"pass": "refine", "ctx_pre_s": 0.0, "ctx_post_s": 0.0})
                stream = self.s.bus.stream_refine
            with self.cv:
                self.sent.add(msg["segment_id"])
                self.last_progress = time.time()
        elif stream == self.s.bus.stream_refine:
            return ""  # pipeline refine disabled; never double-transcribe
        return self.bus.publish(stream, msg)

    def __getattr__(self, name):  # set_value / get_value / ack … pass through
        return getattr(self.bus, name)


# --------------------------------------------------------------------------- runner
class UploadRunner:
    def __init__(self, s: Settings, bus=None, poll_s: float = 2.0) -> None:
        self.s = s
        self.poll_s = poll_s
        self._stop = threading.Event()
        if bus is None:
            from live_stt.supervisor import make_bus

            bus = make_bus(s)
        self.bus = bus
        self.current: UploadBus | None = None

    def stop(self) -> None:
        self._stop.set()
        if self.current:
            with self.current.cv:
                self.current.stopped = True
                self.current.cv.notify_all()

    # ------------------------------------------------------------ db helpers (sync, own loop)
    def _db(self, sql: str, *args, fetch: bool = False):
        import asyncio

        import asyncpg

        async def go():
            conn = await asyncpg.connect(self.s.db.dsn, timeout=5)
            try:
                return await (conn.fetch(sql, *args) if fetch else conn.execute(sql, *args))
            finally:
                await conn.close()

        return asyncio.run(go())

    def _listen(self, ub: UploadBus) -> threading.Thread:
        import redis

        r = redis.Redis.from_url(self.s.bus.redis_url, socket_connect_timeout=3)
        ps = r.pubsub(ignore_subscribe_messages=True)
        ps.subscribe(self.s.bus.pubsub_live)
        ready = threading.Event()

        def run():
            ready.set()
            try:
                while not ub.stopped:
                    m = ps.get_message(timeout=1.0)
                    if m is None:
                        continue
                    try:
                        ub.on_event(json.loads(m["data"]))
                    except (ValueError, TypeError):
                        continue
            finally:
                try:
                    ps.close()
                    r.close()
                except Exception:  # noqa: BLE001
                    pass

        t = threading.Thread(target=run, name="upload-events", daemon=True)
        t.start()
        ready.wait(5)
        return t

    def recover(self) -> None:
        self._db("UPDATE uploads SET status='queued' WHERE status='processing'")

    def run(self) -> None:
        try:
            self.recover()
        except Exception as e:  # noqa: BLE001
            log.warning("uploads_recover_failed", error=str(e))
        while not self._stop.is_set():
            try:
                rows = self._db("SELECT * FROM uploads WHERE status='queued' "
                                "ORDER BY created_at LIMIT 1", fetch=True)
            except Exception as e:  # noqa: BLE001 - DB down: retry
                log.debug("uploads_poll_failed", error=str(e))
                rows = []
            if not rows:
                self._stop.wait(self.poll_s)
                continue
            self.process(dict(rows[0]))

    def process(self, up: dict) -> None:
        uid = up["id"]
        root = uploads_dir(self.s) / uid
        src = root / up["source_file"]
        log.info("upload_start", upload_id=uid, file=up["filename"])
        self._db("UPDATE uploads SET status='processing', started_at=now(), error=NULL, "
                 "decoded_s=0, segments_sent=0, segments_done=0 WHERE id=$1", uid)
        try:
            info = probe(src)
            media = make_playable(src, root / "media.mp4", info)
            self._db("UPDATE uploads SET duration_s=$2, has_video=$3, media_file=$4 WHERE id=$1",
                     uid, info["duration_s"], info["has_video"],
                     media.name if media else None)
            n_sent, n_done = self._transcribe(up, src)
            self._db("UPDATE uploads SET status='done', finished_at=now(), segments_sent=$2, "
                     "segments_done=$3 WHERE id=$1", uid, n_sent, n_done)
            log.info("upload_done", upload_id=uid, segments=n_sent, transcribed=n_done)
        except Cancelled:
            log.info("upload_cancelled", upload_id=uid)
            time.sleep(5)  # let in-flight segments land, then remove what arrived after DELETE
            try:
                self._db("DELETE FROM asr_segments WHERE channel_id=$1", uid)
                self._db("DELETE FROM stream_events WHERE channel_id=$1", uid)
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            log.error("upload_failed", upload_id=uid, error=f"{type(e).__name__}: {e}")
            try:
                self._db("UPDATE uploads SET status='failed', finished_at=now(), error=$2 "
                         "WHERE id=$1", uid, f"{type(e).__name__}: {e}"[:500])
            except Exception:  # noqa: BLE001
                pass

    def _transcribe(self, up: dict, src: Path) -> tuple[int, int]:
        from live_stt.ingest.source import AudioSource
        from live_stt.vad.silero_onnx import SileroVAD
        from live_stt.worker import ChannelPipeline

        uid = up["id"]
        s = self.s.model_copy(deep=True)
        s.streaming.mode = "segment"
        s.streaming.partials_enabled = False
        s.streaming.max_segment_s = self.s.uploads.max_segment_s
        s.asr.refine_enabled = False
        ch = ChannelConfig(id=uid, name=up["filename"], url=str(src), language=up["language"])
        s.channels = [ch]
        ub = UploadBus(self.bus, s, uid)
        self.current = ub
        listener = self._listen(ub)
        base = up["base_utc"].timestamp() if isinstance(up["base_utc"], datetime) \
            else float(up["base_utc"])
        try:
            vad = SileroVAD(str(resolve_path(s.vad.model_path)))
            pipe = ChannelPipeline(s, ch, ub, vad, wall=time.time)
            src_it = AudioSource(str(src), ch.audio_track, s.ingest, s.clock,
                                 wallclock=lambda: base, max_connects=1)
            last = 0.0
            from live_stt.ingest.source import SourceEvent

            for item in src_it:
                if self._stop.is_set():
                    src_it.stop()
                    break
                if isinstance(item, SourceEvent):
                    pipe.on_source_event(item)
                    continue
                pipe.on_chunk(item, src_it.clock.warm if src_it.clock else True)
                if time.time() - last > 2:
                    last = time.time()
                    self._progress(uid, src_it._media_s, ub)
            pipe._flush()
            self._progress(uid, src_it._media_s, ub)
            # wait for the remaining segments to be transcribed
            while ub.inflight() and not self._stop.is_set():
                ub.wait_slot(ub.inflight())  # returns as soon as one more segment is done
                self._progress(uid, src_it._media_s, ub)
            self._progress(uid, src_it._media_s, ub)
            if self._stop.is_set():
                raise InterruptedError("stopped")
            return len(ub.sent), len(ub.done)
        finally:
            with ub.cv:
                ub.stopped = True
                ub.cv.notify_all()
            listener.join(3)
            self.current = None

    def _progress(self, uid: str, decoded_s: float, ub: UploadBus) -> None:
        try:
            rows = self._db("UPDATE uploads SET decoded_s=$2, segments_sent=$3, segments_done=$4 "
                            "WHERE id=$1 RETURNING id", uid, float(decoded_s), len(ub.sent),
                            len(ub.done), fetch=True)
        except Exception:  # noqa: BLE001 - DB hiccup: keep going
            return
        if not rows:
            raise Cancelled(uid)


def run_uploads(s: Settings) -> None:
    import signal

    r = UploadRunner(s)

    def _term(*_):
        r.stop()

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    uploads_dir(s).mkdir(parents=True, exist_ok=True)
    r.run()


def base_utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)
