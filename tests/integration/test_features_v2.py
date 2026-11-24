"""Video archive, time-range replay/clips, topic search, reports, channel management from the UI,
uploaded-file transcription, UI pages and the system/links page backend."""

import asyncio
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from live_stt.api.app import create_app
from live_stt.ingest import archive as A
from live_stt.ingest.relay import HlsRelay
from live_stt.ingest.source import AudioChunk, AudioSource
from live_stt.settings import PROJECT_ROOT, AudioTrack, ClockConfig, IngestConfig
from live_stt.store.db import migrate
from tests.integration.conftest import requires_pg, requires_redis

T0 = 1_795_000_000.0
REF_WAV = PROJECT_ROOT / "testbed" / "reference" / "fleurs_ur.wav"
VAD = PROJECT_ROOT / "models" / "silero_vad.onnx"


def ff(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True,
                   timeout=180)


@pytest.fixture(scope="module")
def ts_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("arch") / "av.ts"
    ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25", "-f", "lavfi", "-i",
       "sine=f=440:r=48000", "-t", "9", "-ac", "2", "-c:v", "libx264", "-preset", "ultrafast",
       "-g", "50", "-c:a", "aac", "-f", "mpegts", str(p))
    return p


def _run(ts, relay, connects=1):
    src = AudioSource(str(ts), AudioTrack(), IngestConfig(), ClockConfig(), max_connects=connects,
                      relay=relay)
    return [c for c in src if isinstance(c, AudioChunk)]


# --------------------------------------------------------------------------- archive (relay)
def test_relay_archive_keeps_and_indexes_every_segment(ts_file, tmp_path):
    d = tmp_path / "ch"
    r = HlsRelay(d, window_s=120, segment_s=2, archive_hours=1)
    _run(ts_file, r)
    assert r.errors == 0
    entries = A.ArchiveIndex(d).read(time.time() - 3600, time.time() + 3600)
    assert len(entries) >= 4 and all((d / e.uri).is_file() for e in entries)
    assert sum(e.d for e in entries) == pytest.approx(9.0, abs=2.1)
    assert all(A.SEG_REL.match(e.uri) for e in entries)
    assert [e.t for e in entries] == sorted(e.t for e in entries)
    # the live playlist (absolute strftime paths from ffmpeg) is served with relative URIs
    live = A.rewrite_live_playlist((d / "index.m3u8").read_text(), d)
    assert str(d) not in live and "\na/" in live
    # a reconnect keeps the old segments (no delete_segments) and never overwrites them
    n_files = len(list((d / "a").rglob("seg_*.ts")))
    _run(ts_file, HlsRelay(d, window_s=120, segment_s=2, archive_hours=1))
    assert len(list((d / "a").rglob("seg_*.ts"))) >= n_files + 4


def test_retention_sweep_deletes_only_old_segments(tmp_path):
    d = tmp_path / "ch"
    day = d / "a" / "20200101"
    day.mkdir(parents=True)
    old, new = day / "seg_20200101T000000_1_0000.ts", day / "seg_20200101T000002_1_0001.ts"
    old.write_bytes(b"x"), new.write_bytes(b"y")
    t = time.time() - 7200
    os.utime(old, (t, t))
    (d / "archive").mkdir()
    (d / "archive" / "20000101.idx").write_text("1\t2\ta/x\t0\n")
    (d / "archive" / f"{A._day(time.time())}.idx").write_text("")
    assert A.sweep_retention(d, hours=1) == 1
    assert not old.exists() and new.exists()
    assert not (d / "archive" / "20000101.idx").exists()
    assert (d / "archive" / f"{A._day(time.time())}.idx").exists()
    assert A.sweep_retention(d, hours=0) == 0


def test_index_read_vod_and_coverage(tmp_path):
    d = tmp_path / "ch"
    idx = A.ArchiveIndex(d)
    es = []
    for i in range(5):  # 5 × 2 s, then a 30 s gap, then 2 more (new session)
        es.append(A.Entry(T0 + 2 * i, 2.0, f"a/20261124/seg_20261124T000000_1_{i:04d}.ts"))
    for i in range(2):
        es.append(A.Entry(T0 + 40 + 2 * i, 2.0, f"a/20261124/seg_20261124T000040_2_{i:04d}.ts",
                          disc=i == 0))
    es.append(A.Entry(T0 + 50, 2.0, "../../etc/passwd"))  # never served
    idx.append(es)
    for e in es[:-1]:
        (d / e.uri).parent.mkdir(parents=True, exist_ok=True)
        (d / e.uri).write_bytes(b"\x47" * 188)
    got = idx.read(T0 + 3, T0 + 41)
    assert [e.t for e in got] == [T0 + 2, T0 + 4, T0 + 6, T0 + 8, T0 + 40]
    assert A.coverage(idx.read(T0, T0 + 100)) == [(T0, T0 + 10), (T0 + 40, T0 + 44)]
    pl = A.vod_playlist(got)
    assert pl.count("#EXT-X-DISCONTINUITY") == 1 and pl.rstrip().endswith("#EXT-X-ENDLIST")
    assert "#EXT-X-PROGRAM-DATE-TIME:" in pl and "PLAYLIST-TYPE:VOD" in pl
    assert A.parse_pdt("2026-11-24T09:29:36.724+0500") == pytest.approx(
        datetime(2026, 11, 24, 4, 29, 36, 724000, tzinfo=timezone.utc).timestamp())
    assert idx.last_t() == T0 + 50


# --------------------------------------------------------------------------- API
@pytest.fixture
def client(settings, fresh_db, tmp_path):
    asyncio.run(migrate(fresh_db))
    settings.relay.dir = str(tmp_path / "hls")
    settings.relay.archive_hours = 24
    settings.uploads.dir = str(tmp_path / "uploads")
    settings.api.token = "s3cret"
    with TestClient(create_app(settings)) as c:
        c.hdr = {"Authorization": "Bearer s3cret"}
        yield c


def _insert(dsn, rows):
    import asyncpg

    async def go():
        c = await asyncpg.connect(dsn)
        for ch, t, p, text, conf in rows:
            words = [{"w": w, "s": t + 0.5 * i, "e": t + 0.5 * i + 0.4, "p": conf}
                     for i, w in enumerate(text.split())]
            await c.execute(
                "INSERT INTO asr_segments(channel_id, segment_id, start_utc, end_utc, pass, "
                "text_raw, text_norm, words, confidence, latency_s) "
                "VALUES ($1,$2,$3,$4,$5,$6,$6,$7,$8,$9)",
                ch, f"{ch}:{int(t * 1000)}", datetime.fromtimestamp(t, timezone.utc),
                datetime.fromtimestamp(t + 0.5 * len(words), timezone.utc), p, text,
                json.dumps(words), conf, 3.0)
        await c.close()

    asyncio.run(go())


ROWS = [
    ("c1", T0, "fast", "وزیراعظم نے آج خطاب کیا", 0.95),
    ("c1", T0 + 30, "refined", "وزیر اعظم نے کہا کہ معیشت بہتر ہے", 0.9),
    ("c1", T0 + 3600, "fast", "موسم آج گرم رہے گا", 0.7),
    ("c2", T0 + 10, "fast", "کرکٹ میچ کل ہوگا", 0.85),
]


@requires_pg
@requires_redis
def test_archive_endpoints_playlist_coverage_clip(client, settings, ts_file):
    d = Path(settings.relay.dir) / "c1"
    _run(ts_file, HlsRelay(d, window_s=120, segment_s=2, archive_hours=24))
    es = A.ArchiveIndex(d).read(time.time() - 3600, time.time() + 3600)
    a, b = es[0].t, es[-1].end
    r = client.get("/hls/c1/archive.m3u8", params={"from": a, "to": b})
    assert r.status_code == 200 and "#EXT-X-ENDLIST" in r.text
    uris = [ln for ln in r.text.splitlines() if ln and not ln.startswith("#")]
    assert len(uris) == len(es)
    seg = client.get(f"/hls/c1/{uris[0]}")
    assert seg.status_code == 200 and seg.headers["content-type"] == "video/mp2t"
    live = client.get("/hls/c1/index.m3u8").text
    assert str(d) not in live
    cov = client.get("/channels/c1/video/coverage", params={"from": a, "to": b + 1}).json()
    assert len(cov["spans"]) == 1 and cov["seconds"] == pytest.approx(sum(e.d for e in es), 0.01)
    mp4 = client.get("/channels/c1/clip", params={"from": a, "to": b, "format": "mp4"})
    assert mp4.status_code == 200 and mp4.headers["content-type"] == "video/mp4"
    assert mp4.content[4:8] == b"ftyp" and "attachment" in mp4.headers["content-disposition"]
    ts = client.get("/channels/c1/clip", params={"from": a, "to": b, "format": "ts"})
    assert ts.status_code == 200 and ts.content[0] == 0x47 and len(ts.content) % 188 == 0
    # validation / safety
    assert client.get("/hls/c1/archive.m3u8", params={"from": a - 7200, "to": a - 3600}).status_code == 404
    assert client.get("/hls/c1/archive.m3u8", params={"from": a, "to": a + 5 * 3600}).status_code == 400
    assert client.get("/hls/c1/archive.m3u8", params={"from": b, "to": a}).status_code == 400
    assert client.get("/channels/c1/clip", params={"from": a, "to": b, "format": "avi"}).status_code == 400
    for bad in ("/hls/c1/a/../../../etc/passwd", "/hls/c1/a%2F..%2F..%2Fsecret",
                "/hls/c1/a/20261124/x.ts", "/hls/..%2Fc1/index.m3u8"):
        assert client.get(bad).status_code == 404, bad


@requires_pg
@requires_redis
def test_search_terms_topic_and_clusters(client, fresh_db):
    _insert(fresh_db, ROWS)
    j = client.get("/search", params={"q": "وزیراعظم | وزیر اعظم"}).json()
    assert len(j["results"]) == 2 and len(j["clusters"]) == 1
    c = j["clusters"][0]
    assert c["hits"] == 2 and c["start_utc"] == pytest.approx(T0) and c["channel_id"] == "c1"
    j = client.get("/search", params={"topic": "prime minister"}).json()
    assert len(j["results"]) == 2 and "وزیراعظم" in j["terms"]
    assert [t["name"] for t in client.get("/topics").json()][0] == "Prime Minister"
    j = client.get("/search", params={"q": "وزیراعظم", "gap_s": 10}).json()
    assert len(j["clusters"]) == 1  # only the first row has the one-word spelling
    j = client.get("/search", params={"q": "وزیراعظم|وزیر اعظم", "gap_s": 10}).json()
    assert len(j["clusters"]) == 2  # 30 s apart > 10 s
    j = client.get("/search", params={"q": "موسم", "from": T0 + 100, "to": T0 + 7200}).json()
    assert [r["channel_id"] for r in j["results"]] == ["c1"]
    assert client.get("/search", params={"topic": "nope"}).status_code == 404
    assert client.get("/search").status_code == 400
    r = client.get("/search", params={"q": "%"})  # punctuation only → empty after normalising
    assert r.status_code == 400
    j = client.get("/search", params={"q": "ab%_cd"}).json()  # LIKE wildcards are literal
    assert j["results"] == []


@requires_pg
@requires_redis
def test_reports_formats_range_and_overall(client, fresh_db):
    _insert(fresh_db, ROWS)
    j = client.get("/reports", params={"format": "json", "channel": "c1", "from": T0 - 1,
                                       "to": T0 + 100, "tz": "Asia/Karachi"}).json()
    assert j["summary"]["segments"] == 2 and j["summary"]["refined"] == 1
    assert j["summary"]["words"] == 13
    assert j["channels"][0]["confidence"] == pytest.approx(0.925, abs=1e-3)
    assert j["channels"][0]["latency_fast_p50_s"] == pytest.approx(3.0)
    pm = next(t for t in j["topics"] if t["name"] == "Prime Minister")
    assert pm["mentions"] == 2
    assert j["hourly"][0]["hour_local"].endswith(":00") and not j["period"]["overall"]
    assert all(w["word"] not in ("نے", "کہ") for w in j["top_words"])  # stopwords out
    overall = client.get("/reports", params={"format": "json"}).json()
    assert overall["period"]["overall"] and overall["summary"]["segments"] == 4
    assert {c["channel_id"] for c in overall["channels"]} >= {"c1", "c2"}
    h = client.get("/reports", params={"channel": "c1", "transcript": "true",
                                       "topics": "معیشت"})
    assert h.status_code == 200 and h.headers["content-type"].startswith("text/html")
    assert "Transcription report" in h.text and "Custom: معیشت" in h.text
    assert "وزیراعظم نے آج خطاب کیا" in h.text and "<script" not in h.text.lower()
    md = client.get("/reports", params={"channel": "c1,c2", "format": "md", "download": "1"})
    assert md.text.startswith("# Transcription report") and "attachment" in md.headers[
        "content-disposition"]
    assert client.get("/reports", params={"format": "pdf"}).status_code == 400
    assert client.get("/reports", params={"tz": "Mars/Base"}).status_code == 400


@requires_pg
@requires_redis
def test_channel_management_from_ui(client):
    ok = {"id": "news1", "name": "News", "url": "udp://@239.1.1.1:5000"}
    assert client.post("/channels", json=ok).status_code == 401
    assert client.post("/channels", json=ok, headers=client.hdr).status_code == 201
    for bad_url in ("file:///etc/passwd", "/etc/passwd", "ftp://x/y", "javascript:alert(1)"):
        r = client.post("/channels", json=dict(ok, id="x2", url=bad_url), headers=client.hdr)
        assert r.status_code == 400, bad_url
    for bad_id in ("up-0123456789", "a/b", "..", "x" * 65):
        r = client.post("/channels", json=dict(ok, id=bad_id), headers=client.hdr)
        assert r.status_code == 400, bad_id
    r = client.patch("/channels/news1", json={"url": "srt://10.0.0.1:9000?mode=caller"},
                     headers=client.hdr)
    assert r.status_code == 200 and r.json()["url"].startswith("srt://")
    assert client.patch("/channels/news1", json={"url": "file:///x"},
                        headers=client.hdr).status_code == 400
    assert client.patch("/channels/news1", json={"url": None},
                        headers=client.hdr).status_code == 400
    assert client.delete("/channels/news1").status_code == 401
    assert client.delete("/channels/news1", headers=client.hdr).status_code == 204
    assert client.delete("/channels/news1", headers=client.hdr).status_code == 404


@requires_pg
@requires_redis
def test_ui_pages_config_and_api_listing(client):
    for p in ("/", "/ui/archive", "/ui/reports", "/ui/uploads", "/ui/channels", "/ui/system",
              "/ui/archive.html"):
        r = client.get(p)
        assert r.status_code == 200 and "/ui/common.js" in r.text, p
    assert client.get("/ui/common.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/ui/common.css").headers["content-type"].startswith("text/css")
    for bad in ("/ui/../src/live_stt/settings.py", "/ui/secret", "/ui/index"):
        assert client.get(bad).status_code == 404, bad
    cfg = client.get("/ui/config").json()
    assert cfg["token_required"] is True and "{host}" in cfg["links"]["grafana"]
    assert cfg["relay"]["archive_hours"] == 24 and ".mp4" in cfg["uploads"]["extensions"]
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/reports", "/search", "/topics", "/uploads", "/channels/{cid}/clip",
              "/channels/{cid}/video/coverage", "/hls/{cid}/{name}"):
        assert p in paths, p
    undocumented = [f"{m} {p}" for p, ops in paths.items() for m, op in ops.items()
                    if not (op.get("description") or "").strip()]
    assert undocumented == []  # the System page shows a description for every endpoint


# --------------------------------------------------------------------------- uploads
@pytest.fixture(scope="module")
def speech_mp4(tmp_path_factory):
    if not REF_WAV.exists():
        pytest.skip("reference audio missing")
    p = tmp_path_factory.mktemp("up") / "clip.mp4"
    ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25", "-ss", "0", "-t", "40",
       "-i", str(REF_WAV), "-t", "40", "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
       "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(p))
    return p


@requires_pg
@requires_redis
def test_upload_api_validation(client, speech_mp4, tmp_path):
    data = speech_mp4.read_bytes()
    assert client.put("/uploads", params={"filename": "a.mp4"}, content=data).status_code == 401
    h = client.hdr
    assert client.put("/uploads", params={"filename": "a.exe"}, content=data,
                      headers=h).status_code == 400
    assert client.put("/uploads", params={"filename": "a.mp4"}, content=b"",
                      headers=h).status_code == 400
    assert client.put("/uploads", params={"filename": "a.mp4"}, content=b"not media" * 100,
                      headers=h).status_code == 400
    assert list((tmp_path / "uploads").glob("up-*")) == []  # rejected uploads leave nothing
    r = client.put("/uploads", params={"filename": "../../news clip.mp4", "language": "ur"},
                   content=data, headers=h)
    assert r.status_code == 201
    up = r.json()
    assert up["filename"] == "news clip.mp4" and up["status"] == "queued"
    assert up["size_bytes"] == len(data) and up["id"].startswith("up-")
    assert client.get("/uploads").json()[0]["id"] == up["id"]
    assert client.get(f"/uploads/{up['id']}/media").status_code == 404  # not processed yet
    assert client.get("/uploads/up-0000000000").status_code == 404
    assert client.delete(f"/uploads/{up['id']}").status_code == 401
    assert client.delete(f"/uploads/{up['id']}", headers=h).status_code == 204
    assert not (tmp_path / "uploads" / up["id"]).exists()


@requires_pg
@requires_redis
@pytest.mark.skipif(not VAD.exists(), reason="Silero VAD model missing (make models)")
def test_upload_transcribed_end_to_end_with_live_priority(client, settings, fresh_db,
                                                          speech_mp4):
    """Upload → runner (real decode + VAD + segmenting) → REFINE stream with no_expire →
    ASR service (fake backend) → persister → transcripts at base_utc + media time; viewer media."""
    from live_stt.asr.service import AsrService
    from live_stt.store.persister import Persister
    from live_stt.supervisor import make_bus
    from live_stt.uploads import UploadRunner
    from tests.fakes import FakeASR

    settings.asr.refine_max_lag_s = 0.001  # live refines would expire instantly; uploads must not
    settings.uploads.max_inflight = 2
    r = client.put("/uploads", params={"filename": "clip.mp4"}, content=speech_mp4.read_bytes(),
                   headers=client.hdr)
    up = r.json()
    svc = AsrService(settings, make_bus(settings), FakeASR())
    pers = Persister(settings)
    t_svc = threading.Thread(target=svc.run, daemon=True)
    t_pers = threading.Thread(target=lambda: asyncio.run(pers.run()), daemon=True)
    t_svc.start(), t_pers.start()
    try:
        runner = UploadRunner(settings)
        row = runner._db("SELECT * FROM uploads WHERE id=$1", up["id"], fetch=True)[0]
        t = time.time()
        runner.process(dict(row))
        took = time.time() - t
        got = client.get(f"/uploads/{up['id']}").json()
        assert got["status"] == "done", got
        assert got["segments_sent"] >= 3 and got["segments_done"] == got["segments_sent"]
        assert got["duration_s"] == pytest.approx(40, abs=1) and got["has_video"]
        assert got["decoded_s"] == pytest.approx(40, abs=1.5) and got["progress"] == 1.0
        assert took < 120
        deadline = time.time() + 20
        rows = []
        while time.time() < deadline:
            rows = client.get(f"/channels/{up['id']}/transcripts", params={"limit": 1000}).json()
            if len(rows) >= 1:
                break
            time.sleep(0.5)
        assert rows and all(x["pass"] == "refined" for x in rows)
        base = got["base_utc"]
        assert all(base - 0.5 <= x["start_utc"] <= base + 41 for x in rows)
        m = client.get(got["media_url"])
        assert m.status_code == 200 and m.headers["content-type"] == "video/mp4"
        rng = client.get(got["media_url"], headers={"Range": "bytes=0-99"})
        assert rng.status_code == 206 and len(rng.content) == 100
        srt = client.get(f"/channels/{up['id']}/export", params={"format": "srt", "from": base})
        assert srt.status_code == 200
        rep = client.get("/reports", params={"channel": up["id"], "format": "json"}).json()
        assert rep["channels"][0]["name"] == "clip.mp4" and rep["summary"]["segments"] == len(rows)
    finally:
        svc.stop()
        pers.stop()
        t_svc.join(5)
        t_pers.join(10)


def test_scheduler_never_expires_or_evicts_upload_jobs():
    from live_stt.asr.scheduler import REFINE, Job, Scheduler

    now = [1000.0]
    s = Scheduler(refine_max=2, refine_max_lag_s=10, wall=lambda: now[0])
    live = Job("l1", REFINE, 5, t_enqueued=1000, payload={})
    up1 = Job("u1", REFINE, 5, t_enqueued=1000, payload={"no_expire": True})
    s.push(up1), s.push(live)
    s.push(Job("u2", REFINE, 5, t_enqueued=1000, payload={"no_expire": True}))  # full: evicts live
    now[0] = 2000.0
    d = s.next()
    assert [j.msg_id for j, _ in d.skipped] == ["l1"]
    assert [j.msg_id for j in d.jobs] == ["u1"]
    assert [j.msg_id for j in s.next().jobs] == ["u2"]


def test_upload_bus_routes_and_backpressure(settings):
    from live_stt.bus.base import InMemoryBus
    from live_stt.uploads import UploadBus

    bus = InMemoryBus()
    ub = UploadBus(bus, settings, "up-0123456789")
    settings.uploads.max_inflight = 1
    settings.uploads.stall_timeout_s = 0.3
    ub.publish(settings.bus.stream_fast, {"kind": "partial", "segment_id": "p"})
    ub.publish(settings.bus.stream_refine, {"kind": "final", "segment_id": "r"})
    ub.publish(settings.bus.stream_fast, {"kind": "final", "segment_id": "s1", "pass": "fast"})
    msgs = bus.messages(settings.bus.stream_refine)
    assert len(msgs) == 1 and msgs[0]["pass"] == "refine" and msgs[0]["no_expire"] is True
    assert bus.messages(settings.bus.stream_fast) == []
    ub.on_event({"channel_id": "other", "type": "refined", "segment_id": "s1"})
    ub.on_event({"channel_id": "up-0123456789", "type": "final", "segment_id": "s1"})
    assert ub.inflight() == 1  # refine mode waits for the refined event
    with pytest.raises(TimeoutError):  # slot never frees → stall timeout, not a hang
        ub.publish(settings.bus.stream_fast, {"kind": "final", "segment_id": "s2"})
    ub.on_event({"channel_id": "up-0123456789", "type": "refine_skipped", "segment_id": "s1"})
    assert ub.inflight() == 0


@requires_pg
@requires_redis
def test_upload_deleted_while_processing_is_cancelled(client, settings):
    from live_stt.bus.base import InMemoryBus
    from live_stt.uploads import Cancelled, UploadBus, UploadRunner

    runner = UploadRunner(settings, bus=InMemoryBus())
    ub = UploadBus(runner.bus, settings, "up-00000000aa")
    with pytest.raises(Cancelled):  # no such record (deleted) → stop processing
        runner._progress("up-00000000aa", 1.0, ub)
