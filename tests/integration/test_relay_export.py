"""HLS video relay on real media + export / accuracy / HLS API endpoints."""

import asyncio
import re
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from live_stt.api.app import create_app
from live_stt.ingest.relay import HlsRelay
from live_stt.ingest.source import AudioChunk, AudioSource
from live_stt.settings import AudioTrack, ClockConfig, IngestConfig
from live_stt.store.accuracy import record
from live_stt.store.db import migrate
from tests.integration.conftest import requires_pg, requires_redis

T0 = 1_795_000_000.0


def ff(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True,
                   timeout=120)


@pytest.fixture(scope="module")
def ts_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("relay") / "av.ts"
    ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25", "-f", "lavfi", "-i",
       "sine=f=440:r=48000", "-t", "7", "-ac", "2", "-c:v", "libx264", "-preset", "ultrafast",
       "-g", "50", "-c:a", "aac", "-f", "mpegts", str(p))
    return p


def _audio(ts, relay=None, connects=1):
    src = AudioSource(str(ts), AudioTrack(), IngestConfig(), ClockConfig(),
                      max_connects=connects, relay=relay)
    return [c.samples for c in src if isinstance(c, AudioChunk) and len(c.samples)]


def test_relay_does_not_change_transcription_audio(ts_file, tmp_path):
    plain = np.concatenate(_audio(ts_file))
    r = HlsRelay(tmp_path / "hls", window_s=60, segment_s=2)
    relayed = np.concatenate(_audio(ts_file, r))
    assert r.errors == 0 and r.packets > 100
    assert np.array_equal(plain, relayed)


def test_relay_rolling_window_discontinuity_and_playable_segments(ts_file, tmp_path):
    d = tmp_path / "hls"
    r = HlsRelay(d, window_s=4, segment_s=2)
    _audio(ts_file, r, connects=3)  # file re-opened 3× ⇒ PTS restarts each time
    assert r.errors == 0
    pl = (d / "index.m3u8").read_text()
    entries = re.findall(r"^seg_\d{8}\.ts$", pl, re.M)
    assert 1 <= len(entries) <= 3  # hls_list_size = max(3, ceil(4/2))
    assert "#EXT-X-PROGRAM-DATE-TIME:" in pl
    assert "#EXT-X-ENDLIST" not in pl  # live playlist
    on_disk = sorted(p.name for p in d.glob("seg_*.ts"))
    assert len(on_disk) <= len(entries) + 1  # trimmed + leftover segments are deleted
    nums = [int(e[4:12]) for e in entries]
    assert nums == sorted(nums) and nums[-1] >= 5  # numbering continued across reconnects
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
                            "-of", "csv=p=0", str(d / entries[-1])],
                           capture_output=True, text=True, check=True, timeout=30).stdout
    assert {"video", "audio"} <= set(probe.split())


def test_relay_marks_reconnect_as_discontinuity(ts_file, tmp_path):
    d = tmp_path / "hls"
    r = HlsRelay(d, window_s=60, segment_s=2)
    _audio(ts_file, r, connects=2)
    pl = (d / "index.m3u8").read_text()
    assert r.errors == 0
    # ffmpeg may also tag the list start; what matters is a tag at the reconnect point
    lines = pl.splitlines()
    first_seg = next(i for i, ln in enumerate(lines) if ln.startswith("seg_"))
    mid = [i for i, ln in enumerate(lines) if ln == "#EXT-X-DISCONTINUITY" and i > first_seg]
    assert len(mid) == 1
    assert sum(ln.startswith("seg_") for ln in lines[:mid[0]]) >= 3
    assert sum(ln.startswith("seg_") for ln in lines[mid[0]:]) >= 3


def test_relay_discards_stale_playlist(ts_file, tmp_path):
    import os
    import time as _t

    d = tmp_path / "hls"
    d.mkdir()
    (d / "index.m3u8").write_text("#EXTM3U\n#EXTINF:2.0,\nseg_00000042.ts\n")
    (d / "seg_00000042.ts").write_bytes(b"x")
    (d / "seg_00099999.ts").write_bytes(b"x")  # unreferenced leftover
    old = _t.time() - 3600
    os.utime(d / "index.m3u8", (old, old))
    r = HlsRelay(d, window_s=60, segment_s=2)
    _audio(ts_file, r)
    pl = (d / "index.m3u8").read_text()
    assert r.errors == 0 and "seg_00000042" not in pl and "DISCONTINUITY" not in pl
    assert not (d / "seg_00000042.ts").exists() and not (d / "seg_00099999.ts").exists()


def test_relay_failures_never_raise(ts_file, tmp_path):
    blocker = tmp_path / "file_not_dir"
    blocker.write_text("x")
    r = HlsRelay(blocker / "hls", window_s=10, segment_s=2, retry_s=0.01)
    audio = _audio(ts_file, r)  # cannot create the directory → relay disabled, audio unaffected
    assert r.errors >= 1 and sum(len(a) for a in audio) > 16000 * 6
    r2 = HlsRelay(tmp_path / "ok", window_s=10, segment_s=2)
    r2.push(object())  # garbage before open: ignored
    r2._templates = [object()]
    r2._retry_at = 1.0
    r2.push(object())  # re-open with bad templates fails → counted, not raised
    assert r2.errors >= 1


# ----------------------------------------------------------------------- API
pytestmark_api = [requires_pg, requires_redis]


@pytest.fixture
def client(settings, fresh_db, tmp_path):
    asyncio.run(migrate(fresh_db))
    settings.relay.dir = str(tmp_path / "hls")
    with TestClient(create_app(settings)) as c:
        yield c


def _insert(dsn, rows):
    import asyncpg
    import json
    from datetime import datetime, timezone

    async def go():
        c = await asyncpg.connect(dsn)
        for ch, t, p, words, conf in rows:
            text = " ".join(w["w"] for w in words)
            await c.execute(
                "INSERT INTO asr_segments(channel_id, segment_id, start_utc, end_utc, pass, "
                "text_raw, text_norm, words, confidence) VALUES ($1,$2,$3,$3,$4,$5,$5,$6,$7)",
                ch, f"{ch}:{int(t * 1000)}", datetime.fromtimestamp(t, timezone.utc), p, text,
                json.dumps(words), conf)
        await c.close()

    asyncio.run(go())


def _w(w, s, p):
    return {"w": w, "s": T0 + s, "e": T0 + s + 0.4, "p": p}


SEG_ROWS = [
    ("c1", T0, "fast", [_w("پہلا", 0, 0.99), _w("جملہ۔", 0.4, 0.98), _w("دوسرا", 1.0, 0.5),
                        _w("جملہ۔", 1.4, 0.6)], 0.77),
    ("c1", T0 + 10, "refined", [_w("تیسرا", 10, 0.97), _w("جملہ", 10.4, 0.96)], 0.965),
    ("c2", T0 + 5, "fast", [_w("دوسرا", 5, 0.9)], 0.9),
]


@requires_pg
@requires_redis
def test_transcripts_have_sentences_and_min_conf(client, fresh_db):
    _insert(fresh_db, SEG_ROWS)
    rows = client.get("/channels/c1/transcripts").json()
    assert [r["confidence"] for r in rows] == pytest.approx([0.77, 0.965])
    assert [s["text"] for s in rows[0]["sentences"]] == ["پہلا جملہ۔", "دوسرا جملہ۔"]
    assert rows[0]["sentences"][1]["start_utc"] == pytest.approx(T0 + 1.0)
    hi = client.get("/channels/c1/transcripts", params={"min_conf": 0.9}).json()
    assert [r["pass"] for r in hi] == ["refined"]
    assert client.get("/channels/c1/transcripts", params={"min_conf": 2}).status_code == 422


@requires_pg
@requires_redis
def test_export_formats_filters_and_validation(client, fresh_db):
    _insert(fresh_db, SEG_ROWS)
    r = client.get("/channels/c1/export", params={"format": "srt"})
    assert r.status_code == 200 and r.headers["x-sentences"] == "3"
    assert "attachment;" in r.headers["content-disposition"] and ".srt" in \
        r.headers["content-disposition"]
    assert r.text.startswith("1\n00:00:00,000 --> 00:00:00,800\nپہلا جملہ۔")
    r = client.get("/channels/c1/export", params={"format": "csv", "min_conf": 0.9,
                                                  "tz": "Asia/Karachi"})
    body = r.content.decode("utf-8")
    assert r.headers["x-sentences"] == "2" and "دوسرا" not in body  # low-confidence sentence out
    assert "+05:00" in body and body.startswith("﻿")
    r = client.get("/channels/c1/export", params={"format": "vtt", "from": str(T0 - 10)})
    assert "00:00:10.000 --> " in r.text  # relative to `from`
    for fmt in ("txt", "json"):
        assert client.get("/channels/c1/export", params={"format": fmt}).status_code == 200
    assert client.get("/channels/c1/export", params={"format": "exe"}).status_code == 400
    assert client.get("/channels/c1/export", params={"tz": "No/Where"}).status_code == 400
    assert client.get("/channels/c1/export", params={"pass": "x"}).status_code == 400
    r = client.get("/channels/none/export", params={"format": "json"})
    assert r.status_code == 200 and r.json() == []
    r = client.get("/channels/a%22b/export")  # unsafe id never reaches the filename header
    assert 'filename="channel_' in r.headers["content-disposition"]


@requires_pg
@requires_redis
def test_hls_endpoint_serves_and_rejects_traversal(client, settings):
    from pathlib import Path

    d = Path(settings.relay.dir) / "c1"
    d.mkdir(parents=True)
    (d / "index.m3u8").write_text("#EXTM3U\n")
    (d / "seg_00000001.ts").write_bytes(b"\x47" * 188)
    (Path(settings.relay.dir) / "secret.txt").write_text("no")
    r = client.get("/hls/c1/index.m3u8")
    assert r.status_code == 200 and r.headers["content-type"].startswith(
        "application/vnd.apple.mpegurl") and r.headers["cache-control"] == "no-cache"
    r = client.get("/hls/c1/seg_00000001.ts")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp2t"
    for bad in ("/hls/c1/..%2Fsecret.txt", "/hls/../secret.txt", "/hls/c1/secret.txt",
                "/hls/..%2F..%2Fetc/passwd", "/hls/c1/seg_1.ts", "/hls/c9/index.m3u8"):
        assert client.get(bad).status_code == 404, bad
    assert client.get("/ui/vendor/hls.min.js").status_code == 200
    assert client.get("/ui/vendor/other.js").status_code == 404


@requires_pg
@requires_redis
def test_record_accuracy_roundtrip(client, fresh_db, tmp_path):
    import json

    f = tmp_path / "calibration_x.json"
    f.write_text(json.dumps({
        "dataset": "fleurs_ur", "model": "large-v3-turbo", "compute_type": "int8", "beam_size": 1,
        "word_level": [{"threshold": 0.9, "precision": 0.888, "precision_below": 0.41,
                        "kept_share": 0.861, "words_kept": 1103}],
        "segment_level": []}))
    assert asyncio.run(record(fresh_db, [str(f)])) == 3
    assert asyncio.run(record(fresh_db, [str(f)])) == 3  # idempotent
    rows = client.get("/accuracy").json()
    assert len(rows) == 3
    assert {r["metric"]: r["value"] for r in rows}["precision"] == pytest.approx(0.888)
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(ValueError):
        asyncio.run(record(fresh_db, [str(bad)]))


@requires_pg
@requires_redis
def test_write_endpoints_require_token_when_configured(settings, fresh_db, tmp_path):
    asyncio.run(migrate(fresh_db))
    settings.api.token = "s3cret-token"
    with TestClient(create_app(settings)) as c:
        body = {"id": "n1", "url": "udp://@239.1.1.1:5000"}
        assert c.post("/channels", json=body).status_code == 401
        assert c.post("/channels", json=body,
                      headers={"Authorization": "Bearer wrong"}).status_code == 401
        r = c.post("/channels", json=body, headers={"Authorization": "Bearer s3cret-token"})
        assert r.status_code == 201
        assert c.patch("/channels/n1", json={"enabled": False}).status_code == 401
        assert c.patch("/channels/n1", json={"enabled": False},
                       headers={"Authorization": "Bearer s3cret-token"}).status_code == 200
        assert c.get("/channels").status_code == 200  # reads stay open
        assert c.get("/health").status_code in (200, 503)
