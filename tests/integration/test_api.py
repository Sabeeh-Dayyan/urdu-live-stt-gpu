import asyncio
import json
import threading
import time

import pytest
import redis
from fastapi.testclient import TestClient

from live_stt.api.app import create_app
from live_stt.store.db import migrate
from tests.integration.conftest import REDIS, requires_pg, requires_redis

pytestmark = [requires_pg, requires_redis]
T0 = 1_763_000_000.0


@pytest.fixture
def client(settings, fresh_db):
    asyncio.run(migrate(fresh_db))
    with TestClient(create_app(settings)) as c:
        yield c


def test_channels_crud_and_validation(client):
    r = client.post("/channels", json={"id": "news1", "url": "udp://@239.1.1.1:5000",
                                       "audio_track": {"language": "urd"}})
    assert r.status_code == 201 and r.json()["audio_track"]["language"] == "urd"
    assert client.post("/channels", json={"id": "news1", "url": "udp://x:1"}).status_code == 409
    assert client.post("/channels", json={"id": "bad"}).status_code == 422  # url missing
    assert client.post("/channels", json={"id": "b", "url": "u", "bogus": 1}).status_code == 422
    r = client.patch("/channels/news1", json={"enabled": False, "url": "udp://@239.1.1.2:5000"})
    assert r.status_code == 200 and r.json()["enabled"] is False
    assert client.patch("/channels/nope", json={"enabled": True}).status_code == 404
    assert client.patch("/channels/news1", json={}).status_code == 400
    assert client.patch("/channels/news1", json={"name; DROP TABLE x": 1}).status_code == 422
    assert [c["id"] for c in client.get("/channels").json()] == ["news1"]


def _insert(dsn, rows):
    import asyncpg
    from datetime import datetime, timezone

    async def go():
        c = await asyncpg.connect(dsn)
        for ch, t, p, txt in rows:
            await c.execute(
                "INSERT INTO asr_segments(channel_id, segment_id, start_utc, end_utc, pass, "
                "text_raw, text_norm) VALUES ($1,$2,$3,$3,$4,$5,$5)",
                ch, f"{ch}:{int(t * 1000)}", datetime.fromtimestamp(t, timezone.utc), p, txt)
        await c.close()

    asyncio.run(go())


def test_transcripts_and_search(client, fresh_db):
    _insert(fresh_db, [("c1", T0, "fast", "وزیراعظم نے خطاب کیا"),
                       ("c1", T0 + 5, "refined", "قومی اسمبلی کا اجلاس"),
                       ("c2", T0 + 6, "fast", "100% گرین_لائن"),
                       ("c1", T0 + 99999, "fast", "بہت بعد")])
    r = client.get(f"/channels/c1/transcripts?from={T0}&to={T0 + 10}")
    assert [x["text_raw"] for x in r.json()] == ["وزیراعظم نے خطاب کیا", "قومی اسمبلی کا اجلاس"]
    r = client.get("/channels/c1/transcripts?pass=refined")
    assert len(r.json()) == 1
    iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(T0 + 1))
    assert len(client.get(f"/channels/c1/transcripts?from={iso}&to={T0 + 10}").json()) == 1
    assert client.get("/channels/c1/transcripts?from=yesterday").status_code == 400
    assert client.get("/channels/c1/transcripts?pass=bogus").status_code == 400
    assert client.get("/channels/c1/transcripts?limit=0").status_code == 422
    # search normalises the query (Arabic yeh/kaf variants) and escapes LIKE wildcards
    r = client.get("/search", params={"q": "اسمبلي"}).json()
    assert [x["text_raw"] for x in r["results"]] == ["قومی اسمبلی کا اجلاس"]
    assert client.get("/search", params={"q": "%"}).status_code == 400  # empty after normalise
    assert client.get("/search", params={"q": "100"}).json()["results"][0]["channel_id"] == "c2"
    assert client.get("/search", params={"q": "' OR 1=1 --"}).json()["results"] == []
    assert client.get("/search", params={"q": "x" * 500}).status_code == 422


def test_health_and_metrics(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["redis"] and r.json()["db"]
    assert "stt_api_ws_clients" in client.get("/metrics").text
    assert client.get("/").status_code == 200


def test_websocket_live_filtering(client, settings):
    r = redis.Redis.from_url(REDIS)

    def pub():
        time.sleep(0.5)
        for ch in ("other", "c1"):
            r.publish(settings.bus.pubsub_live, json.dumps({"channel_id": ch, "type": "final",
                                                            "text_raw": f"from {ch}"}))
        r.publish(settings.bus.pubsub_live, b"not json")

    threading.Thread(target=pub, daemon=True).start()
    with client.websocket_connect("/ws/live?channels=c1") as ws:
        msg = json.loads(ws.receive_text())
        assert msg["channel_id"] == "c1"


def test_api_degrades_when_db_down(settings):
    settings.db.dsn = "postgresql://stt:stt@localhost:5499/none"  # nothing listens
    with TestClient(create_app(settings)) as c:
        assert c.get("/channels").status_code == 503
        h = c.get("/health")
        assert h.status_code == 503 and h.json()["db"] is False and h.json()["redis"] is True
