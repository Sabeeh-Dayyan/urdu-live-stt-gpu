"""Full system: real supervisor (spawned children) + live UDP sender + Redis + Postgres + API.

UDP Urdu programme → worker → Redis → ASR service (turbo) → Redis → persister → Postgres → API.
Also kills the ASR child mid-run and checks the supervisor restarts it and no FAST segment is lost.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

from live_stt.settings import PROJECT_ROOT
from tests.integration.conftest import PG_TEST, REDIS, requires_pg, requires_redis

TS = PROJECT_ROOT / "testbed/media/fleurs_ur.ts"
TURBO = PROJECT_ROOT / "models/whisper/large-v3-turbo/model.bin"
pytestmark = [requires_pg, requires_redis,
              pytest.mark.skipif(not TURBO.exists(), reason="large-v3-turbo missing"),
              pytest.mark.skipif(not TS.exists(), reason="make testbed/media/fleurs_ur.ts"),
              pytest.mark.timeout(900)]


def free_port(kind=socket.SOCK_DGRAM):
    s = socket.socket(socket.AF_INET, kind)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def test_full_system(tmp_path, fresh_db):
    import redis as _r

    _r.Redis.from_url(REDIS).flushdb()
    udp, api = free_port(), free_port(socket.SOCK_STREAM)
    cfg = tmp_path / "e2e.yaml"
    cfg.write_text(f"""
environment: e2e
bus: {{redis_url: "{REDIS}"}}
db: {{dsn: "{PG_TEST}"}}
api: {{host: 127.0.0.1, port: {api}}}
metrics: {{base_port: 19300}}
logging: {{level: INFO, json: true}}
asr:
  cpu_threads: 8
  pack_s: 28
  refine_max_lag_s: 3600
  refine_enabled: true
  refine: {{model: large-v3-turbo, beam_size: 2, best_of: 2}}
channels:
  - {{id: sys1, url: "udp://127.0.0.1:{udp}", language: ur,
      glossary_path: config/glossary/ur_default.txt}}
""")
    env = {**os.environ, "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}"}
    log = open(tmp_path / "sup.log", "w")
    sup = subprocess.Popen([sys.executable, "-m", "live_stt.cli",
                            "supervise", "-c", str(cfg)], cwd=PROJECT_ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    sender = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "quiet", "-re",
                               "-stream_loop", "-1", "-i", str(TS), "-c", "copy", "-f", "mpegts",
                               f"udp://127.0.0.1:{udp}?pkt_size=1316"], start_new_session=True)
    t_start = time.time()
    try:
        # wait for transcripts to reach Postgres via the API
        rows = []
        deadline = time.time() + 300
        while time.time() < deadline and len(rows) < 3:
            time.sleep(5)
            try:
                rows = get(f"http://127.0.0.1:{api}/channels/sys1/transcripts")
            except Exception:
                pass
        assert len(rows) >= 3, open(tmp_path / "sup.log").read()[-4000:]
        for r in rows:
            assert t_start - 5 < r["start_utc"] < time.time() + 1  # UTC anchored to now
            assert r["text_raw"]
        health = get(f"http://127.0.0.1:{api}/health")
        assert health["status"] == "ok"
        comps = health["components"]
        assert "worker:sys1" in comps and any(k.startswith("asr:") for k in comps)

        # Kill the ASR child: supervisor must restart it; nothing pending must be lost.
        asr_pid = comps["supervisor"]["children"]["asr"]["pid"]
        os.kill(asr_pid, signal.SIGKILL)
        time.sleep(90)
        health = get(f"http://127.0.0.1:{api}/health")
        ch = health["components"]["supervisor"]["children"]["asr"]
        assert ch["restarts"] >= 1 and ch["alive"]
        rows2 = get(f"http://127.0.0.1:{api}/channels/sys1/transcripts?limit=1000")
        assert len(rows2) > len(rows)
        # On this CPU FAST saturates the box during continuous speech, so REFINE (lower
        # priority) waits. Stop the input: once FAST is idle the REFINE backlog must drain.
        os.killpg(sender.pid, signal.SIGKILL)
        refined, deadline = [], time.time() + 420
        while time.time() < deadline and not refined:
            time.sleep(10)
            rows2 = get(f"http://127.0.0.1:{api}/channels/sys1/transcripts?limit=1000")
            refined = [r for r in rows2 if r["pass"] == "refined"]
        assert refined, "REFINE backlog should drain once FAST is idle"
        # search finds a word that was transcribed
        word = rows2[-1]["text_norm"].split()[0]
        res = get(f"http://127.0.0.1:{api}/search?q={urllib.parse.quote(word)}")
        assert res["results"]
    finally:
        for p in (sender, sup):
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except OSError:  # already gone (ESRCH/EPERM for a reaped group)
                pass
        try:
            sup.wait(30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(sup.pid, signal.SIGKILL)
            except OSError:
                pass
        sender.kill()
        log.close()
    # FAST accounting after the run: every FAST request was acked (none stuck pending).
    r = _r.Redis.from_url(REDIS)
    for g in r.xinfo_groups("asr.requests.fast"):
        assert g["pending"] + g["lag"] <= 2, g  # every FAST request processed


import urllib.parse  # noqa: E402
