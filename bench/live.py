"""Live latency measurement from bus events (`stt bench-live`).

Subscribes to transcript pub/sub for N seconds and reports p50/p95/p99 of
* speech_end→final, speech_end→refined, speech→partial (from event timings)
* queue wait (t_infer_start − t_enqueued) and RTF per pass
* timestamp error vs. the reference programme: stream position of the first word =
  (word_utc − sender_start) mod programme_duration, compared to the nearest true utterance start
  (needs testbed/run/sender_<port>.start and the reference JSON used to build the TS).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import redis


def pct(xs):
    if not xs:
        return {"n": 0, "p50": None, "p95": None, "p99": None}
    a = np.asarray(xs, float)
    return {"n": len(xs), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99))}


def ts_layout(ts: str) -> tuple[float, float]:
    """(loop period, audio start offset) of the looped TS, from ffprobe."""
    import subprocess

    fmt = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,start_time",
                          "-of", "json", ts], capture_output=True, text=True, check=True)
    aud = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=start_time", "-of", "json", ts],
                         capture_output=True, text=True, check=True)
    f = json.loads(fmt.stdout)["format"]
    a0 = float(json.loads(aud.stdout)["streams"][0]["start_time"])
    return float(f["duration"]), a0 - float(f.get("start_time", a0))


def main(s, seconds: float, out: str, reference: str = "testbed/reference/fleurs_ur.json",
         ts: str = "testbed/media/fleurs_ur.ts") -> dict:
    r = redis.Redis.from_url(s.bus.redis_url)
    ps = r.pubsub(ignore_subscribe_messages=True)
    ps.subscribe(s.bus.pubsub_live)
    t_end = time.time() + seconds
    evs = []
    while time.time() < t_end:
        m = ps.get_message(timeout=1.0)
        if m:
            evs.append(json.loads(m["data"]))
    lat = {"speech_end→final": [], "speech_end→refined": [], "speech→partial": [],
           "queue_wait_fast": [], "queue_wait_refine": [], "rtf_fast": [], "rtf_refine": []}
    for e in evs:
        t = e.get("timings") or {}
        if e.get("type") == "final":
            lat["speech_end→final"].append(t.get("latency_s"))
        elif e.get("type") == "refined":
            lat["speech_end→refined"].append(t.get("latency_s"))
        elif e.get("type") == "partial":
            lat["speech→partial"].append(t.get("latency_s"))
        p = e.get("pass")
        if t.get("t_infer_start") and t.get("t_enqueued") and p in ("fast", "refine"):
            lat[f"queue_wait_{p}"].append(t["t_infer_start"] - t["t_enqueued"])
            lat[f"rtf_{p}"].append(t.get("rtf"))
    notes = {}
    # timestamp accuracy vs reference programme
    try:
        ref = json.loads(Path(reference).read_text("utf-8"))
        dur, a_off = ts_layout(ts)
        starts = sorted(Path("testbed/run").glob("sender_*.start"))
        sender = {p.stem.split("_")[1]: float(p.read_text()) for p in starts}
        port_of = {c.id: c.url.rsplit(":", 1)[-1] for c in s.channels}
        errs = []
        for e in evs:
            if e.get("type") != "final" or not e.get("words"):
                continue
            port = port_of.get(e.get("channel_id"))
            if port not in sender:
                continue
            pos = ((e["words"][0]["s"] - sender[port]) % dur) - a_off
            near = min(ref["segments"], key=lambda u: abs(u.get("speech_start", u["start"]) - pos))
            onset = near.get("speech_start", near["start"])
            if abs(onset - pos) < 3.0:
                errs.append(pos - onset)
        lat["word_ts_error_s (signed)"] = errs
        if errs:
            notes["timestamp"] = (f"median signed error {np.median(errs):+.3f}s vs VAD speech onsets "
                                  "(approximate: includes ffmpeg sender start-up/initial-burst "
                                  "timing; precise accuracy evidence is in the e2e test)")
    except Exception as ex:  # noqa: BLE001
        notes["timestamp"] = f"not computed: {ex}"
    res = {"seconds": seconds, "events": len(evs),
           "channels": sorted({e.get("channel_id") for e in evs if e.get("channel_id")}),
           "config": s.environment,
           "percentiles": {k: pct([x for x in v if x is not None]) for k, v in lat.items()},
           "notes": notes}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res["percentiles"], indent=1, ensure_ascii=False))
    return res
