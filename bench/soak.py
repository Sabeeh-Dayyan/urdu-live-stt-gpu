"""Soak test monitor (`python -m bench.soak --hours 24`).

Run alongside `stt supervise` + looping senders. Every ``interval`` seconds it records, from the
Redis health keys and pub/sub:
* RSS (MB) of every supervised process (ps) → RSS slope per process (MB/hour, 2nd half of run)
* worker stats: segments emitted, lost (audio overrun), decode errors, drift_ms, reconnects
* ASR: FAST/REFINE queue depth and lag
* FAST accounting: segments emitted by workers vs. `final`+`error` events seen on pub/sub
Writes bench/out/soak.csv and bench/out/soak_summary.json (also on Ctrl-C).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import redis

sys.path.insert(0, "src")
from live_stt.settings import load_settings  # noqa: E402


def rss_mb(pid: int) -> float | None:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        return int(out) / 1024 if out else None
    except Exception:  # noqa: BLE001
        return None


class Counter:
    def __init__(self) -> None:
        self.types: dict[str, int] = {}
        self.fast_done: dict[str, int] = {}
        self.lock = threading.Lock()

    def listen(self, r: redis.Redis, channel: str) -> None:
        ps = r.pubsub(ignore_subscribe_messages=True)
        ps.subscribe(channel)
        for m in ps.listen():
            try:
                e = json.loads(m["data"])
            except ValueError:
                continue
            with self.lock:
                t = e.get("type")
                self.types[t] = self.types.get(t, 0) + 1
                if t in ("final", "error") and e.get("pass") == "fast":
                    ch = e.get("channel_id")
                    self.fast_done[ch] = self.fast_done.get(ch, 0) + 1


def sample(r: redis.Redis, t0: float, cnt: Counter) -> tuple[dict, dict]:
    sup = json.loads(r.get("stt:health:supervisor") or "{}")
    row: dict = {"t_h": round((time.time() - t0) / 3600, 4)}
    for name, c in (sup.get("children") or {}).items():
        row[f"rss_{name}"] = rss_mb(c["pid"]) if c.get("pid") else None
        row[f"restarts_{name}"] = c.get("restarts")
    segs = {}
    for k in r.scan_iter("stt:health:worker:*"):
        w = json.loads(r.get(k) or "{}")
        ch = k.rsplit(":", 1)[1]
        segs[ch] = w.get("segments", 0)
        row[f"drift_ms_{ch}"] = w.get("drift_ms")
        row[f"lost_{ch}"] = w.get("lost")
        row[f"decode_err_{ch}"] = w.get("decode_errors")
        row[f"connects_{ch}"] = w.get("connects")
    for k in r.scan_iter("stt:health:asr:*"):
        h = json.loads(r.get(k) or "{}")
        row["fast_depth"] = h.get("fast_depth")
        row["refine_depth"] = h.get("refine_depth")
        row["fast_lag_s"] = h.get("fast_lag_s")
    with cnt.lock:
        done = dict(cnt.fast_done)
        row.update({f"n_{k}": v for k, v in cnt.types.items()})
    return row, {"segs": segs, "done": done}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--config", default="config/dev-mac-cpu.yaml")
    ap.add_argument("--out", default="bench/out")
    a = ap.parse_args()
    s = load_settings(a.config)
    r = redis.Redis.from_url(s.bus.redis_url, decode_responses=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cnt = Counter()
    threading.Thread(target=cnt.listen, args=(r, s.bus.pubsub_live), daemon=True).start()
    t0 = time.time()
    rows: list[dict] = []
    base = None
    fields: list[str] = []
    try:
        with open(out / "soak.csv", "w", newline="") as fh:
            wr = None
            while time.time() - t0 < a.hours * 3600:
                time.sleep(a.interval)
                row, acc = sample(r, t0, cnt)
                if base is None:
                    base = acc
                row["fast_emitted"] = sum(acc["segs"].values()) - sum(base["segs"].values())
                row["fast_done"] = sum(acc["done"].values()) - sum(base["done"].values())
                rows.append(row)
                if wr is None:
                    fields = list(row) + ["n_final", "n_refined", "n_partial", "n_error",
                                          "n_refine_skipped"]
                    wr = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(fields)),
                                        extrasaction="ignore")
                    wr.writeheader()
                wr.writerow(row)
                fh.flush()
                print(json.dumps(row), flush=True)
    except KeyboardInterrupt:
        pass
    summarize(rows, out, a.hours)


def summarize(rows: list[dict], out: Path, hours: float) -> dict:
    summ: dict = {"hours_requested": hours, "hours_run": rows[-1]["t_h"] if rows else 0,
                  "samples": len(rows), "rss_slope_mb_per_h": {}, "rss_first_mb": {},
                  "rss_last_mb": {}, "max_abs_drift_ms": {}}
    if not rows:
        (out / "soak_summary.json").write_text(json.dumps(summ, indent=1))
        return summ
    half = rows[len(rows) // 2:]
    keys = {k for r in rows for k in r}
    for k in sorted(keys):
        if k.startswith("rss_"):
            pts = [(x["t_h"], x[k]) for x in half if x.get(k) is not None]
            if len(pts) >= 3:
                t, v = np.array(pts, dtype=float).T
                summ["rss_slope_mb_per_h"][k[4:]] = round(float(np.polyfit(t, v, 1)[0]), 2)
            vals = [x[k] for x in rows if x.get(k) is not None]
            if vals:
                summ["rss_first_mb"][k[4:]] = round(vals[0], 1)
                summ["rss_last_mb"][k[4:]] = round(vals[-1], 1)
        if k.startswith("drift_ms_"):
            summ["max_abs_drift_ms"][k[9:]] = max(abs(x.get(k) or 0) for x in rows)
    last = rows[-1]
    summ["fast_emitted"] = last["fast_emitted"]
    summ["fast_done"] = last["fast_done"]
    summ["fast_in_flight"] = last.get("fast_depth") or 0
    summ["fast_unaccounted"] = max(0, last["fast_emitted"] - last["fast_done"] -
                                   summ["fast_in_flight"])
    (out / "soak_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    return summ


if __name__ == "__main__":
    main()
