"""Add the "Live transcript & accuracy" section to infra/grafana/dashboards/live-stt.json.

Idempotent: panels with id >= 100 are regenerated, the original metric panels (ids < 100) are kept
and placed below. Run:  python infra/grafana/build_dashboard.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DASH = Path(__file__).parent / "dashboards" / "live-stt.json"
PG = {"type": "grafana-postgresql-datasource", "uid": "${pg}"}
PROM = {"type": "prometheus", "uid": "prometheus"}
UI = "${ui_url}"  # dashboard variable (textbox); default from STT_UI_URL at build time
CH_FILTER = "channel_id IN (${channel:sqlstring})"


def sql(ref: str, q: str, fmt: str = "table") -> dict:
    return {"refId": ref, "datasource": PG, "editorMode": "code", "rawQuery": True,
            "format": fmt, "rawSql": " ".join(q.split())}


def stat(pid, title, x, y, w, target, unit="percentunit", decimals=0, desc="", ds=PG,
         thresholds=None):
    steps = thresholds or [{"color": "text", "value": None}]
    return {"id": pid, "type": "stat", "title": title, "description": desc, "datasource": ds,
            "gridPos": {"x": x, "y": y, "w": w, "h": 4},
            "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals,
                                         "thresholds": {"mode": "absolute", "steps": steps}},
                            "overrides": []},
            "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                        "colorMode": "value", "graphMode": "none", "textMode": "auto"},
            "targets": [target]}


CONF_STEPS = [{"color": "red", "value": None}, {"color": "orange", "value": 0.8},
              {"color": "green", "value": 0.9}]


def panels() -> list[dict]:
    y = 0
    out = [{"id": 100, "type": "row", "title": "Live transcript & accuracy", "collapsed": False,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []}]
    y += 1
    out.append({
        "id": 101, "type": "text", "title": "Live player (video + transcript)",
        "description": "The live-stt web page embedded. Click any sentence to play the video from "
                       "that moment; LIVE returns to the live edge.",
        "gridPos": {"x": 0, "y": y, "w": 14, "h": 20},
        "options": {"mode": "html", "content":
                    f'<iframe src="{UI}/?embed=1&min_conf=${{min_conf}}" '
                    'style="width:100%;height:100%;border:0" allow="autoplay"></iframe>'},
    })
    # live stats (Prometheus, last 5 min)
    out.append(stat(102, "Avg confidence (live, 5 min)", 14, y, 5, {
        "refId": "A", "datasource": PROM,
        "expr": "sum(rate(stt_asr_segment_confidence_sum[5m])) / "
                "sum(rate(stt_asr_segment_confidence_count[5m]))"},
        decimals=1, ds=PROM, thresholds=CONF_STEPS,
        desc="Mean word probability of final+refined segments — model confidence, not accuracy."))
    out.append(stat(103, "Segments > 90% confidence (live, 5 min)", 19, y, 5, {
        "refId": "A", "datasource": PROM,
        "expr": "1 - sum(rate(stt_asr_segment_confidence_bucket{le=\"0.9\"}[5m])) / "
                "sum(rate(stt_asr_segment_confidence_count[5m]))"}, ds=PROM,
        desc="Share of segments in the last 5 minutes whose confidence is above 90 %."))
    # range stats (PostgreSQL, dashboard time range)
    rng = f"$__timeFilter(start_utc) AND {CH_FILTER}"
    out.append(stat(104, "Avg confidence (range)", 14, y + 4, 5, sql(
        "A", f"SELECT avg(confidence) FROM asr_segments WHERE {rng}"),
        decimals=1, thresholds=CONF_STEPS,
        desc="Average model confidence of stored segments in the selected time range."))
    out.append(stat(105, "Segments ≥ min confidence (range)", 19, y + 4, 5, sql(
        "A", f"SELECT avg(CASE WHEN confidence >= $min_conf THEN 1.0 ELSE 0.0 END) "
             f"FROM asr_segments WHERE {rng}"),
        desc="Share of stored segments at or above the Min confidence variable."))
    out.append(stat(106, "Segments (range)", 14, y + 8, 5, sql(
        "A", f"SELECT count(*) FROM asr_segments WHERE {rng}"), unit="none",
        desc="Number of stored transcript segments (≈ 8–10 s of speech each) in the time range."))
    out.append(stat(107, "Refined share (range)", 19, y + 8, 5, sql(
        "A", f"SELECT avg(CASE WHEN pass='refined' THEN 1.0 ELSE 0.0 END) FROM asr_segments "
             f"WHERE {rng}"),
        desc="Share of segments whose REFINE pass completed (the rest keep FAST text; "
             "refine_skipped is flagged)."))
    out.append(stat(108, "Measured word accuracy — FAST (reference set)", 14, y + 12, 5, sql(
        "A", "SELECT 1 - value FROM accuracy_results WHERE kind='bench' AND metric='wer' "
             "AND config LIKE 'large-v3-turbo int8 b1 seg10%' ORDER BY recorded_at DESC LIMIT 1"),
        decimals=1, thresholds=CONF_STEPS,
        desc="1 − WER against hand-made reference transcripts (stt bench → stt record-accuracy). "
             "Read speech (FLEURS) until broadcast references exist."))
    out.append(stat(109, "Words ≥ 90% conf: measured correct", 19, y + 12, 5, sql(
        "A", "SELECT value FROM accuracy_results WHERE kind='calibration_word' "
             "AND metric='precision' AND threshold BETWEEN 0.899 AND 0.901 "
             "ORDER BY recorded_at DESC LIMIT 1"),
        decimals=1, thresholds=CONF_STEPS,
        desc="Of the words the model rated ≥ 90% confident, the share that matched the reference "
             "(bench/calibration.py)."))
    out.append(stat(110, "Words ≥ 90% conf: share of all words", 14, y + 16, 5, sql(
        "A", "SELECT value FROM accuracy_results WHERE kind='calibration_word' "
             "AND metric='kept_share' AND threshold BETWEEN 0.899 AND 0.901 "
             "ORDER BY recorded_at DESC LIMIT 1"), decimals=1,
        desc="How many of all words the model rated ≥ 90 % confident (reference set)."))
    out.append(stat(111, "Refine skipped (range)", 19, y + 16, 5, sql(
        "A", f"SELECT count(*) FROM asr_segments WHERE {rng} "
             "AND EXISTS (SELECT 1 FROM unnest(flags) f WHERE f LIKE 'refine_skipped%')"),
        unit="none", desc="Segments whose second pass was skipped because the CPU was busy "
                          "(they keep the live FAST text)."))
    y += 20
    out.append({
        "id": 112, "type": "table",
        "title": "Transcript — click the text to open that moment with video",
        "description": "Stored transcript segments in the selected time range (top-right picker), "
                       "filtered by Channel and Min confidence (top-left). Click a row's text to "
                       "open the player at that moment.",
        "datasource": PG, "gridPos": {"x": 0, "y": y, "w": 16, "h": 14},
        "targets": [sql("A", f"""
            SELECT start_utc AS "time", channel_id, pass,
                   round(confidence::numeric, 3) AS confidence, text_raw AS text,
                   extract(epoch FROM start_utc) AS t
            FROM asr_segments
            WHERE {rng} AND ($min_conf = 0 OR confidence >= $min_conf)
            ORDER BY start_utc DESC LIMIT 500""")],
        "fieldConfig": {"defaults": {"custom": {"align": "auto", "cellOptions": {"type": "auto"}}},
                        "overrides": [
            {"matcher": {"id": "byName", "options": "t"},
             "properties": [{"id": "custom.hidden", "value": True}]},
            {"matcher": {"id": "byName", "options": "time"},
             "properties": [{"id": "custom.width", "value": 190}]},
            {"matcher": {"id": "byName", "options": "channel_id"},
             "properties": [{"id": "custom.width", "value": 90}]},
            {"matcher": {"id": "byName", "options": "pass"},
             "properties": [{"id": "custom.width", "value": 80}]},
            {"matcher": {"id": "byName", "options": "confidence"},
             "properties": [{"id": "custom.width", "value": 100}, {"id": "unit", "value": "percentunit"},
                            {"id": "custom.cellOptions", "value": {"type": "color-text"}},
                            {"id": "thresholds", "value": {"mode": "absolute", "steps": CONF_STEPS}}]},
            {"matcher": {"id": "byName", "options": "text"},
             "properties": [{"id": "links", "value": [{
                 "title": "Open this moment with video",
                 "url": UI + "/?ch=${__data.fields.channel_id}&t=${__data.fields.t}",
                 "targetBlank": True}]}]},
        ]},
        "options": {"showHeader": True, "cellHeight": "md"},
    })
    out.append({
        "id": 113, "type": "table", "title": "Measured accuracy (reference transcripts)",
        "description": "From result files recorded with `stt record-accuracy`. accuracy = 1 − WER.",
        "datasource": PG, "gridPos": {"x": 16, "y": y, "w": 8, "h": 14},
        "targets": [sql("A", """
            SELECT dataset, kind, config, metric, threshold, round(value::numeric, 3) AS value, n
            FROM accuracy_results
            WHERE (kind = 'bench' AND metric IN ('wer','cer','rtf'))
               OR (kind LIKE 'calibration%' AND threshold IN (0, 0.8, 0.9, 0.95))
            ORDER BY dataset, kind, config, metric, threshold""")],
        "options": {"showHeader": True, "cellHeight": "sm"},
    })
    y += 14
    out.append({
        "id": 114, "type": "timeseries", "title": "Model confidence over time (5 min)",
        "description": "Average model confidence per channel and pass. A drop often means music, "
                       "noise, crosstalk or a language the model handles poorly.",
        "datasource": PROM, "gridPos": {"x": 0, "y": y, "w": 24, "h": 7},
        "fieldConfig": {"defaults": {"unit": "percentunit", "min": 0, "max": 1}, "overrides": []},
        "targets": [{"refId": "A", "datasource": PROM,
                     "expr": "sum by (channel, pass) (rate(stt_asr_segment_confidence_sum[5m])) / "
                             "sum by (channel, pass) (rate(stt_asr_segment_confidence_count[5m]))",
                     "legendFormat": "{{channel}} {{pass}}"}],
    })
    y += 7
    out.append({"id": 115, "type": "row", "title": "Pipeline metrics", "collapsed": False,
                "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []})
    return out


def templating() -> dict:
    return {"list": [
        {"name": "pg", "label": "Database", "type": "datasource",
         "query": "grafana-postgresql-datasource", "current": {}},  # first available
        {"name": "ui_url", "label": "Player URL", "type": "textbox",
         "query": os.environ.get("STT_UI_URL", "http://localhost:8000"),
         "current": {"text": os.environ.get("STT_UI_URL", "http://localhost:8000"),
                     "value": os.environ.get("STT_UI_URL", "http://localhost:8000")},
         "description": "Where the live-stt web UI is reachable from your browser "
                        "(e.g. http://your-server:8000)."},
        {"name": "channel", "label": "Channel", "type": "query", "datasource": PG,
         "query": "SELECT id FROM channels ORDER BY id", "refresh": 1, "multi": True,
         "includeAll": True, "current": {"text": "All", "value": "$__all"}},
        {"name": "min_conf", "label": "Min confidence", "type": "custom",
         "query": "0,0.8,0.9,0.95", "current": {"text": "0.9", "value": "0.9"},
         "options": [{"text": v, "value": v, "selected": v == "0.9"}
                     for v in ("0", "0.8", "0.9", "0.95")]},
    ]}


OLD_DESCRIPTIONS = {  # shown in the panel's (i) tooltip
    1: "How long after someone stops speaking the live (FAST) text appears. p50 = half of the "
       "sentences are faster than this; p95 = 95 % are. Rising lines = the CPU is falling behind.",
    2: "Same, for the careful second pass (REFINE). On CPU it only runs when FAST is idle, so it "
       "can be minutes behind or empty during continuous speech.",
    3: "Time from speech start to the first partial caption (only in streaming mode; empty in "
       "segment mode).",
    4: "Age of the oldest waiting transcription job. FAST lag growing without limit = overloaded.",
    5: "How many transcription jobs are waiting. FAST should stay near 0–2; REFINE may grow "
       "(it is capped and older work is skipped, flagged refine_skipped).",
    6: "Real-time factor: seconds of compute per second of audio. Below 1 = faster than real time.",
    7: "1 when the channel's UDP stream is being received, 0 when it is down.",
    8: "Difference between the stream's own clock (PTS) and the computer clock. Stays near 0; "
       "spikes at loop restarts / encoder resets are re-anchored automatically.",
    9: "Share of the last minute that the voice detector classified as speech (music/silence "
       "lower it).",
    10: "Stream events per minute: up/down, discontinuities (timestamp jumps), drift corrections.",
    11: "Segments removed or flagged by the hallucination guards, and REFINE passes skipped "
        "under load.",
    12: "Speech segments sent for transcription per minute (silence = cut at a pause, "
        "forced = cut at max length inside continuous speech).",
    13: "Incoming audio bitrate per channel — a flat line at 0 means no packets arrive.",
}


def main() -> None:
    d = json.loads(DASH.read_text(encoding="utf-8"))
    old = [p for p in d["panels"] if p.get("id", 0) < 100]
    for p in old:
        if p.get("id") in OLD_DESCRIPTIONS:
            p["description"] = OLD_DESCRIPTIONS[p["id"]]
    new = panels()
    top = max(p["gridPos"]["y"] + p["gridPos"]["h"] for p in new)
    base = min(p["gridPos"]["y"] for p in old) if old else 0
    for p in old:
        p["gridPos"]["y"] = p["gridPos"]["y"] - base + top
    d["panels"] = new + old
    d["templating"] = templating()
    d["version"] = d.get("version", 1) + 1
    d["timezone"] = "browser"
    DASH.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{DASH}: {len(new)} new + {len(old)} existing panels")


if __name__ == "__main__":
    main()
