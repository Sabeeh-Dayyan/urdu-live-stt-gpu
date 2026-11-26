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
         thresholds=None, h=4, graph="none", no_value="–", mappings=None, color_mode="value"):
    steps = thresholds or [{"color": "text", "value": None}]
    return {"id": pid, "type": "stat", "title": title, "description": desc, "datasource": ds,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals, "noValue": no_value,
                                         "mappings": mappings or [],
                                         "color": {"mode": "thresholds"},
                                         "thresholds": {"mode": "absolute", "steps": steps}},
                            "overrides": []},
            "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                        "colorMode": color_mode, "graphMode": graph, "textMode": "value",
                        "justifyMode": "center", "wideLayout": True,
                        "text": {"titleSize": 13, "valueSize": 30}},
            "targets": [target]}


def prom(expr, ref="A", legend=""):
    return {"refId": ref, "datasource": PROM, "expr": expr, "legendFormat": legend}


def ts(pid, title, x, y, w, h, targets, unit="s", desc="", minv=None, maxv=None, steps=None):
    d = {"unit": unit, "custom": {"drawStyle": "line", "lineWidth": 2, "fillOpacity": 18,
                                  "gradientMode": "opacity", "showPoints": "never",
                                  "spanNulls": True, "lineInterpolation": "smooth"},
         "color": {"mode": "palette-classic"}}
    if minv is not None:
        d["min"] = minv
    if maxv is not None:
        d["max"] = maxv
    if steps:
        d["thresholds"] = {"mode": "absolute", "steps": steps}
        d["custom"]["thresholdsStyle"] = {"mode": "dashed"}
    return {"id": pid, "type": "timeseries", "title": title, "description": desc,
            "datasource": PROM, "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "fieldConfig": {"defaults": d, "overrides": []},
            "options": {"legend": {"displayMode": "list", "placement": "bottom",
                                   "calcs": ["lastNotNull"]},
                        "tooltip": {"mode": "multi", "sort": "desc"}},
            "targets": targets}


def row(pid, title, y, collapsed=False, inner=None):
    return {"id": pid, "type": "row", "title": title, "collapsed": collapsed,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": inner or []}


CONF_STEPS = [{"color": "red", "value": None}, {"color": "orange", "value": 0.8},
              {"color": "green", "value": 0.9}]
DELAY_STEPS = [{"color": "green", "value": None}, {"color": "orange", "value": 20},
               {"color": "red", "value": 60}]
ch_sel = 'channel="$channel"'
PILL = ("padding:5px 12px;border-radius:999px;border:1px solid rgba(128,128,128,.35);"
        "text-decoration:none;font-size:13.5px;font-weight:600;white-space:nowrap;color:#1f8f84")
THEME_LINK = ("?orgId=1&theme={t}&var-ui_theme={t}&var-channel=${{channel}}&var-pg=${{pg}}"
              "&var-ui_url=${{ui_url}}&var-min_conf=${{min_conf}}&from=now-30m&to=now&refresh=5s")


def header() -> dict:
    links = [("Find a moment", "/ui/archive"), ("Reports", "/ui/reports"),
             ("Upload a file", "/ui/uploads"), ("Channel setup", "/ui/channels"),
             ("System &amp; links", "/ui/system"), ("API docs", "/docs")]
    theme = (f'<span style="display:inline-flex;border:1px solid rgba(128,128,128,.35);'
             f'border-radius:999px;overflow:hidden;font-size:13px">'
             f'<a href="{THEME_LINK.format(t="light")}" style="padding:5px 12px;'
             f'text-decoration:none">☀ Light</a><a href="{THEME_LINK.format(t="dark")}" '
             f'style="padding:5px 12px;text-decoration:none;border-left:1px solid '
             f'rgba(128,128,128,.35)">☾ Dark</a></span>')
    html = (
        '<div style="display:flex;align-items:center;gap:12px 16px;flex-wrap:wrap;'
        'padding:2px 64px 2px 2px">'  # right padding keeps clear of Grafana's hover menu
        '<div style="width:40px;height:40px;border-radius:12px;display:grid;place-items:center;'
        'color:#fff;font-size:20px;background:#16191d;box-shadow:inset 0 -3px 0 #c96a12">◉</div>'
        '<div><div style="font-size:20px;font-weight:700;line-height:1.2">Urdu Live '
        'Transcription</div><div style="font-size:13px;opacity:.7">Channel <b>${channel}</b> · '
        'live monitoring, refreshed every 5 s</div></div>' + theme +
        '<span style="flex:1"></span>'
        f'<a href="{UI}/" target="_blank" rel="noopener" style="{PILL};background:#0b6e66;'
        'color:#fff;border-color:#0b6e66">▶ Open live app</a>' +
        "".join(f'<a href="{UI}{h}" target="_blank" rel="noopener" style="{PILL}">{t}</a>'
                for t, h in links) + "</div>")
    return {"id": 120, "type": "text", "title": "", "transparent": True,
            "description": "", "gridPos": {"x": 0, "y": 0, "w": 24, "h": 2},
            "options": {"mode": "html", "content": html}}


def panels() -> list[dict]:
    out = [header()]
    y = 2
    # ---------------------------------------------------------------- live now
    out.append({
        "id": 101, "type": "text", "title": "Live video & transcript",
        "description": "The live page: video and transcript side by side. The highlighted line "
                       "follows the video. Click any line to replay it; 'Go to' opens any time.",
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 20},
        "options": {"mode": "html", "content":
                    f'<iframe src="{UI}/?embed=1&ch=${{channel}}&theme=${{ui_theme}}" '
                    'style="width:100%;height:100%;border:0" allow="autoplay; fullscreen">'
                    '</iframe>'},
    })
    y += 20
    out.append(row(131, "● Live now", y))
    y += 1
    up_map = [{"type": "value", "options": {
        "0": {"text": "No signal", "color": "red", "index": 0},
        "1": {"text": "Receiving", "color": "green", "index": 1}}}]
    out.append(stat(132, "Stream", 0, y, 4, prom(f"max(stt_stream_up{{{ch_sel}}})"), unit="none",
                    ds=PROM, mappings=up_map, color_mode="background", h=5,
                    no_value="No signal", thresholds=[{"color": "red", "value": None},
                                                    {"color": "green", "value": 1}],
                    desc="Is the channel's stream arriving right now (UDP packets decoded)."))
    out.append(stat(133, "Text delay", 4, y, 4, prom(
        f"histogram_quantile(0.5, sum by (le) (rate(stt_latency_speech_end_to_final_seconds_bucket"
        f"{{{ch_sel}}}[2m])))"), unit="s", decimals=1, ds=PROM, h=5, graph="area",
        thresholds=DELAY_STEPS, no_value="waiting for speech",
        desc="Median time from the end of a spoken segment to its live text, last 2 minutes."))
    out.append(stat(134, "Behind live", 8, y, 4, prom(
        'max(stt_asr_queue_lag_seconds{pass="fast"})'), unit="s", decimals=0, ds=PROM, h=5,
        graph="area", thresholds=DELAY_STEPS,
        desc="Age of the oldest audio waiting for transcription. Near 0 = keeping up. "
             "Above asr.fast_max_lag_s (CPU: 120 s) audio is skipped and counted below."))
    out.append(stat(135, "Model confidence", 12, y, 4, prom(
        f"sum(rate(stt_asr_segment_confidence_sum{{{ch_sel}}}[2m])) / "
        f"sum(rate(stt_asr_segment_confidence_count{{{ch_sel}}}[2m]))"), decimals=1, ds=PROM, h=5,
        graph="area", thresholds=CONF_STEPS,
        desc="Average word probability of the text produced in the last 2 minutes. The model's "
             "own estimate, not verified accuracy."))
    out.append(stat(136, "Speech in audio", 16, y, 4, prom(f"max(stt_vad_speech_ratio{{{ch_sel}}})"),
                    ds=PROM, h=5, graph="area",
                    desc="Share of the last minute the voice detector heard speech (music, "
                         "ads and silence lower it)."))
    out.append(stat(137, "Segments / min", 20, y, 4, prom(
        f"sum(increase(stt_segments_emitted_total{{{ch_sel}}}[1m]))"), unit="none", decimals=0,
        ds=PROM, h=5, graph="area",
        desc="Speech segments sent for transcription in the last minute."))
    y += 5
    # ---------------------------------------------------------------- live trends
    out.append(row(138, "Live trends", y))
    y += 1
    out.append(ts(139, "Text delay (speech end → live text)", 0, y, 12, 8, [
        prom(f"histogram_quantile(0.5, sum by (le) (rate(stt_latency_speech_end_to_final_seconds_"
             f"bucket{{{ch_sel}}}[1m])))", "A", "median"),
        prom(f"histogram_quantile(0.95, sum by (le) (rate(stt_latency_speech_end_to_final_seconds"
             f"_bucket{{{ch_sel}}}[1m])))", "B", "95 %")], minv=0, steps=DELAY_STEPS,
        desc="How long after speech the live text appears (1-minute windows)."))
    out.append(ts(140, "Model confidence", 12, y, 12, 8, [
        prom(f"sum by (pass) (rate(stt_asr_segment_confidence_sum{{{ch_sel}}}[2m])) / "
             f"sum by (pass) (rate(stt_asr_segment_confidence_count{{{ch_sel}}}[2m]))", "A",
             "{{pass}}")], unit="percentunit", minv=0.5, maxv=1, steps=CONF_STEPS,
        desc="Average model confidence per pass. A drop often means music, crosstalk or "
             "English speech."))
    y += 8
    out.append(ts(141, "Transcription queue (behind live)", 0, y, 12, 7, [
        prom("max by (pass) (stt_asr_queue_lag_seconds)", "A", "{{pass}}")], minv=0,
        desc="Age of the oldest waiting job per pass. FAST should stay low; REFINE only runs "
             "when FAST is idle."))
    out.append(ts(142, "Stream bitrate (audio)", 12, y, 12, 7, [
        prom(f"rate(stt_ingest_bytes_total{{{ch_sel}}}[1m]) * 8", "A", "audio")], unit="bps", minv=0,
        desc="Incoming audio bitrate. A flat line at 0 means no packets arrive."))
    y += 7
    # ---------------------------------------------------------------- accuracy
    out.append(row(143, "Accuracy · measured on reference speech", y))
    y += 1
    out.append(stat(108, "Word accuracy (1 − WER)", 0, y, 6, sql(
        "A", "SELECT 1 - value FROM accuracy_results WHERE kind='bench' AND metric='wer' "
             "AND config LIKE 'large-v3-turbo int8 b1 seg10%' ORDER BY recorded_at DESC LIMIT 1"),
        decimals=1, thresholds=CONF_STEPS,
        desc="1 − WER of the live (FAST) configuration against hand-made reference transcripts "
             "(read speech, FLEURS)."))
    out.append(stat(109, "Sure words that were correct", 6, y, 6, sql(
        "A", "SELECT value FROM accuracy_results WHERE kind='calibration_word' "
             "AND metric='precision' AND threshold BETWEEN 0.899 AND 0.901 "
             "ORDER BY recorded_at DESC LIMIT 1"), decimals=1, thresholds=CONF_STEPS,
        desc="'Sure' = words the model rated ≥ 90% confident. Share of them that matched the "
             "reference (bench/calibration.py)."))
    out.append(stat(110, "Share of words that are 'sure'", 12, y, 6, sql(
        "A", "SELECT value FROM accuracy_results WHERE kind='calibration_word' "
             "AND metric='kept_share' AND threshold BETWEEN 0.899 AND 0.901 "
             "ORDER BY recorded_at DESC LIMIT 1"), decimals=1,
        desc="How many of all words the model rated ≥ 90 % confident (reference set)."))
    out.append({"id": 144, "type": "text", "title": "", "transparent": True,
                "gridPos": {"x": 18, "y": y, "w": 6, "h": 4},
                "options": {"mode": "markdown", "content":
                            "**How to read this:** these numbers were measured once against "
                            "hand-checked transcripts of read Urdu speech. Live TV is harder. "
                            "*Confidence* above is the model's own guess, not accuracy."}})
    y += 4
    # ---------------------------------------------------------------- stored transcript
    out.append(row(145, "Stored transcript · selected time range", y))
    y += 1
    rng = f"$__timeFilter(start_utc) AND {CH_FILTER}"
    W = 4
    out.append(stat(106, "Segments", 0, y, W, sql(
        "A", f"SELECT count(*) FROM asr_segments WHERE {rng}"), unit="none",
        desc="Stored transcript segments (≈ 8–10 s of speech each) in the time range."))
    out.append(stat(104, "Avg confidence", W, y, W, sql(
        "A", f"SELECT avg(confidence) FROM asr_segments WHERE {rng}"),
        decimals=1, thresholds=CONF_STEPS, desc="Average model confidence in the time range."))
    out.append(stat(105, "≥ min confidence", 2 * W, y, W, sql(
        "A", f"SELECT avg(CASE WHEN confidence >= $min_conf THEN 1.0 ELSE 0.0 END) "
             f"FROM asr_segments WHERE {rng}"),
        desc="Share of stored segments at or above the Min confidence variable."))
    out.append(stat(107, "Re-checked (2nd pass)", 3 * W, y, W, sql(
        "A", f"SELECT avg(CASE WHEN pass='refined' THEN 1.0 ELSE 0.0 END) FROM asr_segments "
             f"WHERE {rng}"),
        desc="Share of segments whose careful second pass finished (CPU: only in pauses)."))
    out.append(stat(111, "Re-check skipped", 4 * W, y, W, sql(
        "A", f"SELECT count(*) FROM asr_segments WHERE {rng} "
             "AND EXISTS (SELECT 1 FROM unnest(flags) f WHERE f LIKE 'refine_skipped%')"),
        unit="none", desc="Second pass skipped because the CPU was busy (live text kept)."))
    out.append(stat(122, "Audio skipped", 5 * W, y, W, sql(
        "A", f"SELECT coalesce(sum((details->>'duration_s')::float), 0) FROM stream_events "
             f"WHERE $__timeFilter(at_utc) AND {CH_FILTER} AND type = 'fast_skipped'"),
        unit="s", decimals=0,
        thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 1}],
        desc="Seconds of speech not transcribed because the machine was overloaded "
             "(asr.fast_max_lag_s). 0 = nothing skipped."))
    y += 4
    out.append({
        "id": 112, "type": "table",
        "title": "Transcript — click the text to open that moment with video",
        "description": "Stored transcript segments in the selected time range (top-right picker), "
                       "filtered by Channel and Min confidence (top-left). Click a row's text to "
                       "open the player at that moment.",
        "datasource": PG, "gridPos": {"x": 0, "y": y, "w": 24, "h": 12},
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
                 "url": UI + "/ui/archive?ch=${__data.fields.channel_id}&t=${__data.fields.t}",
                 "targetBlank": True}]}]},
        ]},
        "options": {"showHeader": True, "cellHeight": "md"},
    })
    y += 12
    # Accuracy table goes into the collapsed "Pipeline details" row (built in main()).
    details = []
    details.append({
        "id": 113, "type": "table", "title": "Measured accuracy (reference transcripts)",
        "description": "From result files recorded with `stt record-accuracy`. accuracy = 1 − WER.",
        "datasource": PG, "gridPos": {"x": 0, "y": 0, "w": 24, "h": 10},
        "targets": [sql("A", """
            SELECT dataset, kind, config, metric, threshold, round(value::numeric, 3) AS value, n
            FROM accuracy_results
            WHERE (kind = 'bench' AND metric IN ('wer','cer','rtf'))
               OR (kind LIKE 'calibration%' AND threshold IN (0, 0.8, 0.9, 0.95))
            ORDER BY dataset, kind, config, metric, threshold""")],
        "options": {"showHeader": True, "cellHeight": "sm"},
    })
    out.append({"id": 115, "type": "row", "title": "Pipeline details (click to expand)",
                "collapsed": True, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
                "panels": details})
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
         "query": "SELECT id FROM channels WHERE enabled ORDER BY id", "refresh": 1,
         "multi": False, "includeAll": False, "current": {}},  # one channel per view
        {"name": "ui_theme", "type": "custom", "hide": 2, "query": "auto,light,dark",
         "current": {"text": "auto", "value": "auto"},   # auto = follow the computer's setting
         "options": [{"text": v, "value": v, "selected": v == "auto"} for v in ("auto", "light", "dark")]},
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
    4: "Age of the oldest waiting transcription job. FAST lag growing without limit = overloaded "
       "(with asr.fast_max_lag_s set, older live audio is skipped and counted as 'Audio skipped').",
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
    flat = []
    for p in d["panels"]:
        flat.append(p)
        flat += p.get("panels") or []
    old = sorted((p for p in flat if p.get("id", 0) < 100 and p.get("type") != "row"),
                 key=lambda p: (p["gridPos"]["y"], p["gridPos"]["x"]))
    for p in old:
        if p.get("id") in OLD_DESCRIPTIONS:
            p["description"] = OLD_DESCRIPTIONS[p["id"]]
        if p.get("type") == "timeseries":  # same look as the new charts
            c = p.setdefault("fieldConfig", {}).setdefault("defaults", {}).setdefault("custom", {})
            c.update({"lineWidth": 2, "fillOpacity": 15, "gradientMode": "opacity",
                      "showPoints": "never", "spanNulls": True})
            p.setdefault("options", {})["legend"] = {"displayMode": "list", "placement": "bottom",
                                                     "calcs": ["lastNotNull"]}
    new = panels()
    details_row = next(p for p in new if p["id"] == 115)
    base = min(p["gridPos"]["y"] for p in old) if old else 0
    off = details_row["panels"][0]["gridPos"]["h"] if details_row["panels"] else 0
    for p in old:
        p["gridPos"]["y"] = p["gridPos"]["y"] - base + off + details_row["gridPos"]["y"] + 1
    for p in details_row["panels"]:
        p["gridPos"]["y"] = details_row["gridPos"]["y"] + 1
    details_row["panels"] += old
    d["panels"] = new
    d["templating"] = templating()
    d["title"] = "Urdu Live Transcription · Monitoring"
    d["graphTooltip"] = 1   # shared crosshair across charts
    d["refresh"] = "5s"
    d["time"] = {"from": "now-30m", "to": "now"}
    d["version"] = d.get("version", 1) + 1
    d["timezone"] = "browser"
    DASH.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{DASH}: {len(new)} panels + {len(old)} in Pipeline details")


if __name__ == "__main__":
    main()
