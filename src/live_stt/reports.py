"""Topic search helpers and transcript reports (custom time range or overall).

A report summarises what the system transcribed for one or more channels (or an uploaded file):
speech time, segments, words, model confidence, refine coverage, latency, guard flags, stream
events, hourly activity, most frequent words, topic mentions, measured offline accuracy, the
video archive available, and optionally the full sentence-level transcript.
Formats: html (self-contained, printable → PDF), md, json.
"""

from __future__ import annotations

import html as _html
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from live_stt.text.sentences import assemble
from live_stt.text.urdu_normalize import normalize

FORMATS = {"html": "text/html", "md": "text/markdown", "json": "application/json"}

# Function words left out of "most frequent words" (normalised forms).
STOPWORDS = sorted({normalize(w) for w in """
کے کی کا کو سے میں نے پر اور ہے ہیں تھا تھی تھے ہو ہوں ہوا ہوئی ہوئے یہ وہ اس ان ایک بھی
جو کہ کر کیا کیے کرنے کرتے کرے گا گی گے لیے لئے تو نہیں ہی اب جب تک تاکہ مگر لیکن یا بہت
کچھ کوئی سب اپنے اپنی اپنا آپ ہم میں جس جن جا جائے جاتا جاتی رہا رہی رہے دیا دی دیے والے والی
والا ساتھ بعد پہلے طرف بات ہونے ہوتا ہوتی سکتا سکتی سکتے چاہیے یہاں وہاں کیونکہ پھر ابھی جی
the a an of to and in is are for on that this with it be
""".split() if normalize(w)})


# --------------------------------------------------------------------------- topics / search
def load_topics(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out = []
    for t in data.get("topics") or []:
        name = str(t.get("name", "")).strip()
        terms = [str(x).strip() for x in (t.get("terms") or []) if str(x).strip()]
        if name and terms:
            out.append({"name": name, "terms": terms})
    return out


def split_terms(q: str | None) -> list[str]:
    """``a | b , c`` → [a, b, c] (any separator of | , ؛ ،)."""
    if not q:
        return []
    for sep in ("|", "،", "؛", ";"):
        q = q.replace(sep, ",")
    return [t.strip() for t in q.split(",") if t.strip()]


def like_patterns(terms: list[str]) -> list[str]:
    pats = []
    for t in terms:
        n = normalize(t)
        if n:
            pats.append("%" + n.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                        + "%")
    return sorted(set(pats))


def cluster_hits(hits: list[dict], gap_s: float = 180.0) -> list[dict]:
    """Group hits of the same channel that are < ``gap_s`` apart: one cluster ≈ one news item /
    speech in which the topic keeps coming up. Newest cluster first."""
    by_ch: dict[str, list[dict]] = {}
    for h in sorted(hits, key=lambda h: h["start_utc"]):
        by_ch.setdefault(h["channel_id"], []).append(h)
    out = []
    for ch, hs in by_ch.items():
        cur: list[dict] = []
        for h in hs:
            if cur and h["start_utc"] - cur[-1]["end_utc"] > gap_s:
                out.append(_cluster(ch, cur))
                cur = []
            cur.append(h)
        if cur:
            out.append(_cluster(ch, cur))
    out.sort(key=lambda c: c["start_utc"], reverse=True)
    return out


def _cluster(ch: str, hs: list[dict]) -> dict:
    return {"channel_id": ch, "start_utc": hs[0]["start_utc"],
            "end_utc": max(h["end_utc"] for h in hs), "hits": len(hs),
            "preview": (hs[0].get("text_raw") or "")[:160]}


# --------------------------------------------------------------------------- report data
WHERE = """channel_id = ANY($1::text[])
           AND ($2::timestamptz IS NULL OR start_utc >= $2)
           AND ($3::timestamptz IS NULL OR start_utc < $3)"""


def _ts(d) -> float | None:
    return d.timestamp() if isinstance(d, datetime) else d


async def build_report(conn, channels: dict[str, str], t0: datetime | None, t1: datetime | None,
                       tz_name: str, topics: list[dict], include_transcript: bool = False,
                       max_rows: int = 50_000, archive: dict | None = None,
                       accuracy: list[dict] | None = None) -> dict:
    """``channels``: id → display name. ``conn``: asyncpg connection/pool."""
    ids = list(channels)
    args = (ids, t0, t1)
    per = await conn.fetch(f"""
        SELECT channel_id, count(*) AS segments,
               count(*) FILTER (WHERE pass = 'refined') AS refined,
               coalesce(sum(extract(epoch FROM end_utc - start_utc)), 0) AS speech_s,
               coalesce(sum(CASE WHEN jsonb_typeof(words) = 'array'
                                 THEN jsonb_array_length(words) END), 0) AS words,
               avg(confidence) AS confidence,
               avg((confidence >= 0.9)::int) FILTER (WHERE confidence IS NOT NULL) AS share_hi,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_s)
                   FILTER (WHERE pass = 'fast') AS lat_fast_p50,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_s)
                   FILTER (WHERE pass = 'fast') AS lat_fast_p95,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_s)
                   FILTER (WHERE pass = 'refined') AS lat_ref_p50,
               min(start_utc) AS first_utc, max(end_utc) AS last_utc
        FROM asr_segments WHERE {WHERE} GROUP BY channel_id ORDER BY channel_id""", *args)
    rows_per = {r["channel_id"]: dict(r) for r in per}
    firsts = [r["first_utc"] for r in per if r["first_utc"]]
    lasts = [r["last_utc"] for r in per if r["last_utc"]]
    period = (_ts(t0) if t0 else (_ts(min(firsts)) if firsts else None),
              _ts(t1) if t1 else (_ts(max(lasts)) if lasts else None))

    hourly = await conn.fetch(f"""
        SELECT date_trunc('hour', start_utc AT TIME ZONE $4) AS h, count(*) AS segments,
               coalesce(sum(extract(epoch FROM end_utc - start_utc)), 0) AS speech_s,
               coalesce(sum(CASE WHEN jsonb_typeof(words) = 'array'
                                 THEN jsonb_array_length(words) END), 0) AS words,
               avg(confidence) AS confidence
        FROM asr_segments WHERE {WHERE} GROUP BY 1 ORDER BY 1""", *args, tz_name)
    flags = await conn.fetch(f"""
        SELECT f AS flag, count(*) AS n FROM asr_segments, unnest(flags) f
        WHERE {WHERE} GROUP BY f ORDER BY n DESC LIMIT 20""", *args)
    words = await conn.fetch(f"""
        SELECT w AS word, count(*) AS n
        FROM asr_segments, regexp_split_to_table(text_norm, '\\s+') w
        WHERE {WHERE} AND char_length(w) > 1 AND NOT (w = ANY($4::text[]))
        GROUP BY w ORDER BY n DESC, w LIMIT 30""", *args, STOPWORDS)
    ev_where = WHERE.replace("start_utc", "at_utc")
    events = await conn.fetch(f"""
        SELECT channel_id, type, count(*) AS n FROM stream_events WHERE {ev_where}
        GROUP BY channel_id, type ORDER BY channel_id, n DESC""", *args)
    downs = await conn.fetch(f"""
        SELECT channel_id, type, at_utc, details FROM stream_events
        WHERE {ev_where} AND type IN ('stream_down', 'stream_up', 'discontinuity')
        ORDER BY at_utc DESC LIMIT 50""", *args)
    topic_rows = []
    for t in topics:
        pats = like_patterns(t["terms"])
        if not pats:
            continue
        r = await conn.fetchrow(f"""
            SELECT count(*) AS n, min(start_utc) AS first_utc, max(start_utc) AS last_utc,
                   count(DISTINCT channel_id) AS channels
            FROM asr_segments WHERE {WHERE} AND text_norm ILIKE ANY($4::text[])""", *args, pats)
        topic_rows.append({"name": t["name"], "terms": t["terms"], "mentions": r["n"],
                           "first_utc": _ts(r["first_utc"]), "last_utc": _ts(r["last_utc"]),
                           "channels": r["channels"]})

    chans = []
    for cid in ids:
        r = rows_per.get(cid, {})
        chans.append({
            "channel_id": cid, "name": channels[cid],
            "segments": int(r.get("segments") or 0), "refined": int(r.get("refined") or 0),
            "speech_s": float(r.get("speech_s") or 0), "words": int(r.get("words") or 0),
            "confidence": _f(r.get("confidence")), "share_hi_conf": _f(r.get("share_hi")),
            "latency_fast_p50_s": _f(r.get("lat_fast_p50")),
            "latency_fast_p95_s": _f(r.get("lat_fast_p95")),
            "latency_refined_p50_s": _f(r.get("lat_ref_p50")),
            "first_utc": _ts(r.get("first_utc")), "last_utc": _ts(r.get("last_utc")),
            "video_archive": (archive or {}).get(cid),
        })
    tot_seg = sum(c["segments"] for c in chans)
    conf_w = [(c["confidence"], c["segments"]) for c in chans if c["confidence"] is not None]
    summary = {
        "channels": len(ids), "segments": tot_seg,
        "refined": sum(c["refined"] for c in chans),
        "speech_s": sum(c["speech_s"] for c in chans), "words": sum(c["words"] for c in chans),
        "confidence": (sum(c * n for c, n in conf_w) / sum(n for _, n in conf_w)) if conf_w else None,
    }
    out = {
        "title": "Transcription report", "generated_utc": time.time(), "tz": tz_name,
        "period": {"from_utc": period[0], "to_utc": period[1],
                   "overall": t0 is None and t1 is None},
        "summary": summary, "channels": chans,
        "hourly": [{"hour_local": r["h"].isoformat(timespec="minutes"),
                    "segments": r["segments"], "speech_s": float(r["speech_s"]),
                    "words": int(r["words"]), "confidence": _f(r["confidence"])} for r in hourly],
        "flags": [dict(r) for r in flags], "top_words": [dict(r) for r in words],
        "stream_events": [dict(r) for r in events],
        "recent_stream_events": [{"channel_id": r["channel_id"], "type": r["type"],
                                  "at_utc": _ts(r["at_utc"]),
                                  "reason": (_details(r["details"]).get("reason") or "")[:120]}
                                 for r in downs],
        "topics": topic_rows, "accuracy": accuracy or [],
        "notes": [
            "Confidence = the model's own mean word probability, not measured accuracy.",
            "Measured accuracy (if listed) comes from offline benchmarks against human reference "
            "transcripts, not from this period's audio.",
            "Latency = end of speech → text published; rows later refined keep only the refined "
            "latency, so fast-pass percentiles cover fast rows that were not (yet) refined.",
        ],
    }
    if include_transcript:
        tr = {}
        for cid in ids:
            rows = await conn.fetch(f"""
                SELECT channel_id, segment_id, start_utc, end_utc, pass, text_raw, words,
                       confidence, latency_s
                FROM asr_segments WHERE channel_id = $1
                  AND ($2::timestamptz IS NULL OR start_utc >= $2)
                  AND ($3::timestamptz IS NULL OR start_utc < $3)
                ORDER BY start_utc LIMIT $4""", cid, t0, t1, max_rows)
            rs = []
            for r in rows:
                d = dict(r)
                d["start_utc"], d["end_utc"] = _ts(d["start_utc"]), _ts(d["end_utc"])
                if isinstance(d.get("words"), str):
                    d["words"] = json.loads(d["words"])
                rs.append(d)
            tr[cid] = [{"start_utc": s["start_utc"], "end_utc": s["end_utc"], "text": s["text"],
                        "confidence": s["confidence"], "pass": s["pass"]} for s in assemble(rs)]
            if len(rows) >= max_rows:
                tr[cid].append({"start_utc": None, "end_utc": None, "confidence": None,
                                "pass": "", "text": f"[truncated at {max_rows} segments]"})
        out["transcript"] = tr
    return out


def _f(x) -> float | None:
    return None if x is None else round(float(x), 4)


def _details(d) -> dict:
    if isinstance(d, str):
        try:
            return json.loads(d)
        except ValueError:
            return {}
    return d or {}


# --------------------------------------------------------------------------- rendering
def _t(t: float | None, tz) -> str:
    return "–" if t is None else datetime.fromtimestamp(t, tz).strftime("%Y-%m-%d %H:%M:%S")


def _dur(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}h {s % 3600 // 60:02d}m {s % 60:02d}s"


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x * 100:.1f}%"


def _sec(x: float | None) -> str:
    return "–" if x is None else f"{x:.1f} s"


def render_report(data: dict, fmt: str, tz=timezone.utc) -> str:
    if fmt == "json":
        return json.dumps(data, ensure_ascii=False, indent=1, default=str)
    if fmt == "md":
        return _md(data, tz)
    if fmt == "html":
        return _html_report(data, tz)
    raise ValueError(f"format must be one of {sorted(FORMATS)}")


def _period(data: dict, tz) -> str:
    p = data["period"]
    s = f"{_t(p['from_utc'], tz)} → {_t(p['to_utc'], tz)} ({data['tz']})"
    return s + (" · overall (all stored data)" if p["overall"] else "")


def _md(d: dict, tz) -> str:
    s = d["summary"]
    L = [f"# {d['title']}", "", f"**Period:** {_period(d, tz)}  ",
         f"**Generated:** {_t(d['generated_utc'], tz)}", "", "## Summary", "",
         "| Channels | Speech | Segments | Refined | Words | Avg confidence |", "|---|---|---|---|---|---|",
         f"| {s['channels']} | {_dur(s['speech_s'])} | {s['segments']} | {s['refined']} | "
         f"{s['words']} | {_pct(s['confidence'])} |", "", "## Channels", "",
         "| Channel | Speech | Segments | Words | Confidence | ≥ 90 % conf | Latency fast p50 / p95 | First → last |",
         "|---|---|---|---|---|---|---|---|"]
    for c in d["channels"]:
        L.append(f"| {c['name'] or c['channel_id']} ({c['channel_id']}) | {_dur(c['speech_s'])} | "
                 f"{c['segments']} | {c['words']} | {_pct(c['confidence'])} | "
                 f"{_pct(c['share_hi_conf'])} | {_sec(c['latency_fast_p50_s'])} / "
                 f"{_sec(c['latency_fast_p95_s'])} | {_t(c['first_utc'], tz)} → {_t(c['last_utc'], tz)} |")
    if d["topics"]:
        L += ["", "## Topics", "", "| Topic | Mentions | First | Last |", "|---|---|---|---|"]
        L += [f"| {t['name']} | {t['mentions']} | {_t(t['first_utc'], tz)} | {_t(t['last_utc'], tz)} |"
              for t in d["topics"]]
    if d["hourly"]:
        L += ["", "## Hourly activity", "", "| Hour | Speech | Words | Confidence |", "|---|---|---|---|"]
        L += [f"| {h['hour_local']} | {_dur(h['speech_s'])} | {h['words']} | {_pct(h['confidence'])} |"
              for h in d["hourly"]]
    if d["top_words"]:
        L += ["", "## Most frequent words", "", ", ".join(f"{w['word']} ({w['n']})" for w in d["top_words"])]
    if d["stream_events"]:
        L += ["", "## Stream events", "", "| Channel | Event | Count |", "|---|---|---|"]
        L += [f"| {e['channel_id']} | {e['type']} | {e['n']} |" for e in d["stream_events"]]
    if d["flags"]:
        L += ["", "## Guard flags", "", ", ".join(f"{f['flag']} ({f['n']})" for f in d["flags"])]
    if d["accuracy"]:
        L += ["", "## Measured accuracy (offline benchmark)", "", "| Dataset | Config | Metric | Value |", "|---|---|---|---|"]
        L += [f"| {a['dataset']} | {a['config']} | {a['metric']} | {a['value']:.4f} |" for a in d["accuracy"]]
    L += ["", "## Notes", ""] + [f"* {n}" for n in d["notes"]]
    for cid, sents in (d.get("transcript") or {}).items():
        L += ["", f"## Transcript — {cid}", ""]
        L += [f"* `{_t(x['start_utc'], tz)}` {x['text']}" for x in sents]
    return "\n".join(L) + "\n"


def _e(x) -> str:
    return _html.escape(str(x))


def _html_report(d: dict, tz) -> str:
    s = d["summary"]
    maxw = max([h["words"] for h in d["hourly"]] or [1]) or 1
    rows_ch = "".join(
        f"<tr><td>{_e(c['name'] or c['channel_id'])} <small>{_e(c['channel_id'])}</small></td>"
        f"<td>{_dur(c['speech_s'])}</td><td>{c['segments']}</td><td>{c['refined']}</td>"
        f"<td>{c['words']}</td><td>{_pct(c['confidence'])}</td><td>{_pct(c['share_hi_conf'])}</td>"
        f"<td>{_sec(c['latency_fast_p50_s'])} / {_sec(c['latency_fast_p95_s'])}</td>"
        f"<td>{_t(c['first_utc'], tz)}<br>{_t(c['last_utc'], tz)}</td>"
        f"<td>{_e(_arch(c.get('video_archive'), tz))}</td></tr>" for c in d["channels"])
    rows_h = "".join(
        f"<tr><td>{_e(h['hour_local'].replace('T', ' '))}</td><td>{_dur(h['speech_s'])}</td>"
        f"<td class=bar><span style='width:{100 * h['words'] / maxw:.1f}%'></span>{h['words']}</td>"
        f"<td>{_pct(h['confidence'])}</td></tr>" for h in d["hourly"])
    rows_t = "".join(
        f"<tr><td>{_e(t['name'])}</td><td class=u>{_e(' · '.join(t['terms']))}</td>"
        f"<td>{t['mentions']}</td><td>{_t(t['first_utc'], tz)}</td><td>{_t(t['last_utc'], tz)}</td></tr>"
        for t in d["topics"])
    words = " ".join(f"<span class=chip><span class=u>{_e(w['word'])}</span> {w['n']}</span>"
                     for w in d["top_words"])
    rows_ev = "".join(f"<tr><td>{_e(e['channel_id'])}</td><td>{_e(e['type'])}</td><td>{e['n']}</td></tr>"
                      for e in d["stream_events"])
    rows_rev = "".join(f"<tr><td>{_t(e['at_utc'], tz)}</td><td>{_e(e['channel_id'])}</td>"
                       f"<td>{_e(e['type'])}</td><td>{_e(e['reason'])}</td></tr>"
                       for e in d["recent_stream_events"])
    flags = " ".join(f"<span class=chip>{_e(f['flag'])} {f['n']}</span>" for f in d["flags"])
    acc = "".join(f"<tr><td>{_e(a['dataset'])}</td><td>{_e(a['config'])}</td><td>{_e(a['metric'])}</td>"
                  f"<td>{a['value']:.4f}</td><td>{_e(a.get('n') or '')}</td></tr>" for a in d["accuracy"])
    trans = ""
    for cid, sents in (d.get("transcript") or {}).items():
        trans += f"<h2>Transcript — {_e(cid)} <small>({len(sents)} sentences)</small></h2><table class=tr>"
        trans += "".join(f"<tr><td class=ts>{_t(x['start_utc'], tz)}</td><td class=u dir=auto>{_e(x['text'])}</td>"
                         f"<td class=ts>{_pct(x['confidence'])}</td></tr>" for x in sents)
        trans += "</table>"

    def sec(title: str, body: str, empty: str = "No data in this period.") -> str:
        return f"<h2>{title}</h2>" + (body if body.strip() else f"<p class=muted>{empty}</p>")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Transcription report</title>
<style>
:root{{--ink:#14171c;--muted:#6b7280;--line:#e3e6eb;--bg:#fff;--acc:#1f6feb;--bar:#c9dbfb}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--ink:#e6edf3;--muted:#8b949e;--line:#30363d;--bg:#0e1116;--acc:#4493f8;--bar:#1b2a44}}}}
body{{margin:0 auto;max-width:1100px;padding:16px;font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--bg)}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:24px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} td,th{{border-bottom:1px solid var(--line);padding:4px 6px;text-align:start;vertical-align:top}}
th{{font-size:12px;color:var(--muted);font-weight:600}} small,.muted{{color:var(--muted)}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin:12px 0}}
.kpis div{{border:1px solid var(--line);border-radius:8px;padding:8px 10px}} .kpis b{{display:block;font-size:20px}}
.kpis span{{font-size:12px;color:var(--muted)}} .u{{font-family:"Noto Nastaliq Urdu",serif;unicode-bidi:plaintext}}
.chip{{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 8px;margin:2px}}
.bar{{position:relative}} .bar span{{position:absolute;inset:3px auto 3px 0;background:var(--bar);z-index:-1;border-radius:3px}}
.tr td.u{{font-size:17px;line-height:2}} .ts{{white-space:nowrap;font-size:12px;color:var(--muted)}}
.wrap{{overflow-x:auto}} button{{font:inherit;padding:4px 10px}} @media print{{button{{display:none}}}}
</style></head><body>
<button onclick="print()" style="float:right">Print / save as PDF</button>
<h1>{_e(d['title'])}</h1>
<div class=muted>{_e(_period(d, tz))} · generated {_t(d['generated_utc'], tz)}</div>
<div class=kpis>
 <div><b>{s['channels']}</b><span>channels</span></div>
 <div><b>{_dur(s['speech_s'])}</b><span>speech transcribed</span></div>
 <div><b>{s['segments']}</b><span>segments ({s['refined']} refined)</span></div>
 <div><b>{s['words']}</b><span>words</span></div>
 <div><b>{_pct(s['confidence'])}</b><span>avg model confidence</span></div>
</div>
<h2>Channels</h2><div class=wrap><table><tr><th>Channel</th><th>Speech</th><th>Segments</th><th>Refined</th><th>Words</th>
<th>Confidence</th><th>≥ 90 % conf</th><th>Latency fast p50 / p95</th><th>First / last</th><th>Video archive</th></tr>{rows_ch}</table></div>
{sec("Topics", f"<div class=wrap><table><tr><th>Topic</th><th>Terms</th><th>Mentions</th><th>First</th><th>Last</th></tr>{rows_t}</table></div>" if rows_t else "", "No topics configured.")}
{sec("Hourly activity (" + _e(d['tz']) + ")", f"<div class=wrap><table><tr><th>Hour</th><th>Speech</th><th>Words</th><th>Confidence</th></tr>{rows_h}</table></div>" if rows_h else "")}
{sec("Most frequent words", f"<p>{words}</p>" if words else "")}
{sec("Stream events", f"<div class=wrap><table><tr><th>Channel</th><th>Event</th><th>Count</th></tr>{rows_ev}</table></div>" if rows_ev else "", "No stream events (no drops or discontinuities).")}
{f"<h2>Recent outages / discontinuities</h2><div class=wrap><table><tr><th>Time</th><th>Channel</th><th>Event</th><th>Reason</th></tr>{rows_rev}</table></div>" if rows_rev else ""}
{sec("Guard flags", f"<p>{flags}</p>" if flags else "", "No guard flags.")}
{sec("Measured accuracy (offline benchmark)", f"<div class=wrap><table><tr><th>Dataset</th><th>Config</th><th>Metric</th><th>Value</th><th>n</th></tr>{acc}</table></div>" if acc else "", "No accuracy results recorded (stt record-accuracy).")}
<h2>Notes</h2><ul>{"".join(f"<li>{_e(n)}</li>" for n in d['notes'])}</ul>
{trans}
</body></html>"""


def _arch(a: dict | None, tz) -> str:
    if not a:
        return "–"
    return f"{a['hours']:.1f} h in {a['spans']} span(s)"
