"""Sentence-level timestamps and transcript export (pure functions; used by the API).

A segment (VAD chunk, ≤ max_segment_s) can hold several sentences. Sentences are split on
end-of-sentence punctuation (Urdu ``۔`` ``؟``, Latin ``. ? !``) or, when the model emitted none, on
a pause of at least ``pause_s`` between words. Each sentence gets the UTC start of its first word and
the end of its last word (word timestamps come from the stream PTS clock).
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from live_stt.text.confidence import mean_confidence

END_CHARS = "۔؟?!.…"
PAUSE_S = 1.0
FORMATS = {
    "srt": "application/x-subrip",
    "vtt": "text/vtt",
    "csv": "text/csv",
    "txt": "text/plain",
    "json": "application/json",
}


def _w(x: dict) -> tuple[str, float | None, float | None]:
    w = str(x.get("w", x.get("word", ""))).strip()
    s, e = x.get("s", x.get("start_utc")), x.get("e", x.get("end_utc"))
    return w, (float(s) if s is not None else None), (float(e) if e is not None else None)


def split_sentences(seg: dict, pause_s: float = PAUSE_S) -> list[dict]:
    """Split one transcript row/event into sentences with UTC times and confidence."""
    words = [x for x in (seg.get("words") or []) if isinstance(x, dict)]
    base = {"channel_id": seg.get("channel_id"), "segment_id": seg.get("segment_id"),
            "pass": seg.get("pass") or seg.get("type")}
    if not words:
        text = (seg.get("text_raw") or "").strip()
        if not text:
            return []
        return [{**base, "start_utc": seg.get("start_utc"), "end_utc": seg.get("end_utc"),
                 "text": text, "confidence": None}]
    out, cur = [], []

    def flush():
        if not cur:
            return
        texts = [_w(x)[0] for x in cur]
        starts = [_w(x)[1] for x in cur if _w(x)[1] is not None]
        ends = [_w(x)[2] for x in cur if _w(x)[2] is not None]
        text = " ".join(t for t in texts if t)
        if text:
            out.append({**base,
                        "start_utc": min(starts) if starts else seg.get("start_utc"),
                        "end_utc": max(ends) if ends else seg.get("end_utc"),
                        "text": text, "confidence": mean_confidence(cur)})
        cur.clear()

    for i, x in enumerate(words):
        cur.append(x)
        w, _, e = _w(x)
        nxt_s = _w(words[i + 1])[1] if i + 1 < len(words) else None
        if (w and w[-1] in END_CHARS) or (
                e is not None and nxt_s is not None and nxt_s - e >= pause_s):
            flush()
    flush()
    return out


MAX_SENTENCE_S = 20.0


def assemble(rows: list[dict], pause_s: float = PAUSE_S,
             max_s: float = MAX_SENTENCE_S) -> list[dict]:
    """Sentences across segment boundaries (rows of ONE channel).

    VAD segments are force-cut at ``max_segment_s`` inside continuous speech (measured on live Urdu TV news:
    67 % of segment boundaries had < 0.35 s between them), and Whisper rarely punctuates broadcast
    Urdu. So words of consecutive segments are joined, and a sentence ends at punctuation, at a
    pause ≥ ``pause_s`` between words, or — if it grows past ``max_s`` — at its longest internal
    pause (the most likely sentence break).
    """
    words: list[tuple[dict, dict]] = []  # (word, row)
    for r in sorted(rows, key=lambda r: r.get("start_utc") or 0.0):
        ws = [x for x in (r.get("words") or []) if isinstance(x, dict) and _w(x)[0]]
        if not ws and (r.get("text_raw") or "").strip():  # no word timings: whole row = one word
            ws = [{"w": r["text_raw"].strip(), "s": r.get("start_utc"), "e": r.get("end_utc")}]
        words += [(x, r) for x in ws]
    out: list[dict] = []

    def emit(chunk: list[tuple[dict, dict]]) -> None:
        if not chunk:
            return
        ws = [x for x, _ in chunk]
        rs = list({id(r): r for _, r in chunk}.values())
        starts = [_w(x)[1] for x in ws if _w(x)[1] is not None]
        ends = [_w(x)[2] for x in ws if _w(x)[2] is not None]
        passes = {("refined" if (r.get("pass") or r.get("type")) == "refined" else "fast")
                  for r in rs}
        lats = [r.get("latency_s", (r.get("timings") or {}).get("latency_s")) for r in rs]
        lats = [float(v) for v in lats if isinstance(v, (int, float))]
        out.append({
            "channel_id": rs[0].get("channel_id"),
            "segment_id": rs[0].get("segment_id"),
            "segment_ids": [r.get("segment_id") for r in rs],
            "pass": passes.pop() if len(passes) == 1 else "mixed",
            "start_utc": min(starts) if starts else rs[0].get("start_utc"),
            "end_utc": max(ends) if ends else rs[-1].get("end_utc"),
            "text": " ".join(_w(x)[0] for x in ws),
            "confidence": mean_confidence(ws),
            "latency_s": max(lats) if lats else None,
        })

    def gap(a: dict, b: dict) -> float:
        e, s = _w(a)[2], _w(b)[1]
        return (s - e) if (e is not None and s is not None) else 0.0

    cur: list[tuple[dict, dict]] = []
    for i, (x, r) in enumerate(words):
        cur.append((x, r))
        t = _w(x)[0]
        nxt = words[i + 1][0] if i + 1 < len(words) else None
        if t[-1] in END_CHARS or (nxt is not None and gap(x, nxt) >= pause_s):
            emit(cur)
            cur = []
            continue
        s0, e1 = _w(cur[0][0])[1], _w(x)[2]
        while len(cur) > 1 and s0 is not None and e1 is not None and e1 - s0 > max_s:
            k = max(range(1, len(cur)), key=lambda j: (gap(cur[j - 1][0], cur[j][0]), j))
            emit(cur[:k])
            cur = cur[k:]
            s0 = _w(cur[0][0])[1]
    emit(cur)
    return out


def keep(sent: dict, min_conf: float) -> bool:
    if min_conf <= 0:
        return True
    c = sent.get("confidence")
    return c is not None and c >= min_conf


def get_tz(name: str | None):
    if not name or name.upper() == "UTC":
        return timezone.utc
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise ValueError(f"unknown time zone {name!r}") from e


def _abs(t: float, tz) -> str:
    return datetime.fromtimestamp(t, tz).isoformat(timespec="milliseconds")


def _rel(t: float, origin: float, sep: str) -> str:
    ms = max(0, round((t - origin) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def render(sents: list[dict], fmt: str, tz=timezone.utc, origin: float | None = None) -> str:
    """Render sentences. SRT/VTT cue times are relative to ``origin`` (default: first sentence);
    the absolute UTC start is given in a header/NOTE. CSV/TXT/JSON use absolute times in ``tz``."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {sorted(FORMATS)}")
    sents = [x for x in sents if x.get("start_utc") is not None and x.get("end_utc") is not None]
    if origin is None:
        origin = sents[0]["start_utc"] if sents else 0.0
    if fmt == "json":
        return json.dumps([{**x, "start": _abs(x["start_utc"], tz), "end": _abs(x["end_utc"], tz)}
                           for x in sents], ensure_ascii=False, indent=1)
    if fmt == "txt":
        return "".join(f"[{_abs(x['start_utc'], tz)} → {_abs(x['end_utc'], tz)}] "
                       f"{x['text']}\n" for x in sents)
    if fmt == "csv":
        buf = io.StringIO()
        buf.write("﻿")  # BOM so Excel opens Urdu UTF-8 correctly
        wr = csv.writer(buf)
        wr.writerow(["channel_id", "segment_id", "pass", "start", "end", "start_utc_epoch",
                     "end_utc_epoch", "confidence", "text"])
        for x in sents:
            c = x.get("confidence")
            wr.writerow([x.get("channel_id"), x.get("segment_id"), x.get("pass"),
                         _abs(x["start_utc"], tz), _abs(x["end_utc"], tz),
                         f"{x['start_utc']:.3f}", f"{x['end_utc']:.3f}",
                         "" if c is None else f"{c:.3f}", x["text"]])
        return buf.getvalue()
    sep = "," if fmt == "srt" else "."
    lines = ["WEBVTT", "", f"NOTE cue times relative to {_abs(origin, tz)}", ""] \
        if fmt == "vtt" else []
    for i, x in enumerate(sents, 1):
        end = max(x["end_utc"], x["start_utc"] + 0.001)
        if fmt == "srt":
            lines.append(str(i))
        lines += [f"{_rel(x['start_utc'], origin, sep)} --> {_rel(end, origin, sep)}",
                  x["text"], ""]
    return "\n".join(lines)
