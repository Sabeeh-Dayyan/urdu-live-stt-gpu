"""Load measured accuracy (``stt bench`` results, ``bench.calibration`` output) into PostgreSQL.

Only measured numbers from result files are stored; the dashboard reads them from
``accuracy_results``. Re-recording the same file replaces its rows (idempotent upsert).
"""

from __future__ import annotations

import json
from pathlib import Path

import asyncpg

Row = tuple  # (source_file, dataset, kind, config, metric, threshold, value, n)

UPSERT = """
INSERT INTO accuracy_results (source_file, dataset, kind, config, metric, threshold, value, n)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
ON CONFLICT (source_file, dataset, kind, config, metric, threshold)
DO UPDATE SET value = EXCLUDED.value, n = EXCLUDED.n, recorded_at = now()
"""


def _ds_name(ds) -> str:
    return ds.get("name", "?") if isinstance(ds, dict) else str(ds or "?")


def rows_from_result(source: str, d: dict) -> list[Row]:
    """Map a result JSON (bench or calibration) to accuracy rows. Unknown shapes → []."""
    rows: list[Row] = []
    if "runs" in d:  # stt bench
        ds = _ds_name(d.get("dataset"))
        for r in d["runs"]:
            for metric in ("wer", "cer", "wer_noguard", "rtf", "halluc_rate"):
                v = r.get(metric)
                if isinstance(v, (int, float)):
                    rows.append((source, ds, "bench", r.get("label", "?"), metric, None,
                                 float(v), r.get("ref_words")))
    elif "word_level" in d and "segment_level" in d:  # bench.calibration
        ds = _ds_name(d.get("dataset"))
        cfg = f"{d.get('model')} {d.get('compute_type', '')} b{d.get('beam_size')}".strip()
        for r in d["word_level"]:
            for metric in ("precision", "precision_below", "kept_share"):
                if r.get(metric) is not None:
                    rows.append((source, ds, "calibration_word", cfg, metric,
                                 float(r["threshold"]), float(r[metric]), r.get("words_kept")))
        for r in d["segment_level"]:
            for metric in ("accuracy", "wer", "kept_share"):
                if r.get(metric) is not None:
                    rows.append((source, ds, "calibration_segment", cfg, metric,
                                 float(r["threshold"]), float(r[metric]), r.get("segments_kept")))
    return rows


async def record(dsn: str, paths: list[str]) -> int:
    rows: list[Row] = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        got = rows_from_result(Path(p).name, d)
        if not got:
            raise ValueError(f"{p}: not a bench or calibration result")
        rows += got
    conn = await asyncpg.connect(dsn)
    try:
        await conn.executemany(UPSERT, rows)
    finally:
        await conn.close()
    return len(rows)
