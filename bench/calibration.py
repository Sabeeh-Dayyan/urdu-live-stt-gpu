"""Confidence calibration: what does "confidence ≥ X" mean in measured accuracy?

Transcribes every utterance of a reference set (hand-checked text) with the FAST decoder, then
reports, for confidence thresholds, how much is kept and how accurate the kept part is:

* word level   — a hypothesis word counts as correct if the WER alignment marks it "equal";
                 precision = correct kept words / kept words.
* segment level — utterance confidence = mean word probability (live_stt.text.confidence);
                 accuracy of kept utterances = 1 − pooled WER over them.

Usage:  python -m bench.calibration --dataset testbed/reference/fleurs_ur.json \
            --config config/dev-mac-cpu.yaml --out bench/out/calibration_fleurs.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jiwer

from live_stt.text.confidence import mean_confidence
from live_stt.text.urdu_normalize import normalize

THRESHOLDS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95)


def hyp_tokens(words: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Normalise each hypothesis word; a word may normalise to 0..n tokens (each keeps its p)."""
    out = []
    for w, p in words:
        out += [(t, p) for t in normalize(w).split()]
    return out


def mark_correct(ref_text: str, toks: list[tuple[str, float]]) -> list[bool]:
    """Per hypothesis token: True if the WER alignment says it matches the reference."""
    ref = normalize(ref_text)
    if not toks:
        return []
    if not ref.split():
        return [False] * len(toks)
    out = jiwer.process_words(ref, " ".join(t for t, _ in toks))
    ok = [False] * len(toks)
    for ch in out.alignments[0]:
        if ch.type == "equal":
            for j in range(ch.hyp_start_idx, ch.hyp_end_idx):
                ok[j] = True
    return ok


def word_table(items: list[dict], thresholds=THRESHOLDS) -> list[dict]:
    allw = [(p, c) for it in items for p, c in zip(it["p"], it["correct"])]
    rows = []
    for t in thresholds:
        kept = [c for p, c in allw if p >= t]
        below = [c for p, c in allw if p < t]
        rows.append({"threshold": t, "words_total": len(allw), "words_kept": len(kept),
                     "kept_share": len(kept) / len(allw) if allw else None,
                     "precision": sum(kept) / len(kept) if kept else None,
                     "precision_below": sum(below) / len(below) if below else None})
    return rows


def segment_table(items: list[dict], thresholds=THRESHOLDS) -> list[dict]:
    rows = []
    for t in thresholds:
        kept = [it for it in items if it["conf"] is not None and it["conf"] >= t]
        refs = " ".join(normalize(it["ref"]) for it in kept).strip()
        hyps = " ".join(normalize(it["hyp"]) for it in kept).strip()
        wer = jiwer.wer(refs, hyps) if refs and hyps else (1.0 if refs else None)
        rows.append({"threshold": t, "segments_total": len(items), "segments_kept": len(kept),
                     "kept_share": len(kept) / len(items) if items else None,
                     "wer": wer, "accuracy": None if wer is None else 1.0 - wer})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="testbed/reference/fleurs_ur.json")
    ap.add_argument("--config", default="config/dev-mac-cpu.yaml")
    ap.add_argument("--out", default="bench/out/calibration_fleurs.json")
    ap.add_argument("--recompute", action="store_true",
                    help="only recompute the tables from the items already in --out")
    a = ap.parse_args()
    if a.recompute:
        d = json.loads(Path(a.out).read_text(encoding="utf-8"))
        d["word_level"], d["segment_level"] = word_table(d["items"]), segment_table(d["items"])
        Path(a.out).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        for r in d["word_level"] + d["segment_level"]:
            print(r)
        return

    from bench.audio import load_audio
    from live_stt.asr.backends.base import AudioJob, DecodeOptions
    from live_stt.asr.backends.faster_whisper import FasterWhisperBackend
    from live_stt.settings import load_settings, resolve_path

    s = load_settings(a.config)
    ds = json.loads(Path(a.dataset).read_text(encoding="utf-8"))
    audio = load_audio(resolve_path(ds["audio"]))
    sr = ds.get("sample_rate", 16000)
    fast = s.asr.fast
    be = FasterWhisperBackend(s.asr.model_copy(update={"pack_s": 0, "batched": False}))
    be.load(fast.model)
    be.warmup(fast.model)
    opts = DecodeOptions(model=fast.model, beam_size=fast.beam_size, best_of=fast.best_of)
    items, t0 = [], time.perf_counter()
    for seg in ds["segments"]:
        clip = audio[int(seg["start"] * sr):int(seg["end"] * sr)]
        res = be.transcribe([AudioJob(seg["id"], clip, language="ur")], opts)[0]
        words = [(w.word, w.probability) for sg in res.segments for w in sg.words]
        toks = hyp_tokens(words)
        items.append({"id": seg["id"], "ref": seg["text"], "hyp": res.text,
                      "conf": mean_confidence([{"p": p} for _, p in words]),
                      "p": [p for _, p in toks], "correct": mark_correct(seg["text"], toks)})
        print(f"{seg['id']}: conf={items[-1]['conf']}", flush=True)
    out = {"dataset": ds["name"], "source": ds.get("source"), "model": fast.model,
           "beam_size": fast.beam_size, "compute_type": s.asr.compute_type,
           "segments": len(items), "elapsed_s": round(time.perf_counter() - t0, 1),
           "word_level": word_table(items), "segment_level": segment_table(items),
           "items": items}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in out["word_level"]:
        print("word", r)
    for r in out["segment_level"]:
        print("seg ", r)


if __name__ == "__main__":
    main()
