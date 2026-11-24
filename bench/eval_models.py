"""Per-utterance model comparison on FLEURS Urdu (test split): WER, CER and SER.

Unlike ``run_bench.py`` (a broadcast-like programme through the live pipeline: VAD, segmenting,
packing, latency), this decodes each reference utterance on its own, the way ASR papers report
accuracy, so models can be compared without the pipeline's segmentation effects.

* WER / CER: corpus-level after ``live_stt.text.urdu_normalize.normalize`` (CER without spaces —
  Urdu word spacing in FLEURS references is inconsistent).
* SER (sentence error rate): share of utterances whose normalised hypothesis is not identical to
  the normalised reference.
* Also: per-utterance WER median, hallucination-ish outputs (hyp ≥ 2× ref words), RTF.

Contamination: models fine-tuned on FLEURS train/test must not be judged on this set (see
docs/MODELS.md). Usage::

    python -m bench.eval_models --model large-v3 --model /path/to/ct2_model --every 2 \\
        --compute-type int8 --beam 5 --out bench/results/eval_fleurs_models.json
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import jiwer

from bench.audio import load_audio
from live_stt.settings import resolve_path
from live_stt.text.urdu_normalize import normalize

FLEURS = Path("testbed/data/fleurs_ur")


def utterances(every: int, limit: int | None) -> list[tuple[str, Path, str]]:
    rows = list(csv.reader(open(FLEURS / "test.tsv", encoding="utf-8"), delimiter="\t",
                           quoting=csv.QUOTE_NONE))
    seen, out = set(), []
    for r in rows:
        if len(r) < 3 or r[1] in seen:
            continue
        p = FLEURS / "audio" / "test" / r[1]
        if p.exists():
            seen.add(r[1])
            out.append((r[0], p, r[2]))
    out = out[::every]
    return out[:limit] if limit else out


def score(refs: list[str], hyps: list[str]) -> dict:
    R = [normalize(x) for x in refs]
    H = [normalize(x) for x in hyps]
    pairs = [(r, h) for r, h in zip(R, H) if r]
    r_all, h_all = " ".join(r for r, _ in pairs), " ".join(h for _, h in pairs)
    per = [jiwer.wer(r, h) if h else 1.0 for r, h in pairs]
    return {
        "wer": jiwer.wer(r_all, h_all) if h_all else 1.0,
        "cer": jiwer.cer(r_all.replace(" ", ""), h_all.replace(" ", "")) if h_all else 1.0,
        "ser": sum(r != h for r, h in pairs) / len(pairs),
        "utt_wer_median": statistics.median(per),
        "long_outputs": sum(len(h.split()) >= 2 * max(1, len(r.split())) for r, h in pairs),
        "utterances": len(pairs), "ref_words": len(r_all.split()), "hyp_words": len(h_all.split()),
    }


def run(model: str, utts, compute_type: str, beam: int, threads: int) -> dict:
    from faster_whisper import WhisperModel

    local = resolve_path("models/whisper") / model
    path = str(local) if local.is_dir() else model
    t = time.time()
    m = WhisperModel(path, device="cpu", compute_type=compute_type, cpu_threads=threads,
                     local_files_only=True)
    load_s = time.time() - t
    hyps, audio_s, infer_s, samples = [], 0.0, 0.0, []
    for i, (uid, p, ref) in enumerate(utts):
        a = load_audio(p)
        audio_s += len(a) / 16000
        t = time.time()
        segs, _ = m.transcribe(a, language="ur", task="transcribe", beam_size=beam,
                               best_of=beam, temperature=0.0, vad_filter=False,
                               condition_on_previous_text=False)
        hyp = " ".join(s.text.strip() for s in segs).strip()
        infer_s += time.time() - t
        hyps.append(hyp)
        if i < 3:
            samples.append({"id": uid, "ref": ref, "hyp": hyp})
        if (i + 1) % 25 == 0:
            print(f"  {model}: {i + 1}/{len(utts)} rtf={infer_s / audio_s:.2f}", flush=True)
    out = {"model": model, "compute_type": compute_type, "beam": beam,
           **score([u[2] for u in utts], hyps), "audio_s": round(audio_s, 1),
           "rtf": infer_s / audio_s, "load_s": round(load_s, 1), "samples": samples,
           "hyps": hyps}
    del m
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="model name under models/whisper, or a CTranslate2 model directory")
    ap.add_argument("--label", action="append", help="display label per --model")
    ap.add_argument("--every", type=int, default=2, help="take every Nth utterance")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="bench/results/eval_fleurs_models.json")
    a = ap.parse_args()
    utts = utterances(a.every, a.limit)
    print(f"{len(utts)} utterances", flush=True)
    out_p = Path(a.out)
    res = json.loads(out_p.read_text()) if out_p.exists() else {"runs": []}
    res.update({"dataset": {"name": "fleurs_ur test", "every": a.every, "limit": a.limit,
                            "utterances": len(utts), "ids": [u[0] for u in utts]},
                "environment": {"machine": platform.machine(), "processor": platform.processor(),
                                "python": platform.python_version()},
                "references": [u[2] for u in utts]})
    labels = a.label or []
    for i, m in enumerate(a.model):
        r = run(m, utts, a.compute_type, a.beam, a.threads)
        r["label"] = labels[i] if i < len(labels) else Path(m).name
        res["runs"] = [x for x in res["runs"] if x.get("label") != r["label"]] + [r]
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                          if k not in ("samples", "hyps")}), flush=True)


if __name__ == "__main__":
    main()
