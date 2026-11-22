"""Hallucination-guard evaluation: feed NON-speech straight to Whisper (bypassing VAD, as happens
when the VAD is fooled by music/applause) and count outputs with and without guards.

python -m bench.guard_eval [--model large-v3-turbo] → bench/out/guards.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from bench.audio import SR, load_audio  # noqa: E402
from bench.make_reference import jingle, noise  # noqa: E402
from live_stt.asr.backends.base import AudioJob, DecodeOptions  # noqa: E402
from live_stt.asr.backends.faster_whisper import FasterWhisperBackend  # noqa: E402
from live_stt.asr.guards import Guards  # noqa: E402
from live_stt.settings import load_settings  # noqa: E402


def clips(ref_path: str, n_synth: int = 6) -> list[tuple[str, np.ndarray]]:
    ref = json.loads(Path(ref_path).read_text("utf-8"))
    audio = load_audio(ref["audio"])
    out = [(f"ref-{ns['kind']}", audio[int(ns["start"] * SR): int(ns["end"] * SR)])
           for ns in ref["nonspeech"]]
    rng = np.random.default_rng(11)
    for i in range(n_synth):
        out.append((f"jingle-{i}", jingle(rng.uniform(4, 8), rng)))
        out.append((f"noise-{i}", noise(rng.uniform(4, 8), rng)))
        hum = (0.02 * np.sin(2 * np.pi * 50 * np.arange(int(5 * SR)) / SR)).astype(np.float32)
        out.append((f"hum-{i}", hum + (0.003 * rng.standard_normal(len(hum))).astype(np.float32)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--config", default="config/dev-mac-cpu.yaml")
    ap.add_argument("--reference", default="testbed/reference/fleurs_ur.json")
    a = ap.parse_args()
    s = load_settings(a.config)
    be = FasterWhisperBackend(s.asr.model_copy(update={"pack_s": 0}))
    guards = Guards(s.guards)
    rows = []
    for name, x in clips(a.reference):
        res = be.transcribe([AudioJob(name, x, language="ur")], DecodeOptions(model=a.model))[0]
        g = guards.apply(res, len(x) / SR)
        rows.append({"clip": name, "raw": res.text, "guarded": g.text, "drops": g.drops})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    n = len(rows)
    summary = {"model": a.model, "clips": n,
               "hallucinated_raw": sum(bool(r["raw"].strip()) for r in rows),
               "hallucinated_after_guards": sum(bool(r["guarded"].strip()) for r in rows),
               "rows": rows}
    Path("bench/out").mkdir(parents=True, exist_ok=True)
    Path("bench/out/guards.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print({k: v for k, v in summary.items() if k != "rows"})


if __name__ == "__main__":
    main()
