"""Build a broadcast-like Urdu reference programme from FLEURS (ur_pk, test split).

Real read Urdu speech with human transcripts, concatenated with random pauses and inserted
non-speech sections (music-like jingle, noise, silence) so that:
* WER/CER can be computed against ground truth,
* every utterance has a known start/end (timestamp accuracy + live latency),
* non-speech regions measure hallucinations.

Usage: python -m bench.make_reference --n 40 --out testbed/reference/fleurs_ur
       python -m bench.make_reference --draft testbed/media/news.ts   (large-v3 draft for a human)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from bench.audio import SR, load_audio, write_wav

FLEURS = Path("testbed/data/fleurs_ur")


def jingle(seconds: float, rng) -> np.ndarray:
    """Music-like: chord progression with tremolo + percussive noise bursts."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    chords = [(261.6, 329.6, 392.0), (220.0, 261.6, 329.6), (174.6, 220.0, 261.6),
              (196.0, 246.9, 293.7)]
    out = np.zeros(n, np.float32)
    beat = int(0.5 * SR)
    for i in range(0, n, beat * 2):
        ch = chords[(i // (beat * 2)) % len(chords)]
        seg = slice(i, min(n, i + beat * 2))
        for f in ch:
            out[seg] += 0.08 * np.sin(2 * np.pi * f * t[seg])
    out *= 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)
    for i in range(0, n, beat):
        k = min(n - i, int(0.06 * SR))
        out[i:i + k] += 0.3 * rng.standard_normal(k) * np.exp(-np.arange(k) / (0.01 * SR))
    return out.astype(np.float32)


def noise(seconds: float, rng) -> np.ndarray:
    n = int(seconds * SR)
    w = rng.standard_normal(n)
    pink = np.cumsum(w)
    pink -= np.convolve(pink, np.ones(400) / 400, mode="same")
    return (0.05 * pink / (np.abs(pink).max() + 1e-9)).astype(np.float32)


def build(n: int, out: Path, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    rows = list(csv.reader(open(FLEURS / "test.tsv", encoding="utf-8"), delimiter="\t",
                           quoting=csv.QUOTE_NONE))
    seen, picked = set(), []
    for r in rows:
        if r[1] in seen or len(r) < 3:
            continue
        p = FLEURS / "audio" / "test" / r[1]
        if p.exists():
            seen.add(r[1])
            picked.append((r[0], p, r[2]))
        if len(picked) >= n:
            break
    parts, segs, nonspeech = [np.zeros(int(1.0 * SR), np.float32)], [], []
    pos = len(parts[0])
    for k, (uid, path, text) in enumerate(picked):
        a = load_audio(path)
        a = a / (np.abs(a).max() + 1e-9) * 0.7
        segs.append({"id": uid, "file": path.name, "start": pos / SR,
                     "end": (pos + len(a)) / SR, "text": text})
        parts.append(a)
        pos += len(a)
        gap = rng.uniform(0.6, 1.6)
        if k % 8 == 3:
            kind = ["jingle", "noise", "silence"][(k // 8) % 3]
            dur = rng.uniform(4, 8)
            filler = jingle(dur, rng) if kind == "jingle" else (
                noise(dur, rng) if kind == "noise" else np.zeros(int(dur * SR), np.float32))
            g1 = np.zeros(int(0.8 * SR), np.float32)
            parts += [g1, filler, g1]
            nonspeech.append({"kind": kind, "start": (pos + len(g1)) / SR,
                              "end": (pos + len(g1) + len(filler)) / SR})
            pos += len(g1) * 2 + len(filler)
        else:
            parts.append(np.zeros(int(gap * SR), np.float32))
            pos += int(gap * SR)
    audio = np.concatenate(parts)
    audio += (0.0005 * rng.standard_normal(len(audio))).astype(np.float32)  # noise floor
    out.parent.mkdir(parents=True, exist_ok=True)
    wav = out.with_suffix(".wav")
    write_wav(wav, audio)
    ref = {"name": out.name, "source": "google/fleurs ur_pk test (CC-BY-4.0)",
           "audio": str(wav), "sample_rate": SR, "duration_s": len(audio) / SR,
           "segments": segs, "nonspeech": nonspeech}
    ref = annotate_onsets(ref, audio)
    out.with_suffix(".json").write_text(json.dumps(ref, ensure_ascii=False, indent=1), "utf-8")
    print(f"wrote {wav} ({len(audio) / SR:.1f}s, {len(segs)} utterances, "
          f"{len(nonspeech)} non-speech) and {out.with_suffix('.json')}")


def annotate_onsets(ref: dict, audio: np.ndarray) -> dict:
    """Add VAD-derived speech_start/speech_end per utterance (FLEURS files start with 0.3–1.3 s
    of silence, so the file start is not the speech onset)."""
    sys.path.insert(0, "src")
    from live_stt.vad.silero_onnx import SileroVAD

    v = SileroVAD("models/silero_vad.onnx")
    for u in ref["segments"]:
        v.reset()
        s, e = int(u["start"] * SR), int(u["end"] * SR)
        probs = [v(audio[i:i + 512]) for i in range(s, e - 512, 512)]
        on = next((k for k, p in enumerate(probs) if p > 0.5), 0)
        off = len(probs) - next((k for k, p in enumerate(reversed(probs)) if p > 0.5), 0)
        u["speech_start"] = round(u["start"] + on * 512 / SR, 3)
        u["speech_end"] = round(u["start"] + off * 512 / SR, 3)
    return ref


def draft(media: str, out: str | None) -> None:
    """Offline large-v3 beam-5 draft transcript for a human to correct into ground truth."""
    sys.path.insert(0, "src")
    from live_stt.asr.backends.faster_whisper import FasterWhisperBackend
    from live_stt.settings import load_settings

    s = load_settings()
    be = FasterWhisperBackend(s.asr)
    m = be.load("large-v3")
    audio = load_audio(media)
    segs, _ = m.transcribe(audio, language="ur", beam_size=5, best_of=5, vad_filter=True,
                           condition_on_previous_text=False)
    ref = {"name": Path(media).stem, "audio": media, "sample_rate": SR,
           "duration_s": len(audio) / SR, "draft": True,
           "segments": [{"id": str(i), "start": x.start, "end": x.end, "text": x.text.strip()}
                        for i, x in enumerate(segs)], "nonspeech": []}
    dst = Path(out or Path(media).with_suffix(".draft.json"))
    dst.write_text(json.dumps(ref, ensure_ascii=False, indent=1), "utf-8")
    print(f"draft written to {dst} — correct the text, then remove \"draft\": true")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="testbed/reference/fleurs_ur")
    ap.add_argument("--draft")
    ap.add_argument("--annotate", help="add speech onsets to an existing reference JSON")
    a = ap.parse_args()
    if a.annotate:
        r = json.loads(Path(a.annotate).read_text("utf-8"))
        r = annotate_onsets(r, load_audio(r["audio"]))
        Path(a.annotate).write_text(json.dumps(r, ensure_ascii=False, indent=1), "utf-8")
    elif a.draft:
        draft(a.draft, None)
    else:
        build(a.n, Path(a.out))
