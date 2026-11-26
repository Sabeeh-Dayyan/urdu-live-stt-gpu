"""Offline benchmark harness (`stt bench`).

For each config in the matrix: Silero VAD + segment policy (production code) → faster-whisper on
each segment → guards (production code) → metrics:
* WER / CER vs. ground truth (whole-programme alignment, so segmentation does not matter)
* RTF = inference time ÷ programme duration (includes VAD savings) and ÷ speech duration
* simulated speech_end→final latency: event-driven single-channel queue simulation using the
  *measured* inference time of every call (segments arrive when the VAD decides them; with
  ``pack`` the server packs everything that has arrived). Live latency: `stt bench-live`.
* hallucination rate: fraction of non-speech regions where any output word lands, with and
  without guards
* timestamp error: |word start − true utterance start| for the first word of each utterance
* peak RSS, process CPU utilisation
"""

from __future__ import annotations

import json
import os
import platform
import resource
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from bench.audio import SR, load_audio
from bench.wer import wer_cer
from live_stt.asr.backends.base import AudioJob, DecodeOptions
from live_stt.asr.guards import Guards
from live_stt.logging import get_logger
from live_stt.settings import Settings, resolve_path
from live_stt.streaming.segment_policy import SegmentPolicy
from live_stt.vad.silero_onnx import FRAME, SileroVAD

log = get_logger("bench")


@dataclass(frozen=True)
class Cfg:
    model: str
    beam: int = 1
    max_seg: float = 8.0
    threads: int = 8
    compute: str = "int8"
    pack: float = 0.0  # adaptive packing budget (s); 0 = one segment per call

    @property
    def label(self) -> str:
        p = f" pack{self.pack:g}" if self.pack else ""
        return f"{self.model} {self.compute} b{self.beam} seg{self.max_seg:g} t{self.threads}{p}"


T, L = "large-v3-turbo", "large-v3"
MATRICES = {
    # Reduced matrix: one-factor-at-a-time around the CPU default (fits an overnight run).
    "default": [
        Cfg(T, 1), Cfg(T, 1, pack=28), Cfg(T, 2, pack=28), Cfg(T, 5, pack=28),
        Cfg(T, 1, 6, pack=28), Cfg(T, 1, 10, pack=28), Cfg(T, 1, threads=16, pack=28),
        Cfg(L, 1, pack=28), Cfg(L, 5, pack=28),
    ],
    "full": [Cfg(m, b, s, t) for m in (T, L) for b in (1, 2, 5) for s in (6, 8, 10)
             for t in (8, 16)],
    "quick": [Cfg(T, 1), Cfg(T, 1, pack=28)],
    # Smaller chunks / smaller packs for lower delay (CPU), vs the current live FAST config.
    "latency": [Cfg(T, 1, 10, pack=28), Cfg(T, 1, 8, pack=16), Cfg(T, 1, 6, pack=12),
                Cfg(T, 1, 4, pack=12), Cfg(T, 1, 6, pack=18)],
    "latency6": [Cfg(T, 1, 6, pack=12), Cfg(T, 1, 6, pack=18)],
    "turbo": [Cfg(T, 1), Cfg(T, 2), Cfg(T, 5), Cfg(T, 1, 6), Cfg(T, 1, 10), Cfg(T, 1, threads=16)],
    "large": [Cfg(L, 1), Cfg(L, 5)],
}


def vad_segments(audio: np.ndarray, s: Settings, max_seg: float):
    vad = SileroVAD(str(resolve_path(s.vad.model_path)))
    sc = s.streaming.model_copy(update={"max_segment_s": max_seg, "partials_enabled": False})
    pol = SegmentPolicy(s.vad, sc, SR, FRAME)
    cuts, decided_at = [], []
    t = time.perf_counter()
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        fr = audio[i:i + FRAME]
        prob = vad(fr)
        for c in pol.process(i, prob, float(np.sqrt(np.mean(fr * fr)))):
            cuts.append(c)
            decided_at.append(i + FRAME)  # sample index at which the cut became known
    for c in pol.flush(len(audio)):
        cuts.append(c)
        decided_at.append(len(audio))
    return cuts, decided_at, time.perf_counter() - t


def run_config(cfg: Cfg, s: Settings, audio, ref, backends: dict) -> dict:
    from live_stt.asr.backends.faster_whisper import FasterWhisperBackend

    key = (cfg.threads, cfg.compute, cfg.pack)
    if key not in backends:
        a = s.asr.model_copy(update={"cpu_threads": cfg.threads, "compute_type": cfg.compute,
                                     "batched": False, "pack_s": cfg.pack})
        backends[key] = FasterWhisperBackend(a)
    be = backends[key]
    be.load(cfg.model)
    be.warmup(cfg.model)
    cuts, decided, vad_s = vad_segments(audio, s, cfg.max_seg)
    guards = Guards(s.guards)
    opts = DecodeOptions(model=cfg.model, beam_size=cfg.beam, best_of=cfg.beam)
    words_g, words_raw, lat, infer_total = [], [], [], 0.0
    hyps_g, hyps_raw = [], []
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    # Event-driven simulation of the live FAST path for one channel: a segment "arrives" when the
    # VAD decided it (audio time); one server takes everything that has arrived (packed up to
    # cfg.pack seconds) and is busy for the *measured* inference time.
    arrivals = [(dec / SR, c) for c, dec in zip(cuts, decided)]
    server_t, i = 0.0, 0
    results = {}
    while i < len(arrivals):
        if arrivals[i][0] > server_t:
            server_t = arrivals[i][0]
        batch = [arrivals[i]]
        i += 1
        total = batch[0][1].n / SR
        while cfg.pack and i < len(arrivals) and arrivals[i][0] <= server_t:
            d = arrivals[i][1].n / SR
            if total + 0.3 + d > cfg.pack:
                break
            batch.append(arrivals[i])
            total += 0.3 + d
            i += 1
        jobs = [AudioJob(str(id(c)), audio[c.start:c.end], language="ur") for _, c in batch]
        t = time.perf_counter()
        res = be.transcribe(jobs, opts)
        dt = time.perf_counter() - t
        infer_total += dt
        server_t += dt
        for (_, c), r in zip(batch, res):
            results[id(c)] = r
            lat.append(server_t - c.speech_end / SR)
    for c in cuts:
        res = results[id(c)]
        dur = c.n / SR
        g = guards.apply(res, dur)
        off = c.start / SR
        hyps_g.append(g.text)
        hyps_raw.append(res.text)
        words_g += [(off + w.start, off + w.end, w.word) for w in g.words]
        words_raw += [(off + w.start, off + w.end, w.word) for sg in res.segments for w in sg.words]
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    dur_total = len(audio) / SR
    speech_s = sum(x["end"] - x["start"] for x in ref["segments"])
    refs = [x["text"] for x in ref["segments"]]

    def halluc(words):
        hit = 0
        for ns in ref["nonspeech"]:
            if any(ns["start"] <= (a + b) / 2 <= ns["end"] for a, b, _ in words):
                hit += 1
        return hit / len(ref["nonspeech"]) if ref["nonspeech"] else None

    ts_err = []
    for u in ref["segments"]:
        onset = u.get("speech_start", u["start"])
        first = [a for a, b, _ in words_g if u["start"] - 0.5 <= a <= u["end"]]
        if first:
            ts_err.append(abs(min(first) - onset))
    wc = wer_cer(refs, hyps_g)
    wc_raw = wer_cer(refs, hyps_raw)
    return {
        "config": asdict(cfg), "label": cfg.label,
        "wer": wc["wer"], "cer": wc["cer"], "wer_noguard": wc_raw["wer"],
        "ref_words": wc["ref_words"], "hyp_words": wc["hyp_words"],
        "rtf": infer_total / dur_total, "rtf_speech": infer_total / max(speech_s, 1e-6),
        "vad_rtf": vad_s / dur_total,
        "segments": len(cuts),
        "latency_sim_p50": float(np.percentile(lat, 50)) if lat else None,
        "latency_sim_p95": float(np.percentile(lat, 95)) if lat else None,
        "halluc_rate": halluc(words_g), "halluc_rate_noguard": halluc(words_raw),
        "ts_err_p50": float(np.percentile(ts_err, 50)) if ts_err else None,
        "ts_err_p95": float(np.percentile(ts_err, 95)) if ts_err else None,
        "cpu_util": cpu / max(wall, 1e-6),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        if platform.system() == "Darwin" else
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "hyp_sample": " | ".join(h for h in hyps_g[:3] if h)[:300],
    }


def environment() -> dict:
    import ctranslate2
    import faster_whisper
    import onnxruntime

    cpu = platform.processor()
    try:
        import subprocess

        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                      text=True).strip()
    except Exception:  # noqa: BLE001
        pass
    return {"cpu": cpu, "cores_logical": os.cpu_count(), "os": platform.platform(),
            "python": platform.python_version(), "faster_whisper": faster_whisper.__version__,
            "ctranslate2": ctranslate2.__version__, "onnxruntime": onnxruntime.__version__,
            "date_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime())}


def main(s: Settings, dataset: str, matrix: str, limit_s: float, out: str) -> None:
    ref = json.loads(Path(dataset).read_text("utf-8"))
    audio = load_audio(ref["audio"])
    if limit_s:
        audio = audio[: int(limit_s * SR)]
        ref = dict(ref, segments=[x for x in ref["segments"] if x["end"] <= limit_s],
                   nonspeech=[x for x in ref["nonspeech"] if x["end"] <= limit_s])
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    res_path = outdir / f"results_{matrix}.json"
    done = json.loads(res_path.read_text()) if res_path.exists() else {"runs": []}
    done_labels = {r["label"] for r in done["runs"]}
    done["environment"] = environment()
    done["dataset"] = {"name": ref["name"], "duration_s": len(audio) / SR,
                       "utterances": len(ref["segments"]), "nonspeech": len(ref["nonspeech"]),
                       "source": ref.get("source")}
    backends: dict = {}
    for cfg in MATRICES[matrix]:
        if cfg.label in done_labels:
            log.info("bench_skip_done", config=cfg.label)
            continue
        log.info("bench_run", config=cfg.label)
        try:
            r = run_config(cfg, s, audio, ref, backends)
        except Exception as e:  # noqa: BLE001 - record and continue the matrix
            log.error("bench_failed", config=cfg.label, error=str(e))
            r = {"label": cfg.label, "config": asdict(cfg), "error": str(e)}
        log.info("bench_result", **{k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in r.items() if k != "config"})
        done["runs"].append(r)
        res_path.write_text(json.dumps(done, ensure_ascii=False, indent=1))
    from bench.report import write_report

    write_report([res_path], Path("docs/BENCHMARKS.md"), s)
