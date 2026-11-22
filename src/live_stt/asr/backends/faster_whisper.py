"""faster-whisper (CTranslate2) backend.

Verified against faster-whisper 1.2.1 / ctranslate2 4.9.0:
* ``WhisperModel(path, device, device_index, compute_type, cpu_threads, num_workers,
  download_root, local_files_only)``
* ``WhisperModel.transcribe(audio: np.ndarray, language, beam_size, best_of, temperature,
  condition_on_previous_text, initial_prompt, hotwords, word_timestamps, vad_filter, ...)`` →
  (lazy generator of Segment(start, end, text, avg_logprob, no_speech_prob, compression_ratio,
  words[Word(start, end, word, probability)]), info)
* ``BatchedInferencePipeline(model).transcribe(audio, clip_timestamps=[{"start": s, "end": e}],
  batch_size=n, ...)``: each clip (seconds) becomes one batch item; segment times are offsets in
  the concatenated audio. We use this for **cross-channel micro-batching**: jobs are concatenated,
  each job is a clip, and results are mapped back by offset. Batched mode shares one
  initial_prompt/hotwords across the batch and has no temperature fallback.
"""

from __future__ import annotations

import bisect
import threading
import time
from pathlib import Path

import numpy as np

from live_stt.asr.backends.base import (
    AsrResult,
    ASRBackend,
    AudioJob,
    DecodeOptions,
    SegmentResult,
    WordResult,
)
from live_stt.logging import get_logger
from live_stt.settings import AsrConfig, resolve_path

log = get_logger(__name__)
SR = 16_000
GAP_S = 0.1  # silence inserted between concatenated batch items


def _resolve_model(name: str, download_root: str) -> tuple[str, bool]:
    """Prefer a local directory (models/whisper/<name>) → (path, local_files_only)."""
    p = Path(name)
    if p.is_dir():
        return str(p), True
    local = resolve_path(download_root) / name
    if (local / "model.bin").exists():
        return str(local), True
    return name, False


class FasterWhisperBackend(ASRBackend):
    name = "faster-whisper"

    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self._models: dict[str, object] = {}
        self._batched: dict[str, object] = {}
        self._lock = threading.Lock()
        dev = cfg.device
        if dev == "auto":
            try:
                import ctranslate2

                dev = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:  # noqa: BLE001
                dev = "cpu"
        self._device = dev

    @property
    def device(self) -> str:
        return self._device

    def load(self, model_name: str):
        with self._lock:
            if model_name in self._models:
                return self._models[model_name]
            from faster_whisper import BatchedInferencePipeline, WhisperModel

            path, local_only = _resolve_model(model_name, self.cfg.download_root)
            t = time.perf_counter()
            m = WhisperModel(
                path,
                device=self._device,
                device_index=self.cfg.device_index,
                compute_type=self.cfg.compute_type,
                cpu_threads=self.cfg.cpu_threads,
                num_workers=self.cfg.num_workers,
                download_root=str(resolve_path(self.cfg.download_root)),
                local_files_only=local_only,
            )
            log.info("model_loaded", model=model_name, path=path, device=self._device,
                     compute_type=self.cfg.compute_type, load_s=round(time.perf_counter() - t, 1))
            self._models[model_name] = m
            self._batched[model_name] = BatchedInferencePipeline(m)
            return m

    def warmup(self, model_name: str) -> float:
        m = self.load(model_name)
        t = time.perf_counter()
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(SR) * 0.01).astype(np.float32)
        segs, _ = m.transcribe(audio, language="ur", beam_size=1, word_timestamps=True,
                               condition_on_previous_text=False)
        list(segs)
        return time.perf_counter() - t

    # ------------------------------------------------------------- transcribe
    def transcribe(self, batch: list[AudioJob], opts: DecodeOptions) -> list[AsrResult]:
        if not batch:
            return []
        if self.cfg.batched and len(batch) > 1:
            return self._transcribe_batched(batch, opts)
        if self.cfg.pack_s > 0 and len(batch) > 1:
            return self._transcribe_packed(batch, opts)
        return [self._transcribe_one(j, opts) for j in batch]

    def _transcribe_packed(self, batch: list[AudioJob], opts: DecodeOptions) -> list[AsrResult]:
        """Concatenate jobs (with silence gaps) into one ≤30 s window and decode it in a single
        encoder pass (BatchedInferencePipeline with one clip, without_timestamps → no seek
        re-encoding). Words are assigned back to jobs by their midpoint."""
        self.load(opts.model)
        pipe = self._batched[opts.model]
        gap = np.zeros(int(PACK_GAP_S * SR), np.float32)
        parts, spans, pos = [], [], 0
        for j in batch:
            a = _sanitize(j.audio)
            spans.append((pos / SR, (pos + len(a)) / SR))
            parts += [a, gap]
            pos += len(a) + len(gap)
        audio = np.concatenate(parts)[: 30 * SR]
        t = time.perf_counter()
        segs, info = pipe.transcribe(
            audio, language=batch[0].language or opts.language,
            clip_timestamps=[{"start": 0.0, "end": len(audio) / SR}], batch_size=1,
            beam_size=opts.beam_size, best_of=opts.best_of, temperature=opts.temperature,
            initial_prompt=batch[0].initial_prompt, hotwords=batch[0].hotwords or None,
            without_timestamps=True, word_timestamps=True, condition_on_previous_text=False,
            vad_filter=False, no_speech_threshold=None,
        )
        segs = list(segs)
        dt = time.perf_counter() - t
        return split_packed(batch, spans, segs, dt, info.language, opts.model)

    def _common_kwargs(self, opts: DecodeOptions) -> dict:
        return dict(
            beam_size=opts.beam_size,
            best_of=opts.best_of,
            condition_on_previous_text=False,  # prevents repetition loops on long streams
            word_timestamps=opts.word_timestamps,
            vad_filter=False,  # we already VAD-gate upstream
            # Thresholds are applied by our guards (so we can count/flag), not by the decoder's
            # internal fallback filtering, except temperature fallback for non-batched decode.
            no_speech_threshold=None,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

    def _transcribe_one(self, job: AudioJob, opts: DecodeOptions) -> AsrResult:
        m = self.load(opts.model)
        audio = _sanitize(job.audio)
        t = time.perf_counter()
        temps = [opts.temperature, 0.2, 0.4, 0.6] if opts.temperature == 0 else [opts.temperature]
        segs, info = m.transcribe(
            audio,
            language=job.language or opts.language,
            temperature=temps,
            initial_prompt=job.initial_prompt if job.initial_prompt is not None else opts.initial_prompt,
            hotwords=(job.hotwords if job.hotwords is not None else opts.hotwords) or None,
            **self._common_kwargs(opts),
        )
        out = [_seg(s, 0.0) for s in segs]  # generator: decoding happens here
        dt = time.perf_counter() - t
        return AsrResult(job.job_id, out, info.language, dt, opts.model)

    def _transcribe_batched(self, batch: list[AudioJob], opts: DecodeOptions) -> list[AsrResult]:
        self.load(opts.model)
        pipe = self._batched[opts.model]
        gap = np.zeros(int(GAP_S * SR), np.float32)
        parts, clips, offsets = [], [], []
        pos = 0
        for j in batch:
            a = _sanitize(j.audio)[: 30 * SR]  # Whisper window
            offsets.append(pos / SR)
            clips.append({"start": pos / SR, "end": (pos + len(a)) / SR})
            parts += [a, gap]
            pos += len(a) + len(gap)
        audio = np.concatenate(parts)
        lang = batch[0].language or opts.language
        prompt = opts.initial_prompt
        if all(j.initial_prompt == batch[0].initial_prompt for j in batch):
            prompt = batch[0].initial_prompt
        t = time.perf_counter()
        segs, info = pipe.transcribe(
            audio,
            language=lang,
            clip_timestamps=clips,
            batch_size=len(batch),
            temperature=opts.temperature,
            initial_prompt=prompt,
            hotwords=opts.hotwords or None,
            without_timestamps=True,
            **self._common_kwargs(opts),
        )
        segs = list(segs)
        dt = time.perf_counter() - t
        per_job: list[list[SegmentResult]] = [[] for _ in batch]
        starts = [c["start"] for c in clips]
        for s in segs:
            k = clip_index(starts, s.start)
            per_job[k].append(_seg(s, offsets[k]))
        share = dt / len(batch)
        return [AsrResult(j.job_id, per_job[i], info.language, share, opts.model)
                for i, j in enumerate(batch)]


PACK_GAP_S = 0.3


def split_packed(batch, spans, segs, dt, language, model) -> list[AsrResult]:
    """Assign words of a packed decode to their jobs (by midpoint) and rebuild per-job results.

    Segment-level stats (avg_logprob, no_speech_prob) come from the packed segment(s) a job's
    words fell into; compression_ratio is recomputed from the job's own text."""
    from faster_whisper.transcribe import get_compression_ratio

    per_words: list[list[WordResult]] = [[] for _ in batch]
    per_stats: list[list] = [[] for _ in batch]
    starts = [s for s, _ in spans]
    for sg in segs:
        for w in sg.words or []:
            mid = (w.start + w.end) / 2
            k = clip_index(starts, mid)
            s0, e0 = spans[k]
            if mid > e0 + PACK_GAP_S / 2:  # inside a gap after the last job
                continue
            per_words[k].append(WordResult(w.word, max(0.0, w.start - s0),
                                           min(e0 - s0, w.end - s0), float(w.probability)))
            per_stats[k].append(sg)
    total_audio = sum(e - s for s, e in spans) or 1.0
    out = []
    for i, j in enumerate(batch):
        ws = per_words[i]
        text = "".join(w.word for w in ws).strip()
        st = per_stats[i]
        seg = SegmentResult(
            text=text, start=ws[0].start if ws else 0.0, end=ws[-1].end if ws else 0.0,
            avg_logprob=float(np.mean([s.avg_logprob for s in st])) if st else 0.0,
            no_speech_prob=float(max(s.no_speech_prob for s in st)) if st else 0.0,
            compression_ratio=float(get_compression_ratio(text)) if text else 0.0, words=ws)
        share = dt * (spans[i][1] - spans[i][0]) / total_audio
        out.append(AsrResult(j.job_id, [seg] if text else [], language, share, model))
    return out


def clip_index(starts: list[float], t: float) -> int:
    """Index of the batch item (clip) whose start is the last one ≤ t."""
    return max(0, bisect.bisect_right(starts, t + 1e-3) - 1)


def _sanitize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(a)):
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if len(a) < SR // 10:  # Whisper needs some audio; pad very short jobs
        a = np.pad(a, (0, SR // 10 - len(a)))
    return np.clip(a, -1.0, 1.0)


def _seg(s, offset: float) -> SegmentResult:
    words = [
        WordResult(w.word, w.start - offset, w.end - offset, float(w.probability))
        for w in (s.words or [])
    ]
    return SegmentResult(
        text=s.text,
        start=s.start - offset,
        end=s.end - offset,
        avg_logprob=float(s.avg_logprob),
        no_speech_prob=float(s.no_speech_prob),
        compression_ratio=float(s.compression_ratio),
        words=words,
    )
