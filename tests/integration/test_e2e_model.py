"""End-to-end with the real models: Silero VAD + faster-whisper on real Urdu (FLEURS programme).

Skips when the model or reference programme is not present (`make models reference`).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from live_stt.settings import PROJECT_ROOT, Settings, ChannelConfig

TURBO = PROJECT_ROOT / "models/whisper/large-v3-turbo/model.bin"
REF = PROJECT_ROOT / "testbed/reference/fleurs_ur.json"
VAD = PROJECT_ROOT / "models/silero_vad.onnx"
pytestmark = [
    pytest.mark.skipif(not TURBO.exists(), reason="large-v3-turbo not downloaded"),
    pytest.mark.skipif(not REF.exists(), reason="reference programme missing (make reference)"),
    pytest.mark.skipif(not VAD.exists(), reason="silero VAD missing"),
    pytest.mark.timeout(900),
]
T0 = 1_763_000_000.0
SR = 16_000


@pytest.fixture(scope="module")
def backend():
    from live_stt.asr.backends.faster_whisper import FasterWhisperBackend

    s = Settings()
    s.asr.cpu_threads = 8
    be = FasterWhisperBackend(s.asr)
    be.load("large-v3-turbo")
    return be


@pytest.fixture(scope="module")
def programme():
    from bench.audio import load_audio

    ref = json.loads(REF.read_text("utf-8"))
    return ref, load_audio(PROJECT_ROOT / ref["audio"])


def run_pipeline(audio, backend, refine=False):
    from live_stt.asr.service import AsrService
    from live_stt.bus.base import InMemoryBus
    from live_stt.ingest.source import AudioChunk
    from live_stt.vad.silero_onnx import SileroVAD
    from live_stt.worker import ChannelPipeline
    from tests.unit.test_asr_service import drain

    s = Settings()
    s.asr.refine_enabled = refine
    s.asr.refine = s.asr.fast  # REFINE with turbo here (large-v3 covered by bench)
    s.channels = [ChannelConfig(id="e2e", url="udp://x:1")]
    bus = InMemoryBus()
    pipe = ChannelPipeline(s, s.channels[0], bus, SileroVAD(str(VAD)))
    for i in range(0, len(audio), 2048):
        pipe.on_chunk(AudioChunk(audio[i:i + 2048], T0 + i / SR), True)
    pipe._flush()
    svc = AsrService(s, bus, backend, consumer="e2e")
    drain(svc)
    return [e for e in bus.messages(s.bus.stream_transcripts)]


def test_real_urdu_transcription_quality_and_timestamps(programme, backend):
    from bench.wer import wer_cer

    ref, audio = programme
    end = ref["segments"][4]["end"] + 1.0  # first 5 utterances (incl. one non-speech section)
    evs = run_pipeline(audio[: int(end * SR)], backend, refine=True)
    finals = [e for e in evs if e["type"] == "final"]
    refined = [e for e in evs if e["type"] == "refined"]
    assert finals and len(refined) == len(finals)
    refs = [u["text"] for u in ref["segments"][:5]]
    wc = wer_cer(refs, [e["text_raw"] for e in finals])
    print("e2e WER/CER", wc)
    assert wc["wer"] < 0.5 and wc["cer"] < 0.3, wc  # sanity bound; real numbers in BENCHMARKS.md
    # Every word must fall inside (or within 0.5 s of) a true utterance window.
    wins = [(T0 + u["start"] - 0.5, T0 + u["end"] + 0.5) for u in ref["segments"][:5]]
    words = [w for e in finals for w in e["words"]]
    inside = sum(any(a <= w["s"] <= b for a, b in wins) for w in words)
    assert inside / max(1, len(words)) > 0.97, (inside, len(words))
    # First word of each utterance starts close to the true speech onset (VAD-annotated; FLEURS
    # recordings begin with 0.3–1.3 s of silence).
    errs = []
    for u, (a, b) in zip(ref["segments"][:5], wins):
        ws = [w["s"] for w in words if a <= w["s"] <= b]
        if ws:
            errs.append(abs(min(ws) - (T0 + u["speech_start"])))
    print("first-word timestamp errors (s):", [round(e, 3) for e in errs])
    assert np.median(errs) < 0.3, errs


def test_no_text_on_non_speech_sections(programme, backend):
    ref, audio = programme
    for ns in ref["nonspeech"][:3]:
        seg = audio[int(ns["start"] * SR): int(ns["end"] * SR)]
        evs = run_pipeline(seg, backend)
        texts = [e["text_raw"] for e in evs if e["type"] == "final" and e["text_raw"]]
        assert texts == [], (ns["kind"], texts)


def test_backend_batched_mode_matches_sequential(programme, backend):
    """Cross-job micro-batching (clip_timestamps) returns per-job results in order."""
    from live_stt.asr.backends.base import AudioJob, DecodeOptions
    from live_stt.asr.backends.faster_whisper import FasterWhisperBackend

    ref, audio = programme
    jobs = [AudioJob(str(i), audio[int(u["start"] * SR): int(u["end"] * SR)])
            for i, u in enumerate(ref["segments"][:3])]
    opts = DecodeOptions(model="large-v3-turbo")
    seq = backend.transcribe(jobs, opts)
    s = Settings()
    s.asr.batched = True
    bb = FasterWhisperBackend(s.asr)
    bb._models = backend._models
    bb._batched = backend._batched
    bat = bb.transcribe(jobs, opts)
    assert [r.job_id for r in bat] == ["0", "1", "2"]
    for a, b in zip(seq, bat):
        sa, sb = set(a.text.split()), set(b.text.split())
        assert len(sa & sb) / max(1, len(sa | sb)) > 0.6, (a.text, b.text)
        for seg in b.segments:
            for w in seg.words:
                assert -0.3 <= w.start <= jobs[int(b.job_id)].duration_s + 0.3
