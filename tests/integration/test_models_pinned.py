"""Downloaded models are exactly the pinned files the benchmarks were measured with."""

from live_stt import models


def test_local_models_match_pins_if_present():
    """The repo's own models (when downloaded) must be exactly the pinned, benchmarked files."""
    from live_stt.settings import resolve_path

    vad = resolve_path("models/silero_vad.onnx")
    if vad.is_file():
        assert models.sha256(vad) == models.VAD["sha256"]
    for name, pin in models.WHISPER.items():
        mb = resolve_path("models/whisper") / name / "model.bin"
        if mb.is_file() and mb.stat().st_size < 4 << 30:
            assert models.sha256(mb) == pin["sha256"], name
