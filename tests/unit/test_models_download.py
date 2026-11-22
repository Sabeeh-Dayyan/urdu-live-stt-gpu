"""Model planning, pins and checksum verification (no network)."""

import hashlib
from pathlib import Path

import pytest

from live_stt import models
from live_stt.settings import Settings


def _settings(tmp_path: Path, **asr) -> Settings:
    s = Settings()
    s.vad.model_path = str(tmp_path / "vad.onnx")
    s.asr.download_root = str(tmp_path / "whisper")
    for k, v in asr.items():
        setattr(s.asr, k, v)
    return s


def test_required_lists_vad_fast_and_refine_once(tmp_path):
    s = _settings(tmp_path)
    s.asr.fast.model, s.asr.refine.model = "large-v3-turbo", "large-v3"
    names = [i.name for i in models.required(s)]
    assert names == ["silero-vad", "large-v3-turbo", "large-v3"]
    assert not any(i.present for i in models.required(s))
    s.asr.refine.model = "large-v3-turbo"
    assert [i.name for i in models.required(s)] == ["silero-vad", "large-v3-turbo"]
    s.asr.refine_enabled = False
    s.asr.refine.model = "large-v3"
    assert [i.name for i in models.required(s)] == ["silero-vad", "large-v3-turbo"]


def test_presence_and_local_model_dir(tmp_path):
    s = _settings(tmp_path)
    s.asr.fast.model = "large-v3-turbo"
    s.asr.refine_enabled = False
    d = tmp_path / "whisper" / "large-v3-turbo"
    d.mkdir(parents=True)
    (d / "model.bin").write_bytes(b"x")
    (tmp_path / "vad.onnx").write_bytes(b"y")
    assert all(i.present for i in models.required(s))
    local = tmp_path / "my-finetune"
    local.mkdir()
    s.asr.fast.model = str(local)
    it = models.required(s)[1]
    assert it.pinned is False and it.present is False and it.path == local


def test_verify_rejects_wrong_checksum(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    s.asr.fast.model = "large-v3-turbo"
    s.asr.refine_enabled = False
    (tmp_path / "vad.onnx").write_bytes(b"not the pinned file")
    d = tmp_path / "whisper" / "large-v3-turbo"
    d.mkdir(parents=True)
    (d / "model.bin").write_bytes(b"corrupt")
    fetched = []
    monkeypatch.setattr(models, "_fetch", lambda url, dst, sha: fetched.append((url, sha)))
    with pytest.raises(RuntimeError, match="sha256"):
        models.download(s, verify=True, log=lambda *_: None)
    assert fetched == [(models.VAD["url"], models.VAD["sha256"])]  # bad VAD re-fetched
    assert models.download(s, verify=False, log=lambda *_: None)  # no verify: accepted as is


def test_pins_are_well_formed():
    for name, pin in models.WHISPER.items():
        assert len(pin["revision"]) == 40 and len(pin["sha256"]) == 64, name
        assert "/" in pin["repo"]
    assert "/raw/v" in models.VAD["url"] and len(models.VAD["sha256"]) == 64


def test_sha256_streams_large_files(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"a" * (3 << 20))
    assert models.sha256(p, chunk=1 << 20) == hashlib.sha256(b"a" * (3 << 20)).hexdigest()

