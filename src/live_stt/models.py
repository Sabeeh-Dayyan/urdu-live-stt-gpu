"""Model files: which ones a config needs, pinned sources, download + SHA-256 verification.

Pins were checked on 2026-11-22: the Hugging Face ``X-Linked-ETag`` of each ``model.bin`` equals
the SHA-256 of the files every benchmark in docs/BENCHMARKS.md was measured with.
``stt download-models -c <config>`` fetches only what that config uses.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from live_stt.settings import Settings, resolve_path

WHISPER = {
    # faster-whisper's own alias points at mobiuslabsgmbh/…, which now redirects here.
    "large-v3-turbo": {"repo": "dropbox-dash/faster-whisper-large-v3-turbo",
                       "revision": "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
                       "sha256": "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da",
                       "size_gb": 1.6},
    "large-v3": {"repo": "Systran/faster-whisper-large-v3",
                 "revision": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
                 "sha256": "69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1",
                 "size_gb": 3.1},
}
VAD = {"url": "https://github.com/snakers4/silero-vad/raw/v6.2/src/silero_vad/data/silero_vad.onnx",
       "sha256": "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"}
WHISPER_FILES = ["config.json", "model.bin", "preprocessor_config.json", "tokenizer.json",
                 "vocabulary.json"]


@dataclass
class Item:
    name: str
    path: Path
    present: bool
    pinned: bool


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def required(s: Settings) -> list[Item]:
    """Models the given config will load (VAD + FAST + REFINE if enabled), with presence."""
    vad = resolve_path(s.vad.model_path)
    items = [Item("silero-vad", vad, vad.is_file(), True)]
    names = [s.asr.fast.model]
    if s.asr.refine_enabled and s.asr.refine.model not in names:
        names.append(s.asr.refine.model)
    root = resolve_path(s.asr.download_root)
    for n in names:
        if os.path.isdir(n):  # a local CTranslate2 model directory configured directly
            items.append(Item(n, Path(n), (Path(n) / "model.bin").is_file(), False))
            continue
        d = root / n
        items.append(Item(n, d, (d / "model.bin").is_file(), n in WHISPER))
    return items


def _fetch(url: str, dst: Path, want_sha: str | None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=dst.name + ".")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as r:
            while b := r.read(1 << 20):
                out.write(b)
        got = sha256(Path(tmp))
        if want_sha and got != want_sha:
            raise RuntimeError(f"{url}: sha256 {got} != pinned {want_sha}")
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def download(s: Settings, verify: bool = True, log=print) -> list[Item]:
    """Download missing models; with ``verify`` also check pinned SHA-256 of present ones."""
    out = []
    for it in required(s):
        if it.name == "silero-vad":
            if not it.present or (verify and sha256(it.path) != VAD["sha256"]):
                log(f"downloading Silero VAD -> {it.path}")
                _fetch(VAD["url"], it.path, VAD["sha256"])
        elif not it.pinned:
            if not it.present:
                from faster_whisper import download_model

                log(f"downloading {it.name} (unpinned) -> {it.path}")
                download_model(it.name, output_dir=str(it.path))
        else:
            pin = WHISPER[it.name]
            mb = it.path / "model.bin"
            if not it.present:
                from huggingface_hub import snapshot_download

                log(f"downloading {it.name} ({pin['size_gb']} GB) from {pin['repo']}@"
                    f"{pin['revision'][:8]} -> {it.path}")
                snapshot_download(pin["repo"], revision=pin["revision"], local_dir=str(it.path),
                                  allow_patterns=WHISPER_FILES)
            if verify:
                log(f"verifying {it.name} model.bin sha256")
                got = sha256(mb)
                if got != pin["sha256"]:
                    raise RuntimeError(f"{mb}: sha256 {got} != pinned {pin['sha256']} "
                                       "(delete the directory and download again)")
        out.append(Item(it.name, it.path, True, it.pinned))
    return out
