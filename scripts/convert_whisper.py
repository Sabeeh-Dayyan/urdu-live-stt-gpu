#!/usr/bin/env python3
"""Convert a Hugging Face Whisper checkpoint (Transformers format) into a CTranslate2 model
directory that faster-whisper / this service loads (``asr.*.model`` = directory name under
``models/whisper/`` or an absolute path).

    python3 scripts/convert_whisper.py <hf_repo> <revision> <output_name> \\
        [--quantization float16] [--sha256 <model.safetensors sha256>] [--inside]

* Downloads the pinned revision (resumable), checks ``model.safetensors`` against ``--sha256``.
* Converts with pinned versions (torch 2.8.0 CPU, ctranslate2 4.9.0, transformers 4.57.1):
  by default in a throwaway ``python:3.11-slim`` Docker container (nothing installed on the host);
  ``--inside`` installs them into the current Python instead — used by the deploy stack's one-shot
  ``convert`` service, which runs in its own python:3.11-slim container.
* ``tokenizer.json`` comes from the repo if present, else from the pinned Systran large-v3 build
  (same vocabulary as every large-v3 / large-v3-turbo fine-tune).
Stdlib only. Needs ~8 GB RAM and ~10 GB free disk for a large-v3-sized model.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TORCH = ("https://download-r2.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl",
         "torch-2.8.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl",
         "cb06175284673a581dd91fb1965662ae4ecaba6e5c357aa0ea7bb8b84b6b7eeb")
PIP = ["ctranslate2==4.9.0", "transformers==4.57.1"]
TOKENIZER_FALLBACK = ("https://huggingface.co/Systran/faster-whisper-large-v3/resolve/"
                      "edaa852ec7e145841d8ffdb056a99866b5f0a478/tokenizer.json")
FILES = ["config.json", "generation_config.json", "preprocessor_config.json",
         "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json",
         "merges.txt", "normalizer.json", "tokenizer.json"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def fetch(url: str, dst: Path, optional: bool = False, tries: int = 40) -> bool:
    """Resumable download (HTTP Range) with retries."""
    for i in range(tries):
        have = dst.stat().st_size if dst.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # server ignored Range: start over
                    have = 0
                with dst.open("ab" if have else "wb") as f:
                    while b := r.read(1 << 20):
                        f.write(b)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 416:  # already complete
                return True
            if e.code in (401, 403, 404) and optional:
                return False
            if e.code in (401, 403, 404):
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            print(f"  retry {i + 1}: {e}", flush=True)
        time.sleep(min(30, 2 + 2 * i))
    raise RuntimeError(f"download failed: {url}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("revision")
    ap.add_argument("name")
    ap.add_argument("--quantization", default="float16")
    ap.add_argument("--sha256", default="")
    ap.add_argument("--inside", action="store_true")
    a = ap.parse_args()
    out = ROOT / "models" / "whisper" / a.name
    if (out / "model.bin").is_file():
        print(f"exists: {out}")
        return
    (ROOT / "models").mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".convert.", dir=ROOT / "models"))
    try:
        hf = work / "hf"
        hf.mkdir()
        base = f"https://huggingface.co/{a.repo}/resolve/{a.revision}"
        for f in FILES:
            fetch(f"{base}/{f}", hf / f, optional=True)
        print(f"downloading {a.repo}@{a.revision[:8]} model.safetensors", flush=True)
        fetch(f"{base}/model.safetensors", hf / "model.safetensors")
        if a.sha256:
            got = sha256(hf / "model.safetensors")
            if got != a.sha256:
                raise SystemExit(f"model.safetensors sha256 {got} != pinned {a.sha256}")
            print("model.safetensors sha256 ok", flush=True)
        if not (hf / "tokenizer.json").is_file():
            fetch(TOKENIZER_FALLBACK, hf / "tokenizer.json")
        whl = work / TORCH[1]
        fetch(TORCH[0], whl)
        if sha256(whl) != TORCH[2]:
            raise SystemExit("torch wheel sha256 mismatch")
        conv = ["ct2-transformers-converter", "--model", "{hf}", "--output_dir", "{out}",
                "--quantization", a.quantization, "--low_cpu_mem_usage",
                "--copy_files", "tokenizer.json", "preprocessor_config.json"]
        if a.inside:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                            "--default-timeout=120", "--retries", "10", str(whl), *PIP], check=True)
            exe = shutil.which("ct2-transformers-converter") or str(
                Path(sys.executable).parent / "ct2-transformers-converter")
            subprocess.run([exe, *[x.format(hf=hf, out=work / "ct2") for x in conv[1:]]],
                           check=True)
        else:
            sh = (f"pip install -q --no-cache-dir --default-timeout=120 --retries 10 /w/{TORCH[1]} "
                  + " ".join(PIP) + " && " + " ".join(x.format(hf="/w/hf", out="/w/ct2") for x in conv))
            subprocess.run(["docker", "run", "--rm", "--platform", "linux/amd64", "-v",
                            f"{work}:/w", "python:3.11-slim", "bash", "-c", sh], check=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        os.replace(work / "ct2", out)
        print(f"converted -> {out}\nmodel.bin sha256: {sha256(out / 'model.bin')}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
