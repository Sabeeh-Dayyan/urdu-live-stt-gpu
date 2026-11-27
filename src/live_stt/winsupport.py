"""Native-Windows support helpers (no-ops on Linux and macOS).

CTranslate2's Windows wheel loads cuBLAS and cuDNN from the DLL search path. Unlike on Linux,
Python on Windows (3.8+) no longer searches PATH for the dependencies of extension modules, and the
``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12`` pip packages put their DLLs in
``site-packages/nvidia/<lib>/bin``, which nothing adds. Without this, ``device: cuda`` fails with
"Could not load library cudnn_ops64_9.dll" even though the libraries are installed.
"""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

_added: list[str] = []


def cuda_dll_dirs(site_dirs: list[str] | None = None, environ: dict | None = None) -> list[Path]:
    """Directories that hold CUDA runtime DLLs: pip ``nvidia/*/bin`` folders first, then the CUDA
    toolkit (``CUDA_PATH``/bin). Only existing directories, in search order, without duplicates."""
    env = os.environ if environ is None else environ
    if site_dirs is None:
        site_dirs = [*site.getsitepackages(), site.getusersitepackages()]
    out: list[Path] = []
    for sp in site_dirs:
        out += sorted(p for p in (Path(sp) / "nvidia").glob("*/bin") if p.is_dir())
    if env.get("CUDA_PATH"):
        out.append(Path(env["CUDA_PATH"]) / "bin")
    seen, uniq = set(), []
    for p in out:
        key = str(p).lower()
        if p.is_dir() and key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def add_cuda_dll_dirs() -> list[str]:
    """Register the CUDA DLL directories with the Windows loader (idempotent). Call before the
    first ``import ctranslate2`` / ``faster_whisper``. Returns the directories added."""
    if sys.platform != "win32" or _added:
        return list(_added)
    for d in cuda_dll_dirs():
        os.add_dll_directory(str(d))  # type: ignore[attr-defined]  # Windows only
        os.environ["PATH"] = f"{d}{os.pathsep}{os.environ.get('PATH', '')}"  # for delay-loaded DLLs
        _added.append(str(d))
    return list(_added)
