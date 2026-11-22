"""Audio helpers for benchmarking (PyAV decode → 16 kHz mono float32; stdlib WAV write)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 16_000


def load_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    import av

    out = []
    with av.open(str(path)) as c:
        stream = c.streams.audio[0]
        rs = av.AudioResampler(format="flt", layout="mono", rate=sr)
        for frame in c.decode(stream):
            for o in rs.resample(frame):
                out.append(o.to_ndarray().reshape(-1))
        for o in rs.resample(None):
            out.append(o.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


def write_wav(path: str | Path, audio: np.ndarray, sr: int = SR) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
