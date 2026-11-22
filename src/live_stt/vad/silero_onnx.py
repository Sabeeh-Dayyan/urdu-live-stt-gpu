"""Silero VAD (v6.2 ONNX; v5 interface) via onnxruntime — no PyTorch.

Model I/O (verified on the downloaded silero_vad.onnx with onnxruntime 1.23.2):
  inputs : input float32 [B, 64+512] (64-sample context + 512-sample frame @16 kHz),
           state float32 [2, B, 128], sr int64 scalar
  outputs: output float32 [B, 1] (speech prob), stateN float32 [2, B, 128]
The 64-sample context (tail of the previous input) mirrors the official ``OnnxWrapper``.
"""

from __future__ import annotations

import numpy as np

FRAME = 512
CONTEXT = 64
SR = 16_000


class SileroVAD:
    def __init__(self, model_path: str, threads: int = 1) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SR, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Speech probability for exactly 512 float32 samples in [-1, 1]."""
        x = np.asarray(frame, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != FRAME:
            raise ValueError(f"expected {FRAME} samples, got {x.shape[1]}")
        if not np.all(np.isfinite(x)):  # corrupt decode → treat as silence, keep state sane
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(x, -1.0, 1.0, out=x)
        inp = np.concatenate([self._context, x], axis=1)
        out, state = self.session.run(None, {"input": inp, "state": self._state, "sr": self._sr})
        self._state = state
        self._context = inp[:, -CONTEXT:]
        return float(out[0, 0])


class FrameAccumulator:
    """Splits arbitrary-size chunks into 512-sample frames with absolute start indices."""

    def __init__(self, frame: int = FRAME) -> None:
        self.frame = frame
        self._buf = np.zeros(0, dtype=np.float32)
        self._next_idx: int | None = None  # absolute index of _buf[0]

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._next_idx = None

    def push(self, samples: np.ndarray, first_idx: int):
        """Yield (start_idx, frame) for every complete frame."""
        if self._next_idx is None or self._next_idx + len(self._buf) != first_idx:
            # Gap or first call: drop partial frame, realign to new data.
            self._buf = np.zeros(0, dtype=np.float32)
            self._next_idx = first_idx
        self._buf = np.concatenate([self._buf, np.asarray(samples, dtype=np.float32)])
        n = len(self._buf) // self.frame
        for i in range(n):
            yield self._next_idx + i * self.frame, self._buf[i * self.frame : (i + 1) * self.frame]
        if n:
            self._buf = self._buf[n * self.frame :].copy()
            self._next_idx += n * self.frame
