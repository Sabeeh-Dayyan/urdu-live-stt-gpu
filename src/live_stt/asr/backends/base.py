"""ASR backend interface. Workers never import an engine; only the ASR service does."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DecodeOptions:
    model: str
    beam_size: int = 1
    best_of: int = 1
    temperature: float = 0.0
    language: str | None = "ur"
    initial_prompt: str | None = None
    hotwords: str | None = None
    word_timestamps: bool = True


@dataclass
class AudioJob:
    job_id: str
    audio: np.ndarray  # float32 mono 16 kHz
    language: str | None = "ur"
    initial_prompt: str | None = None
    hotwords: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return len(self.audio) / 16_000


@dataclass
class WordResult:
    word: str
    start: float  # seconds relative to job audio start
    end: float
    probability: float


@dataclass
class SegmentResult:
    text: str
    start: float
    end: float
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    words: list[WordResult] = field(default_factory=list)


@dataclass
class AsrResult:
    job_id: str
    segments: list[SegmentResult]
    language: str | None
    infer_s: float  # wall time attributed to this job (batch time / batch size)
    model: str

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())


class ASRBackend(ABC):
    name: str = "base"

    @abstractmethod
    def transcribe(self, batch: list[AudioJob], opts: DecodeOptions) -> list[AsrResult]:
        """Transcribe each job independently. Must return one result per job, same order."""

    @property
    def device(self) -> str:
        return "cpu"
