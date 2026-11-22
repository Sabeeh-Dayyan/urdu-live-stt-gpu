"""Segment confidence = mean Whisper word probability (0–1).

This is the model's own confidence, **not** measured accuracy. What a confidence threshold means in
accuracy terms is measured by ``bench/calibration.py`` against hand-made references.
"""

from __future__ import annotations

from collections.abc import Iterable


def word_prob(w) -> float | None:
    p = w.get("p") if isinstance(w, dict) else getattr(w, "probability", None)
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    return p if 0.0 <= p <= 1.0 else None


def mean_confidence(words: Iterable) -> float | None:
    ps = [p for p in (word_prob(w) for w in words or []) if p is not None]
    return sum(ps) / len(ps) if ps else None
