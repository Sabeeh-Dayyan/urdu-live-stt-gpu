"""Per-channel decoding prompts: glossary + last committed sentence (pure logic)."""

from __future__ import annotations

import re
from pathlib import Path

from live_stt.settings import resolve_path

_SENT_END = re.compile(r"[.!?۔؟]\s*")


def load_glossary(path: str | Path | None) -> list[str]:
    if not path:
        return []
    p = resolve_path(path)
    if not p.exists():
        return []
    terms, seen = [], set()
    for line in p.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if t and not t.startswith("#") and t not in seen:
            seen.add(t)
            terms.append(t)
    return terms


def last_sentence(text: str, max_chars: int) -> str:
    t = " ".join((text or "").split())
    if not t:
        return ""
    parts = [p for p in _SENT_END.split(t) if p.strip()]
    s = parts[-1].strip() if parts else t
    if len(s) > max_chars:
        s = s[-max_chars:]
        sp = s.find(" ")
        if 0 <= sp < len(s) - 1:
            s = s[sp + 1 :]  # do not start mid-word
    return s


def build_prompt(glossary: list[str], previous_text: str, max_chars: int = 200) -> str | None:
    """Glossary terms first (as many as fit in half the budget), then the last sentence."""
    budget = max(max_chars, 0)
    gl = ""
    for term in glossary:
        cand = f"{gl}، {term}" if gl else term
        if len(cand) > budget // 2:
            break
        gl = cand
    prev = last_sentence(previous_text, budget - len(gl) - 1) if budget - len(gl) > 10 else ""
    out = " ".join(x for x in (gl, prev) if x).strip()
    return out[:budget] or None


def build_hotwords(glossary: list[str], max_chars: int = 200) -> str | None:
    out = ""
    for term in glossary:
        cand = f"{out} {term}".strip()
        if len(cand) > max_chars:
            break
        out = cand
    return out or None
