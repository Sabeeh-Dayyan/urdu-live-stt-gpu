"""Hallucination guards and word-timestamp hygiene (pure logic).

Per Whisper segment, drop when any of:
* ``no_speech``   : no_speech_prob > no_speech_threshold AND avg_logprob < logprob_threshold
* ``compression`` : compression_ratio > threshold (repetitive text)
* ``repetition``  : most frequent word n-gram occurs > max_ngram_repeats times
* ``blocklist``   : normalised text is dominated (≥ 50 % of characters) by a known boilerplate
                    hallucination phrase ("thanks for watching", "subscribe", …)
Trim (segment kept): a word or 2–3-word phrase repeated ≥ ``min_repeat_run`` times in a row is
cut back to one copy (Whisper's looping tail, e.g. «اسمبلی وزیراعظم وزیراعظم وزیراعظم …»), flag
``repeat_trimmed``.
Words: drop those whose timestamps are outside [0, duration] (± 0.2 s slack, then clamped) or,
for REFINE jobs that carry context, whose midpoint lies inside the context margins (they belong
to the neighbouring segment).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from live_stt.asr.backends.base import AsrResult, WordResult
from live_stt.settings import GuardsConfig, resolve_path
from live_stt.text.urdu_normalize import normalize

WORD_SLACK_S = 0.2


def load_blocklist(path: str | Path | None) -> list[str]:
    if not path:
        return []
    p = resolve_path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            n = normalize(line)
            if n:
                out.append(n)
    return out


def max_ngram_count(tokens: list[str], n: int) -> int:
    if len(tokens) < n:
        return 0
    c = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return max(c.values())


def collapse_runs(keys: list[str], min_run: int = 3, max_n: int = 3) -> list[int]:
    """Indices to keep after cutting any n-gram (n ≤ max_n) that repeats ≥ min_run times in a row
    back to a single copy. ``min_run`` ≤ 1 disables."""
    keep = list(range(len(keys)))
    if min_run <= 1:
        return keep
    changed = True
    while changed:
        changed = False
        k = [keys[i] for i in keep]
        for n in range(1, max_n + 1):
            i = 0
            while i + n * min_run <= len(k):
                unit = k[i:i + n]
                reps = 1
                while k[i + reps * n:i + (reps + 1) * n] == unit:
                    reps += 1
                if reps >= min_run:
                    del keep[i + n:i + reps * n]
                    changed = True
                    break
                i += 1
            if changed:
                break
    return keep


@dataclass
class GuardedOutput:
    text: str
    words: list[WordResult]
    flags: list[str] = field(default_factory=list)
    drops: list[str] = field(default_factory=list)  # reasons for dropped segments
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


class Guards:
    def __init__(self, cfg: GuardsConfig, blocklist: list[str] | None = None) -> None:
        self.cfg = cfg
        self.blocklist = blocklist if blocklist is not None else load_blocklist(cfg.blocklist_path)

    def segment_drop_reason(self, text: str, avg_logprob: float, no_speech_prob: float,
                            compression_ratio: float) -> str | None:
        c = self.cfg
        norm = normalize(text)
        if not norm:
            return "empty"
        if no_speech_prob > c.no_speech_threshold and avg_logprob < c.logprob_threshold:
            return "no_speech"
        if compression_ratio > c.compression_ratio_threshold:
            return "compression"
        toks = norm.split()
        if max_ngram_count(toks, c.ngram_n) > c.max_ngram_repeats or (
            len(toks) >= 2 * (c.max_ngram_repeats + 1)
            and max_ngram_count(toks, 1) > max(c.max_ngram_repeats * 2, len(toks) * 0.6)
        ):
            return "repetition"
        for phrase in self.blocklist:
            if phrase in norm and len(phrase) >= 0.5 * len(norm):
                return "blocklist"
        return None

    def apply(self, res: AsrResult, duration_s: float, ctx_pre_s: float = 0.0,
              ctx_post_s: float = 0.0) -> GuardedOutput:
        kept_text: list[str] = []
        words: list[WordResult] = []
        flags: set[str] = set()
        drops: list[str] = []
        lp, nsp, cr = [], [], []
        lo, hi = ctx_pre_s, duration_s - ctx_post_s
        has_ctx = ctx_pre_s > 0 or ctx_post_s > 0
        for seg in res.segments:
            reason = self.segment_drop_reason(seg.text, seg.avg_logprob, seg.no_speech_prob,
                                              seg.compression_ratio)
            if reason:
                if reason != "empty":
                    drops.append(reason)
                continue
            seg_words = []
            for w in seg.words:
                if not w.word.strip():
                    continue
                if w.end < -WORD_SLACK_S or w.start > duration_s + WORD_SLACK_S or w.end < w.start:
                    flags.add("word_ts_out_of_range")
                    continue
                s, e = max(0.0, w.start), min(duration_s, w.end)
                if has_ctx:
                    mid = (s + e) / 2
                    if mid < lo or mid > hi:
                        continue  # belongs to the neighbouring segment's context
                seg_words.append(WordResult(w.word, s, e, w.probability))
            run = getattr(self.cfg, "min_repeat_run", 3)
            if seg_words:
                keep = collapse_runs([normalize(w.word) for w in seg_words], run)
                if len(keep) < len(seg_words):
                    seg_words = [seg_words[i] for i in keep]
                    flags.add("repeat_trimmed")
                    trimmed = True
                else:
                    trimmed = False
            else:
                trimmed = False
            if (has_ctx or trimmed) and seg.words:
                if not seg_words:
                    continue
                text = "".join(w.word for w in seg_words).strip()
            else:
                toks = seg.text.split()
                keep = collapse_runs([normalize(t) for t in toks], run)
                if len(keep) < len(toks):
                    flags.add("repeat_trimmed")
                    text = " ".join(toks[i] for i in keep)
                else:
                    text = seg.text.strip()
            kept_text.append(text)
            words += seg_words
            lp.append(seg.avg_logprob)
            nsp.append(seg.no_speech_prob)
            cr.append(seg.compression_ratio)
        for d in drops:
            flags.add(f"dropped:{d}")
        return GuardedOutput(
            text=" ".join(t for t in kept_text if t),
            words=words,
            flags=sorted(flags),
            drops=drops,
            avg_logprob=(sum(lp) / len(lp)) if lp else None,
            no_speech_prob=max(nsp) if nsp else None,
            compression_ratio=max(cr) if cr else None,
        )
