"""WER / CER after Urdu normalisation (jiwer 4.0.0: ``jiwer.wer(ref, hyp)``, ``jiwer.cer``)."""

from __future__ import annotations

import jiwer

from live_stt.text.urdu_normalize import normalize


def wer_cer(refs: list[str], hyps: list[str]) -> dict:
    r = " ".join(normalize(x) for x in refs).strip()
    h = " ".join(normalize(x) for x in hyps).strip()
    if not r:
        return {"wer": None, "cer": None, "ref_words": 0, "hyp_words": len(h.split())}
    if not h:
        return {"wer": 1.0, "cer": 1.0, "ref_words": len(r.split()), "hyp_words": 0}
    # CER without spaces: Urdu word segmentation (space/ZWNJ usage) is inconsistent.
    return {
        "wer": jiwer.wer(r, h),
        "cer": jiwer.cer(r.replace(" ", ""), h.replace(" ", "")),
        "ref_words": len(r.split()),
        "hyp_words": len(h.split()),
    }
