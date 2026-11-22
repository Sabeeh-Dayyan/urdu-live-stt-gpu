"""Urdu text normalisation for search and WER/CER.

Pipeline (order matters):
1. Unicode NFKC — folds Arabic presentation forms (U+FB50–FDFF, U+FE70–FEFF) to base letters.
2. Remove invisible format characters (ZWJ/ZWNJ/ZWSP, bidi marks, BOM) and tatweel.
3. Strip diacritics (harakat zer/zabar/pesh/tanween/shadda/sukun, superscript alef, Quranic marks).
   Hamza-above (U+0654) and maddah (U+0653) are *not* stripped as free marks, because after NFC
   they are composed into letters (ۓ, آ, ئ) that are distinct in Urdu.
4. Unify Arabic → Urdu letter variants: ي ى → ی, ك → ک, ه ە ۀ ۂ → ہ, ة → ۃ, أ إ ٱ → ا.
   Do-chashmi he ھ (aspiration) is intentionally kept distinct from ہ.
5. Digits: Arabic-Indic (٠–٩) and Extended/Urdu (۰–۹) → ASCII 0–9.
6. Punctuation: Urdu/Arabic marks → ASCII; with ``strip_punct`` all punctuation is removed.
7. Latin text casefolded (code-switched English); whitespace collapsed.

``text_norm`` stored in PostgreSQL and used for WER = ``normalize(text, strip_punct=True)``.
"""

from __future__ import annotations

import re
import unicodedata

_FORMAT_CHARS = (
    "​‌‍‎‏"  # ZWSP, ZWNJ, ZWJ, LRM, RLM
    "‪‫‬‭‮"  # bidi embeddings/overrides
    "⁦⁧⁨⁩"  # bidi isolates
    "﻿"  # BOM
    "ـ"  # tatweel
    "؜"  # Arabic letter mark
)

_DIACRITICS = re.compile(
    "["
    "ً-ْ"  # fathatan … sukun (zabar, zer, pesh, tanween, shadda, sukun)
    "ٕ-ٟ"  # hamza below, subscript alef, other marks (not U+0653/U+0654)
    "ٰ"  # superscript alef (khari zabar)
    "ۖ-ۜ۟-۪ۤۧۨ-ۭ"  # Quranic annotation marks
    "࣓-ࣿ"  # extended Arabic marks
    "]"
)

_LETTER_MAP = {
    "ي": "ی",  # ي Arabic yeh → ی Farsi/Urdu yeh
    "ى": "ی",  # ى alef maksura → ی
    "ك": "ک",  # ك Arabic kaf → ک keheh
    "ه": "ہ",  # ه Arabic heh → ہ heh goal
    "ە": "ہ",  # ە ae → ہ
    "ۀ": "ہ",  # ۀ heh with yeh above → ہ (izafat mark dropped for search)
    "ۂ": "ہ",  # ۂ heh goal with hamza above → ہ
    "ة": "ۃ",  # ة teh marbuta → ۃ teh marbuta goal
    "أ": "ا",  # أ → ا
    "إ": "ا",  # إ → ا
    "ٱ": "ا",  # ٱ alef wasla → ا
}

_DIGITS = {chr(0x0660 + i): str(i) for i in range(10)}
_DIGITS.update({chr(0x06F0 + i): str(i) for i in range(10)})

_PUNCT_MAP = {
    "،": ",",  # ، comma
    "؛": ";",  # ؛ semicolon
    "؟": "?",  # ؟ question mark
    "۔": ".",  # ۔ full stop
    "٪": "%",  # ٪ percent
    "٫": ".",  # ٫ decimal separator
    "٬": ",",  # ٬ thousands separator
    "۝": "",  # end of ayah
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "–": "-",
    "—": "-",
    "…": "...",
    "«": '"',
    "»": '"',
}

_TRANS = str.maketrans(
    {**{c: None for c in _FORMAT_CHARS}, **_LETTER_MAP, **_DIGITS, **_PUNCT_MAP}
)


def _strip_punct(t: str) -> str:
    """Replace Unicode punctuation/symbols (categories P*, S*) with spaces.

    Apostrophes and periods between two alphanumerics are kept (``don't``, ``3.5``).
    Combining marks (Mn, e.g. hamza above) are letters' parts and are kept.
    """
    out = []
    n = len(t)
    for i, ch in enumerate(t):
        cat = unicodedata.category(ch)
        if cat[0] in "PS" or ch == "_":
            if ch in "'." and 0 < i < n - 1 and t[i - 1].isalnum() and t[i + 1].isalnum():
                out.append(ch)
            else:
                out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


_WS = re.compile(r"\s+")


def normalize(text: str, strip_punct: bool = True) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_TRANS)
    # NFKC can re-introduce Arabic letters from presentation forms; diacritics removal after.
    t = _DIACRITICS.sub("", t)
    t = unicodedata.normalize("NFC", t)
    t = t.casefold()
    if strip_punct:
        t = _strip_punct(t)
    else:
        t = re.sub(r"\s+([,.;:?!])", r"\1", t)
    return _WS.sub(" ", t).strip()
