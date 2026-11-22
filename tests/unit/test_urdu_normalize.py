import pytest

from live_stt.text.urdu_normalize import normalize


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("علي", "علی"),  # Arabic yeh → Urdu yeh
        ("مصطفى", "مصطفی"),  # alef maksura → yeh
        ("كتاب", "کتاب"),  # Arabic kaf → keheh
        ("الله", "اللہ"),  # Arabic heh → heh goal
        ("مدرسۀ", "مدرسہ"),  # heh with yeh above
        ("صلوة", "صلوۃ"),  # teh marbuta
        ("أحمد", "احمد"),
    ],
)
def test_letter_variants(raw, expected):
    assert normalize(raw) == expected


def test_do_chashmi_he_kept_distinct():
    assert normalize("بھائی") == "بھائی"
    assert "ھ" in normalize("کھانا")


def test_diacritics_stripped():
    assert normalize("پاکِستان") == "پاکستان"
    assert normalize("مُحَمَّد") == "محمد"
    assert normalize("اَللّٰہ") == "اللہ"


def test_composed_letters_preserved():
    # آ (alef madda), ئ (yeh hamza), ۓ (yeh barree hamza) must survive
    assert normalize("آج") == "آج"
    assert normalize("آئے") == "آئے"
    assert normalize("گئۓ") == "گئۓ"
    # decomposed alef + madda composes to آ
    assert normalize("آج") == "آج"


def test_digits():
    assert normalize("۲۰۲۴") == "2024"
    assert normalize("٢٠٢٤") == "2024"


def test_punctuation_and_whitespace():
    assert normalize("سلام،   دنیا۔ کیا حال ہے؟") == "سلام دنیا کیا حال ہے"
    assert normalize("سلام، دنیا۔", strip_punct=False) == "سلام, دنیا."
    assert normalize("rate 3.5 percent, don't") == "rate 3.5 percent don't"


def test_invisible_chars_and_tatweel():
    assert normalize("پا‌کستان") == "پاکستان"
    assert normalize("‏سلام‎") == "سلام"
    assert normalize("سـلام") == "سلام"


def test_presentation_forms():
    # ﻛﺘﺎﺏ written with Arabic presentation forms
    assert normalize("ﻛﺘﺎﺏ") == "کتاب"


def test_code_switching_casefold():
    assert normalize("وزیراعظم نے Press Conference کی") == "وزیراعظم نے press conference کی"


def test_empty():
    assert normalize("") == ""
    assert normalize("  ۔ ، ") == ""
