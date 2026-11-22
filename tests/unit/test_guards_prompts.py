from live_stt.asr.backends.base import AsrResult, SegmentResult, WordResult
from live_stt.asr.guards import Guards, max_ngram_count
from live_stt.asr.prompts import build_hotwords, build_prompt, last_sentence
from live_stt.settings import GuardsConfig


def seg(text, lp=-0.3, nsp=0.1, cr=1.2, words=None, start=0.0, end=2.0):
    return SegmentResult(text, start, end, lp, nsp, cr, words or [])


def res(*segs):
    return AsrResult("j", list(segs), "ur", 0.1, "m")


G = Guards(GuardsConfig(), blocklist=["thanks for watching", "چینل کو سبسکرائب کریں"])


def test_keeps_good_segment():
    out = G.apply(res(seg("وزیراعظم نے خطاب کیا")), 3.0)
    assert out.text == "وزیراعظم نے خطاب کیا" and out.drops == []


def test_no_speech_drop_requires_both_conditions():
    assert G.apply(res(seg("کچھ", lp=-1.5, nsp=0.9)), 2).drops == ["no_speech"]
    assert G.apply(res(seg("کچھ", lp=-0.2, nsp=0.9)), 2).drops == []


def test_compression_and_repetition():
    assert G.apply(res(seg("abc", cr=3.0)), 2).drops == ["compression"]
    rep = " ".join(["پاکستان زندہ باد"] * 6)
    assert G.apply(res(seg(rep, cr=1.5)), 2).drops == ["repetition"]
    assert max_ngram_count("a b c a b c a b c".split(), 3) == 3


def test_blocklist_dominant_only():
    assert G.apply(res(seg("Thanks for watching!")), 2).drops == ["blocklist"]
    long = "آج قومی اسمبلی میں بجٹ پیش کیا گیا اور وزیر خزانہ نے کہا thanks for watching"
    assert G.apply(res(seg(long)), 5).drops == []


def test_mixed_segments_flags():
    out = G.apply(res(seg("اچھا متن"), seg("Thanks for watching")), 4)
    assert out.text == "اچھا متن"
    assert "dropped:blocklist" in out.flags


def test_word_hygiene_out_of_range_and_clamp():
    words = [WordResult(" a", -0.1, 0.3, 0.9), WordResult(" b", 0.3, 0.6, 0.9),
             WordResult(" c", 5.0, 5.5, 0.9), WordResult(" d", 1.0, 0.5, 0.9)]
    out = G.apply(res(seg("a b c d", words=words)), 2.0)
    assert [w.word for w in out.words] == [" a", " b"]
    assert out.words[0].start == 0.0
    assert "word_ts_out_of_range" in out.flags


def test_context_trimming_for_refine():
    words = [WordResult(" ctxL", 0.0, 0.4, 0.9), WordResult(" mid", 0.6, 1.0, 0.9),
             WordResult(" ctxR", 2.6, 2.9, 0.9)]
    out = G.apply(res(seg("ctxL mid ctxR", words=words, end=3.0)), 3.0, ctx_pre_s=0.5,
                  ctx_post_s=0.5)
    assert out.text == "mid" and [w.word for w in out.words] == [" mid"]


def test_empty_result():
    out = G.apply(res(), 2)
    assert out.text == "" and out.words == [] and out.avg_logprob is None


def test_prompt_budget_and_glossary():
    gl = ["پاکستان", "اسلام آباد", "وزیراعظم"] * 20
    p = build_prompt(gl, "پہلا جملہ۔ دوسرا جملہ یہاں ہے۔ آخری بات", 200)
    assert len(p) <= 200 and p.startswith("پاکستان") and p.endswith("آخری بات")
    assert build_prompt([], "", 200) is None
    assert build_prompt([], "x" * 1000, 50) and len(build_prompt([], "x" * 1000, 50)) <= 50


def test_last_sentence_and_hotwords():
    assert last_sentence("ایک۔ دو؟ تین", 100) == "تین"
    assert last_sentence("a b c d e f", 5) in ("e f", "d e f"[-5:])
    assert build_hotwords(["a" * 150, "b" * 100], 200) == "a" * 150
    assert build_hotwords([]) is None


def test_shipped_blocklist_catches_measured_nonspeech_labels():
    g = Guards(GuardsConfig())  # loads config/hallucination_blocklist.txt
    for t in ["موسیقی", " موسیقی", "[Music]", "(موسیقی)", "♪ موسیقی ♪", "تالیاں"]:
        assert g.segment_drop_reason(t, -0.3, 0.0, 1.2) == "blocklist", t
    real = "وزیر ثقافت نے کہا کہ پاکستانی موسیقی دنیا بھر میں مقبول ہے"
    assert g.segment_drop_reason(real, -0.3, 0.0, 1.2) is None
