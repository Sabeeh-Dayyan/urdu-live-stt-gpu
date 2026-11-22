"""Sentence timestamps, export formats, confidence, relay timestamp re-basing, accuracy rows."""

import csv
import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bench.calibration import mark_correct, segment_table, word_table
from live_stt.asr.backends.base import WordResult
from live_stt.ingest.relay import JUMP_BACK_S, TimestampRebaser
from live_stt.store.accuracy import rows_from_result
from live_stt.text.confidence import mean_confidence
from live_stt.text.sentences import assemble, get_tz, keep, render, split_sentences

T = 1_795_000_000.0


def W(w, s, e, p=0.95):
    return {"w": w, "s": T + s, "e": T + e, "p": p}


SEG = {"channel_id": "ch1", "segment_id": "ch1:1", "pass": "fast", "start_utc": T,
       "end_utc": T + 9, "text_raw": "x",
       "words": [W("وزیراعظم", 0.0, 0.6), W("نے", 0.6, 0.8, 0.5), W("خطاب", 0.8, 1.2),
                 W("کیا۔", 1.2, 1.6), W("اسمبلی", 2.0, 2.5, 0.99), W("کا", 2.5, 2.7, 0.97),
                 W("اجلاس؟", 2.7, 3.3, 0.98), W("آج", 5.0, 5.3), W("ہوا", 5.3, 5.6)]}


# ------------------------------------------------------------------ confidence
def test_mean_confidence_handles_dicts_objects_and_garbage():
    assert mean_confidence([{"p": 0.5}, {"p": 1.0}]) == pytest.approx(0.75)
    assert mean_confidence([WordResult("a", 0, 1, 0.2), WordResult("b", 1, 2, 0.4)]) == \
        pytest.approx(0.3)
    assert mean_confidence([{"p": None}, {"p": "x"}, {"p": 1.5}, {"p": -1}, {}]) is None
    assert mean_confidence([]) is None and mean_confidence(None) is None


# ------------------------------------------------------------------- sentences
def test_split_on_urdu_and_latin_punctuation_and_pause():
    s = split_sentences(SEG)
    assert [x["text"] for x in s] == ["وزیراعظم نے خطاب کیا۔", "اسمبلی کا اجلاس؟", "آج ہوا"]
    assert s[0]["start_utc"] == T and s[0]["end_utc"] == T + 1.6
    assert s[1]["start_utc"] == T + 2.0 and s[2]["start_utc"] == T + 5.0
    assert s[0]["confidence"] == pytest.approx((0.95 + 0.5 + 0.95 + 0.95) / 4)
    assert all(x["segment_id"] == "ch1:1" and x["pass"] == "fast" for x in s)


def test_pause_split_without_punctuation():
    seg = {"words": [W("ایک", 0, 0.5), W("دو", 0.6, 1.0), W("تین", 2.2, 2.6)]}
    assert [x["text"] for x in split_sentences(seg)] == ["ایک دو", "تین"]  # 1.2 s gap
    assert len(split_sentences(seg, pause_s=5)) == 1


def test_segment_without_words_and_empty_segment():
    s = split_sentences({"text_raw": " متن ", "start_utc": T, "end_utc": T + 3, "words": []})
    assert s == [{"channel_id": None, "segment_id": None, "pass": None, "start_utc": T,
                  "end_utc": T + 3, "text": "متن", "confidence": None}]
    assert split_sentences({"text_raw": "  ", "words": None}) == []
    assert split_sentences({"words": [{"w": "  ", "s": T, "e": T}]}) == []


def test_words_missing_timestamps_fall_back_to_segment_times():
    s = split_sentences({"start_utc": T, "end_utc": T + 4,
                         "words": [{"w": "a"}, {"w": "b", "s": None, "e": None, "p": 0.9}]})
    assert s[0]["start_utc"] == T and s[0]["end_utc"] == T + 4 and s[0]["text"] == "a b"


@given(st.lists(st.tuples(st.text(min_size=1, max_size=6), st.floats(0, 0.5),
                          st.floats(0, 3), st.floats(0, 1)), max_size=30))
def test_split_never_loses_or_reorders_words(spec):
    t, words = T, []
    for w, dur, gap, p in spec:
        words.append({"w": w, "s": t, "e": t + dur, "p": p})
        t += dur + gap
    sents = split_sentences({"words": words})
    joined = " ".join(x["text"] for x in sents).split()
    expect = " ".join(w["w"].strip() for w in words if w["w"].strip()).split()
    assert joined == expect
    starts = [x["start_utc"] for x in sents]
    assert starts == sorted(starts)


def test_keep_threshold():
    assert keep({"confidence": None}, 0) and not keep({"confidence": None}, 0.9)
    assert keep({"confidence": 0.9}, 0.9) and not keep({"confidence": 0.89}, 0.9)


# ---------------------------------------------------------------------- export
def test_render_srt_vtt_relative_times():
    s = split_sentences(SEG)
    srt = render(s, "srt")
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,600\nوزیراعظم نے خطاب کیا۔\n")
    assert "3\n00:00:05,000 --> 00:00:05,600\nآج ہوا" in srt
    vtt = render(s, "vtt", origin=T - 60)
    assert vtt.startswith("WEBVTT\n\nNOTE cue times relative to ")
    assert "00:01:02.000 --> 00:01:03.300" in vtt


def test_render_csv_bom_quoting_tz_and_json():
    s = split_sentences(SEG)
    s[0]["text"] = 'متن, "quoted"\nline'
    out = render(s, "csv", get_tz("Asia/Karachi"))
    assert out.startswith("﻿")
    rows = list(csv.reader(io.StringIO(out[1:])))
    assert rows[0][:4] == ["channel_id", "segment_id", "pass", "start"]
    assert rows[1][-1] == 'متن, "quoted"\nline' and rows[1][3].endswith("+05:00")
    assert float(rows[1][5]) == pytest.approx(T)
    j = json.loads(render(s, "json"))
    assert j[1]["start"].endswith("+00:00") and j[1]["text"] == "اسمبلی کا اجلاس؟"
    txt = render(s, "txt")
    assert txt.count("\n") >= 3 and txt.startswith("[")


def test_render_errors_and_empty():
    with pytest.raises(ValueError):
        render([], "docx")
    with pytest.raises(ValueError):
        get_tz("Mars/Olympus")
    assert render([], "srt") == "" and json.loads(render([], "json")) == []
    assert render([{"text": "x", "start_utc": None, "end_utc": None}], "srt") == ""


def test_zero_length_cue_gets_minimum_duration():
    srt = render([{"text": "x", "start_utc": T, "end_utc": T}], "srt")
    assert "00:00:00,000 --> 00:00:00,001" in srt


def _run_js_assemble(rows):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    html = (Path(__file__).parents[2] / "ui" / "index.html").read_text(encoding="utf-8")
    js = html.split("<script>")[1].split("</script>")[0]
    body = js[js.index("function wordOf("):js.index("const confClass")]
    prog = ("const END_CHARS = \"۔؟?!.…\", PAUSE_S = 1.0, MAX_SENT_S = 20.0;\n" + body +
            "\nconst rows = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
            "\nprocess.stdout.write(JSON.stringify(assemble(rows).map(s => [s.s, s.e, s.text, s.c, s.pass, s.segs])));")
    out = subprocess.run([node, "-e", prog], input=json.dumps(rows), capture_output=True,
                         text=True, timeout=30, check=True).stdout
    return json.loads(out)


def test_js_assemble_matches_python():
    rows = _cut_rows()
    py = [[x["start_utc"], x["end_utc"], x["text"], x["confidence"], x["pass"], x["segment_ids"]]
          for x in assemble(rows)]
    js_res = _run_js_assemble(rows)
    assert [r[:3] + r[4:] for r in js_res] == [r[:3] + r[4:] for r in py]
    assert [r[3] for r in js_res] == pytest.approx([r[3] for r in py])
    one = [SEG]
    assert [r[2] for r in _run_js_assemble(one)] == [x["text"] for x in assemble(one)]


# ----------------------------------------------------------------- HLS rebaser
def test_rebaser_monotonic_across_pts_reset():
    r = TimestampRebaser()
    outs = []
    for t in [10.0, 10.04, 10.08, 0.0, 0.04, 0.08]:  # PTS reset (looped file / encoder restart)
        off = r.map(0, t)
        outs.append(t + off)
    assert outs == sorted(outs) and len(set(outs)) == len(outs)
    assert outs[3] == pytest.approx(10.08 + 0.04)


def test_rebaser_forward_jump_and_glitch_drop_and_two_streams():
    r = TimestampRebaser()
    assert r.map(0, 5.0) == 0.0
    assert r.map(1, 4.99) == 0.0  # other stream slightly behind: fine
    assert r.map(0, 4.99) is None  # same stream going backwards a little: dropped
    off = r.map(0, 500.0)  # forward jump > 30 s
    assert 500.0 + off == pytest.approx(5.0 + 0.04)
    # continuing after the jump keeps the new offset (no second re-base for normal progress)
    assert r.map(0, 500.04) == pytest.approx(off)
    # a backward jump smaller than JUMP_BACK_S on another stream is passed through, not re-based
    assert r.map(2, 500.0 - JUMP_BACK_S / 2) == pytest.approx(off)


# -------------------------------------------------------- accuracy / calibration
def test_rows_from_bench_and_calibration_and_unknown():
    bench = {"dataset": {"name": "fleurs_ur"},
             "runs": [{"label": "turbo b1", "wer": 0.25, "cer": 0.1, "rtf": 0.9,
                       "wer_noguard": 0.26, "halluc_rate": 0.0, "ref_words": 700}]}
    rows = rows_from_result("r.json", bench)
    assert ("r.json", "fleurs_ur", "bench", "turbo b1", "wer", None, 0.25, 700) in rows
    assert len(rows) == 5
    cal = {"dataset": "fleurs_ur", "model": "large-v3-turbo", "compute_type": "int8",
           "beam_size": 1,
           "word_level": [{"threshold": 0.9, "precision": 0.88, "precision_below": 0.41,
                           "kept_share": 0.86, "words_kept": 1103}],
           "segment_level": [{"threshold": 0.9, "accuracy": 0.8, "wer": 0.2, "kept_share": 1.0,
                              "segments_kept": 40}]}
    rows = rows_from_result("c.json", cal)
    assert ("c.json", "fleurs_ur", "calibration_word", "large-v3-turbo int8 b1", "precision",
            0.9, 0.88, 1103) in rows
    assert len(rows) == 6
    assert rows_from_result("x.json", {"foo": 1}) == []


def test_calibration_marking_and_tables():
    toks = [("یہ", 0.99), ("ایک", 0.4), ("غلط", 0.95), ("ٹیسٹ", 0.97)]
    ok = mark_correct("یہ ایک ٹیسٹ", toks)
    assert ok == [True, True, False, True]
    assert mark_correct("", toks) == [False] * 4 and mark_correct("x", []) == []
    items = [{"p": [p for _, p in toks], "correct": ok, "conf": 0.83,
              "ref": "یہ ایک ٹیسٹ", "hyp": "یہ ایک غلط ٹیسٹ"}]
    wt = {r["threshold"]: r for r in word_table(items, (0.0, 0.9))}
    assert wt[0.0]["precision"] == 0.75 and wt[0.9]["words_kept"] == 3
    assert wt[0.9]["precision"] == pytest.approx(2 / 3) and wt[0.9]["precision_below"] == 1.0
    sg = {r["threshold"]: r for r in segment_table(items, (0.0, 0.9))}
    assert sg[0.0]["wer"] == pytest.approx(1 / 3) and sg[0.9]["segments_kept"] == 0
    assert sg[0.9]["wer"] is None


def test_persister_row_carries_confidence():
    from live_stt.store.persister import segment_row

    ev = {"type": "final", "channel_id": "ch1", "segment_id": "ch1:1795000000000",
          "start_utc": T, "end_utc": T + 2, "text_raw": "a b",
          "words": [{"w": "a", "s": T, "e": T + 1, "p": 0.8}, {"w": "b", "s": T + 1, "e": T + 2, "p": 1.0}]}
    assert segment_row(ev)[-1] == pytest.approx(0.9)
    assert segment_row(dict(ev, words=[]))[-1] is None


# ------------------------------------------------- sentences across segment cuts
def _cut_rows():
    """A sentence force-cut across two segments (no gap), then a real pause, then a third."""
    a = {"channel_id": "ch1", "segment_id": "ch1:a", "pass": "refined", "start_utc": T,
         "end_utc": T + 2, "latency_s": 12.0,
         "words": [W("حکومت", 0.0, 0.5), W("اپنے", 0.5, 1.0), W("اتحادی", 1.0, 2.0)]}
    b = {"channel_id": "ch1", "segment_id": "ch1:b", "pass": "fast", "start_utc": T + 2.1,
         "end_utc": T + 4, "latency_s": 30.0,
         "words": [W("کو", 2.1, 2.4), W("قائل", 2.4, 3.0), W("کرے", 3.0, 3.5, 0.5)]}
    c = {"channel_id": "ch1", "segment_id": "ch1:c", "pass": "fast", "start_utc": T + 5,
         "end_utc": T + 6, "words": [W("اگلی", 5.0, 5.5), W("خبر", 5.5, 6.0)]}
    return [b, c, a]  # unsorted on purpose


def test_assemble_joins_cut_segments_and_splits_at_pause():
    s = assemble(_cut_rows())
    assert [x["text"] for x in s] == ["حکومت اپنے اتحادی کو قائل کرے", "اگلی خبر"]
    assert s[0]["segment_ids"] == ["ch1:a", "ch1:b"] and s[0]["pass"] == "mixed"
    assert s[0]["start_utc"] == T and s[0]["end_utc"] == T + 3.5 and s[0]["latency_s"] == 30.0
    assert s[1]["pass"] == "fast" and s[1]["segment_ids"] == ["ch1:c"]


def test_assemble_long_unpunctuated_speech_splits_at_longest_pause():
    words, t = [], 0.0
    for i in range(60):  # 60 words × 0.5 s = 30 s of continuous speech
        g = 0.8 if i == 25 else 0.05  # one clearly longest pause (< PAUSE_S)
        words.append(W(f"w{i}", t, t + 0.45))
        t += 0.45 + g
    s = assemble([{"words": words, "start_utc": T, "end_utc": T + t}])
    assert len(s) >= 2 and s[0]["text"].split()[-1] == "w25"
    assert all(x["end_utc"] - x["start_utc"] <= 20.0 + 0.5 for x in s)
    assert " ".join(x["text"] for x in s).split() == [f"w{i}" for i in range(60)]


def test_assemble_rows_without_words_and_empty():
    s = assemble([{"segment_id": "x", "text_raw": "صرف متن", "start_utc": T, "end_utc": T + 2}])
    assert s[0]["text"] == "صرف متن" and s[0]["confidence"] is None
    assert assemble([]) == [] and assemble([{"words": [], "text_raw": " "}]) == []


@given(st.lists(st.tuples(st.integers(1, 8), st.floats(0, 2.5)), min_size=1, max_size=12),
       st.floats(0, 1))
def test_assemble_never_loses_reorders_or_overlaps(segs, p):
    rows, t, n = [], T, 0
    for k, (nw, seg_gap) in enumerate(segs):
        ws = []
        for _ in range(nw):
            ws.append({"w": f"w{n}", "s": t, "e": t + 0.3, "p": p})
            t += 0.35
            n += 1
        rows.append({"segment_id": f"s{k}", "start_utc": ws[0]["s"], "end_utc": ws[-1]["e"],
                     "words": ws})
        t += seg_gap
    s = assemble(rows)
    assert " ".join(x["text"] for x in s).split() == [f"w{i}" for i in range(n)]
    assert all(a["end_utc"] <= b["start_utc"] for a, b in zip(s, s[1:]))
    assert all(x["end_utc"] - x["start_utc"] <= 20.0 + 0.35 for x in s)


def test_time_zones_work_without_system_tz_database():
    """CUDA/slim base images ship no /usr/share/zoneinfo; the tzdata package must cover it."""
    import zoneinfo

    zoneinfo.reset_tzpath(to=[])  # behave like an image without a system tz database
    try:
        zoneinfo.ZoneInfo.clear_cache()
        out = render([{"text": "x", "start_utc": T, "end_utc": T + 1}], "txt", get_tz("Asia/Karachi"))
        assert "+05:00" in out
    finally:
        zoneinfo.reset_tzpath()
        zoneinfo.ZoneInfo.clear_cache()
