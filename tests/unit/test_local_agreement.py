import numpy as np

from live_stt.streaming.local_agreement import (
    HypothesisBuffer, LocalAgreementStreamer, TWord, common_prefix,
)

T0 = 1_700_000_000.0


def w(t, s, e=None):
    return TWord(s, e if e is not None else s + 0.3, t)


def test_common_prefix_normalised():
    a = [w("سلام", 0), w("دنیا", 1)]
    b = [w("سلام۔", 0), w("دنيا", 1), w("x", 2)]  # punctuation / Arabic yeh variant
    assert common_prefix(a, b) == 2
    assert common_prefix([], b) == 0


def test_local_agreement_two_hypotheses_commit():
    h = HypothesisBuffer()
    h.insert([w("a", 0), w("b", 1), w("c", 2)])
    assert h.flush() == []
    h.insert([w("a", 0), w("b", 1), w("x", 2), w("d", 3)])
    assert [x.text for x in h.flush()] == ["a", "b"]
    assert [x.text for x in h.uncommitted()] == ["x", "d"]
    h.insert([w("a", 0), w("b", 1), w("x", 2), w("d", 3)])  # re-recognised committed words
    assert [x.text for x in h.flush()] == ["x", "d"]


def test_ngram_overlap_with_committed_tail_removed():
    h = HypothesisBuffer()
    h.committed_in_buffer = [w("a", 0), w("b", 1)]
    h.last_committed_time = 1.3
    h.insert([w("b", 1.25), w("c", 1.6)])  # 'b' starts just before last commit (+-0.1)
    assert [x.text for x in h.new] == ["c"]


def test_streamer_absolute_times_survive_trim():
    s = LocalAgreementStreamer(trim_s=4, max_buffer_s=6)
    s.append(np.zeros(16000 * 5, np.float32), T0)
    hyp = [(0.5, 0.8, "a", 1), (1.5, 1.8, "b", 1), (3.0, 3.3, "c", 1)]
    s.update(hyp)
    r = s.update(hyp)  # agreement → commit a, b, c ; buffer 5 s > trim 4 → trim at c.end
    assert [x.text for x in r.committed] == ["a", "b", "c"]
    assert abs(r.committed[0].start - (T0 + 0.5)) < 1e-9
    assert s.buffer_utc == T0 + 3.3
    assert abs(s.buffer_s - 1.7) < 1e-3
    assert "a b c" in s.context_text
    # next hypothesis is relative to the trimmed buffer
    s.append(np.zeros(16000, np.float32), T0 + 5.0)
    s.update([(0.5, 0.8, "d", 1)])
    r = s.update([(0.5, 0.8, "d", 1)])
    assert [x.text for x in r.committed] == ["d"] and abs(r.committed[0].start - (T0 + 3.8)) < 1e-6


def test_hard_cap_when_nothing_commits():
    s = LocalAgreementStreamer(trim_s=4, max_buffer_s=6)
    s.append(np.zeros(16000 * 7, np.float32), T0)
    s.update([(0.1, 0.2, "x", 1)])
    s.update([(0.1, 0.2, "y", 1)])  # never agrees
    assert s.buffer_s <= 6.0 + 1e-6


def test_end_segment_returns_committed_plus_tail_once():
    s = LocalAgreementStreamer()
    s.append(np.zeros(16000 * 3, np.float32), T0)
    s.update([(0.1, 0.4, "a", 1), (1.0, 1.3, "b", 1)])
    s.update([(0.1, 0.4, "a", 1), (1.0, 1.3, "b", 1), (2.0, 2.3, "c", 1)])
    words = s.end_segment(T0 + 2.5)
    assert [x.text for x in words] == ["a", "b", "c"]
    assert s.end_segment(T0 + 2.5) == []  # nothing duplicated on a second close


def test_gap_handling():
    s = LocalAgreementStreamer()
    s.append(np.zeros(1600, np.float32), T0)
    s.append(np.zeros(1600, np.float32), T0 + 0.1 + 0.5)  # 0.5 s gap → padded
    assert abs(s.buffer_s - 0.7) < 1e-3
    s.append(np.zeros(1600, np.float32), T0 + 100)  # discontinuity → restart
    assert s.buffer_utc == T0 + 100 and abs(s.buffer_s - 0.1) < 1e-6
