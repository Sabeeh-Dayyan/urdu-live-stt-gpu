from types import SimpleNamespace

import numpy as np
import pytest

from live_stt.asr.backends.faster_whisper import _sanitize, clip_index
from live_stt.ingest.ringbuffer import RingBuffer
from live_stt.ingest.source import NoAudioStream, build_url, select_audio_stream
from live_stt.settings import AudioTrack, IngestConfig, load_settings
from live_stt.store.persister import _segment_start, segment_row
from live_stt.streaming.segment import make_segment_id
from live_stt.vad.silero_onnx import FrameAccumulator


def test_build_url():
    u = build_url("udp://@239.1.1.1:5000", IngestConfig(localaddr="10.0.0.2"))
    assert u.startswith("udp://@239.1.1.1:5000?") and "fifo_size=1000000" in u
    assert "overrun_nonfatal=1" in u and "localaddr=10.0.0.2" in u
    assert "fifo_size=5" in build_url("udp://h:1?fifo_size=5", IngestConfig())
    assert build_url("/tmp/a.ts", IngestConfig()) == "/tmp/a.ts"
    assert build_url("srt://h:1", IngestConfig()) == "srt://h:1"


def _s(i, pid, lang):
    return SimpleNamespace(type="audio", id=pid, language=lang, metadata={"language": lang})


def test_select_audio_stream():
    streams = [SimpleNamespace(type="video", id=256), _s(0, 257, "eng"), _s(1, 258, "urd")]
    assert select_audio_stream(streams, AudioTrack()).id == 257
    assert select_audio_stream(streams, AudioTrack(language="urd")).id == 258
    assert select_audio_stream(streams, AudioTrack(pid=258)).id == 258
    assert select_audio_stream(streams, AudioTrack(language="fra", index=1)).id == 258
    with pytest.raises(NoAudioStream):
        select_audio_stream(streams, AudioTrack(pid=999))
    with pytest.raises(NoAudioStream):
        select_audio_stream(streams[:1], AudioTrack())
    with pytest.raises(NoAudioStream):
        select_audio_stream(streams, AudioTrack(index=5))


def test_clip_index_and_sanitize():
    starts = [0.0, 5.1, 9.3]
    assert clip_index(starts, 0.0) == 0 and clip_index(starts, 5.0) == 0
    assert clip_index(starts, 5.1) == 1 and clip_index(starts, 100) == 2
    assert clip_index(starts, -1) == 0
    a = _sanitize(np.array([np.nan, 2.0, -3.0], np.float32))
    assert len(a) == 1600 and np.all(np.isfinite(a)) and a.max() <= 1.0


def test_frame_accumulator_gap_realigns():
    acc = FrameAccumulator(4)
    out = list(acc.push(np.arange(6, dtype=np.float32), 0))
    assert [i for i, _ in out] == [0]
    out = list(acc.push(np.arange(6, dtype=np.float32), 6))
    assert [i for i, _ in out] == [4, 8]
    out = list(acc.push(np.arange(4, dtype=np.float32), 100))  # gap
    assert [i for i, _ in out] == [100]


def test_ring_shift_from_index_only_affects_new_session():
    rb = RingBuffer(10, 16000)
    rb.write(np.zeros(16000, np.float32), 1000.0)
    rb.write(np.zeros(16000, np.float32), 2000.0)
    rb.shift_utc(-0.5, from_idx=16000)
    assert rb.index_to_utc(0) == 1000.0
    assert rb.index_to_utc(16000) == pytest.approx(1999.5)
    rb2 = RingBuffer(10, 16000)
    rb2.write(np.zeros(32000, np.float32), 1000.0)
    rb2.shift_utc(-0.5, from_idx=16000)  # split inside one anchor span
    assert rb2.index_to_utc(8000) == pytest.approx(1000.5)
    assert rb2.index_to_utc(16000) == pytest.approx(1000.5)


def test_segment_row_mapping_and_key_stability():
    sid = make_segment_id("ch1", 1700000000.1234)
    assert sid == "ch1:1700000000123"
    fast = {"type": "final", "channel_id": "ch1", "segment_id": sid, "start_utc": 1700000000.1234,
            "end_utc": 1700000003.0, "text_raw": "سلام", "text_norm": "سلام", "words": [],
            "flags": [], "model": "m", "timings": {"latency_s": 1.2}}
    refined = dict(fast, type="refined", start_utc=1700000000.6, text_raw="سلام علیکم")
    rf, rr = segment_row(fast), segment_row(refined)
    assert rf[2] == rr[2], "refined must upsert the same row key as fast"
    assert rf[4] == "fast" and rr[4] == "refined" and rf[12] == 1.2
    assert segment_row(dict(fast, text_raw="  ")) is None
    assert segment_row(dict(refined, text_raw="")) is not None  # refine may blank a hallucination
    assert segment_row(dict(fast, type="partial")) is None
    assert segment_row({"type": "final", "text_raw": "x"}) is None
    assert _segment_start("garbage", 5.0) == 5.0


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("STT_ASR__BEAM_SIZE_NOT_A_FIELD", "1")
    with pytest.raises(Exception):
        load_settings()
    monkeypatch.delenv("STT_ASR__BEAM_SIZE_NOT_A_FIELD")
    monkeypatch.setenv("STT_ASR__CPU_THREADS", "16")
    assert load_settings("config/dev-mac-cpu.yaml").asr.cpu_threads == 16


def test_split_packed_assigns_words_by_midpoint():
    from types import SimpleNamespace as NS

    from live_stt.asr.backends.base import AudioJob
    from live_stt.asr.backends.faster_whisper import split_packed

    jobs = [AudioJob("a", np.zeros(16000 * 2, np.float32)),
            AudioJob("b", np.zeros(16000 * 3, np.float32)),
            AudioJob("c", np.zeros(16000 * 1, np.float32))]
    spans = [(0.0, 2.0), (2.3, 5.3), (5.6, 6.6)]
    W = lambda t, s, e: NS(word=t, start=s, end=e, probability=0.9)  # noqa: E731
    segs = [NS(avg_logprob=-0.2, no_speech_prob=0.1,
               words=[W(" a1", 0.1, 0.5), W(" a2", 1.7, 2.1),  # a2 midpoint 1.9 → job a
                      W(" b1", 2.2, 2.6),  # midpoint 2.4 → job b (starts in the gap)
                      W(" b2", 4.9, 5.4), W(" gap", 5.4, 5.5),  # gap word mid 5.45 → job b edge
                      W(" c1", 5.7, 6.0)])]
    out = split_packed(jobs, spans, segs, 3.0, "ur", "m")
    assert [r.text for r in out] == ["a1 a2", "b1 b2 gap", "c1"]
    b = out[1].segments[0]
    assert b.words[0].start == 0.0 and b.words[-1].end <= 3.0  # clamped, job-relative
    assert abs(sum(r.infer_s for r in out) - 3.0) < 1e-9
    empty = split_packed(jobs, spans, [], 1.0, "ur", "m")
    assert all(r.segments == [] for r in empty)


def test_profiles_sharing_a_redis_server_do_not_share_live_pubsub():
    """Redis pub/sub ignores the DB number: two profiles on one Redis server that use the same
    `pubsub_live` see each other's live captions (happened with dev + broadcast, 2026-11-21)."""
    from itertools import combinations
    from urllib.parse import urlsplit

    from live_stt.settings import PROJECT_ROOT, load_settings

    profiles = {}
    for f in sorted((PROJECT_ROOT / "config").glob("*.yaml")):
        if f.name == "default.yaml":
            continue
        s = load_settings(f)
        u = urlsplit(s.bus.redis_url)
        profiles[f.name] = ((u.hostname, u.port), s.bus.redis_url, s.bus.pubsub_live, s.db.dsn)
    assert "dev-mac-cpu-broadcast.yaml" in profiles
    for (a, pa), (b, pb) in combinations(profiles.items(), 2):
        if pa[0] == pb[0] and (pa[1] != pb[1] or pa[3] != pb[3]):  # same server, meant to be separate
            assert pa[2] != pb[2], f"{a} and {b} share pubsub channel {pa[2]!r}"
