"""Worker pipeline + ASR service end-to-end with the fake ASR and the in-memory bus."""

import time

import numpy as np
import pytest

from live_stt.asr.service import AsrService
from live_stt.bus.base import InMemoryBus
from live_stt.ingest.source import AudioChunk
from live_stt.settings import ChannelConfig, Settings
from live_stt.worker import ChannelPipeline
from tests.fakes import FakeASR, word_audio

T0 = 1_700_000_000.0
SR = 16_000


class EnergyVAD:
    def __call__(self, f):
        return 0.95 if float(np.sqrt(np.mean(f * f))) > 0.05 else 0.02

    def reset(self):
        pass


def make(mode="segment", refine=True, **kw):
    s = Settings()
    s.channels = [ChannelConfig(id="ch1", url="udp://x:1"), ChannelConfig(id="ch2", url="udp://x:2")]
    s.streaming.mode = mode
    s.asr.refine_enabled = refine
    s.vad.min_silence_ms = 300
    for k, v in kw.items():
        sec, key = k.split("__")
        setattr(getattr(s, sec), key, v)
    return s


def programme(word_groups, pause_s=1.0):
    """Utterances of tone-words separated by pauses; returns audio and the id sequence."""
    parts, ids = [np.zeros(int(0.5 * SR), np.float32)], []
    for g in word_groups:
        parts.append(word_audio(g))
        parts.append(np.zeros(int(pause_s * SR), np.float32))
        ids += g
    return np.concatenate(parts), ids


def feed(pipe, audio, n=1600, utc0=T0):
    for i in range(0, len(audio), n):
        pipe.on_chunk(AudioChunk(audio[i:i + n], utc0 + i / SR), True)


def drain(svc, max_steps=10_000):
    idle = 0
    for _ in range(max_steps):
        # pull everything available from the bus like the reader thread would
        for p, stream in svc.streams.items():
            while svc.sched.space(p):
                got = svc.bus.consume(stream, "asr", svc.consumer, count=32)
                if not got:
                    break
                for mid, msg in got:
                    svc.sched.push(svc._job(mid, p, msg))
        if svc.step() == 0:
            idle += 1
            if idle > 3:
                return
        else:
            idle = 0


def events(bus, s, t=None):
    ev = bus.messages(s.bus.stream_transcripts)
    return [e for e in ev if t is None or e["type"] == t]


def ids_of(text):
    return [int(x[1:]) for x in text.split()]


def test_segment_mode_final_then_refined_exact_words():
    s = make()
    bus, asr = InMemoryBus(), FakeASR()
    pipe = ChannelPipeline(s, s.channels[0], bus, EnergyVAD())
    audio, ids = programme([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    feed(pipe, audio)
    svc = AsrService(s, bus, asr, consumer="t")
    drain(svc)
    finals, refined = events(bus, s, "final"), events(bus, s, "refined")
    assert [ids_of(e["text_raw"]) for e in finals] == [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    assert [ids_of(e["text_raw"]) for e in refined] == [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    for f, r in zip(finals, refined):
        assert f["segment_id"] == r["segment_id"]
        assert r["start_utc"] == pytest.approx(f["start_utc"], abs=1e-3)
        for wf, wr in zip(f["words"], r["words"]):
            assert abs(wf["s"] - wr["s"]) < 0.02  # context does not shift word UTC
    # word UTC matches where the tone was placed: first word starts at T0 + 0.5
    assert finals[0]["words"][0]["s"] == pytest.approx(T0 + 0.5, abs=0.02)
    # FAST strictly before REFINE in processing order
    order = [e["type"] for e in events(bus, s)]
    assert order.index("refined") > order.index("final")
    # everything acked → nothing pending
    assert bus.depth(s.bus.stream_fast, "asr") == 0 and bus.depth(s.bus.stream_refine, "asr") == 0


def test_refine_never_delays_fast_under_load():
    s = make()
    bus, asr = InMemoryBus(), FakeASR()
    pipes = [ChannelPipeline(s, c, bus, EnergyVAD()) for c in s.channels]
    for p in pipes:
        audio, _ = programme([[1, 2]] * 10, pause_s=0.6)
        feed(p, audio)
        p._flush()  # release the last refine (waits for post-context audio otherwise)
    svc = AsrService(s, bus, asr, consumer="t")
    drain(svc)
    order = [e["type"] for e in events(bus, s)]
    n_final = order.count("final")
    assert n_final == 20 and order[:n_final] == ["final"] * n_final  # all FAST first
    assert order.count("refined") == 20


def test_refine_skipped_when_lagging_is_explicit():
    s = make(asr__refine_max_lag_s=0.0)
    bus, asr = InMemoryBus(), FakeASR()
    pipe = ChannelPipeline(s, s.channels[0], bus, EnergyVAD(), wall=lambda: time.time() - 10)
    audio, _ = programme([[1, 2], [3, 4]])
    feed(pipe, audio)
    svc = AsrService(s, bus, asr, consumer="t")
    drain(svc)
    assert len(events(bus, s, "final")) == 2
    sk = events(bus, s, "refine_skipped")
    assert len(sk) == 2 and all("refine_skipped:refine_lag" in e["flags"] for e in sk)
    assert bus.depth(s.bus.stream_refine, "asr") == 0


def test_poison_job_isolated_and_reported():
    s = make(refine=False)
    bus = InMemoryBus()
    asr = FakeASR(fail_on=lambda j: len(j.audio) > 0 and j.meta.get("seq") == 2)
    pipe = ChannelPipeline(s, s.channels[0], bus, EnergyVAD())
    audio, _ = programme([[1], [2, 3], [4]])
    feed(pipe, audio)
    svc = AsrService(s, bus, asr, consumer="t")
    drain(svc)
    assert [ids_of(e["text_raw"]) for e in events(bus, s, "final")] == [[1], [4]]
    err = events(bus, s, "error")
    assert len(err) == 1 and "boom" in err[0]["error"]
    assert bus.depth(s.bus.stream_fast, "asr") == 0


def test_hallucination_on_music_dropped():
    s = make(refine=False)
    bus = InMemoryBus()

    class Hallu(FakeASR):
        def transcribe(self, batch, opts):
            from live_stt.asr.backends.base import AsrResult, SegmentResult
            return [AsrResult(j.job_id, [SegmentResult("Thanks for watching!", 0, 1, -1.5, 0.9,
                                                       1.1, [])], "ur", 0.01, opts.model)
                    for j in batch]

    pipe = ChannelPipeline(s, s.channels[0], bus, EnergyVAD())
    feed(pipe, programme([[1, 2, 3]])[0])
    svc = AsrService(s, bus, Hallu(), consumer="t")
    drain(svc)
    f = events(bus, s, "final")
    assert len(f) == 1 and f[0]["text_raw"] == "" and any(x.startswith("dropped:") for x in f[0]["flags"])


def test_partials_superseded_and_final_wins():
    s = make(refine=False, streaming__partials_enabled=True, streaming__partial_interval_s=0.5)
    bus, asr = InMemoryBus(), FakeASR()
    pipe = ChannelPipeline(s, s.channels[0], bus, EnergyVAD())
    feed(pipe, programme([list(range(1, 12))])[0])
    svc = AsrService(s, bus, asr, consumer="t")
    drain(svc)
    f = events(bus, s, "final")
    assert len(f) == 1 and ids_of(f[0]["text_raw"]) == list(range(1, 12))
    # superseded partials were acked (not processed), nothing left pending
    assert bus.depth(s.bus.stream_fast, "asr") == 0
    assert len(asr.calls) < 1 + len(bus.messages(s.bus.stream_fast))


def test_local_agreement_mode_no_lost_or_duplicated_words():
    s = make(mode="local_agreement", refine=True, streaming__step_s=0.5,
             streaming__max_segment_s=4.0, streaming__max_buffer_s=8.0)
    bus, asr = InMemoryBus(), FakeASR()
    pipe = ChannelPipeline(s, s.channels[0], bus, EnergyVAD())
    groups = [list(range(1, 15)), [20, 21, 22], list(range(30, 40))]
    audio, ids = programme(groups)
    svc = AsrService(s, bus, asr, consumer="t")
    for i in range(0, len(audio), 1600):  # interleave worker and ASR like real time
        pipe.on_chunk(AudioChunk(audio[i:i + 1600], T0 + i / SR), True)
        drain(svc)
    pipe._flush()
    drain(svc)
    finals = events(bus, s, "final")
    got = [x for e in finals for x in ids_of(e["text_raw"])]
    assert got == ids, f"words lost/duplicated: {got}"
    partials = events(bus, s, "partial")
    assert partials, "local_agreement must emit partials"
    refined = events(bus, s, "refined")
    assert {e["segment_id"] for e in refined} == {e["segment_id"] for e in finals}


def test_empty_refine_without_hallucination_verdict_keeps_fast():
    s = make()
    bus = InMemoryBus()

    class EmptyRefine(FakeASR):
        def transcribe(self, batch, opts):
            if opts.model == s.asr.refine.model:
                from live_stt.asr.backends.base import AsrResult
                return [AsrResult(j.job_id, [], "ur", 0.01, opts.model) for j in batch]
            return super().transcribe(batch, opts)

    pipe = ChannelPipeline(s, s.channels[0], bus, EnergyVAD())
    feed(pipe, programme([[1, 2]])[0])
    pipe._flush()
    svc = AsrService(s, bus, EmptyRefine(), consumer="t")
    drain(svc)
    assert len(events(bus, s, "final")) == 1
    assert events(bus, s, "refined") == []
    sk = events(bus, s, "refine_skipped")
    assert len(sk) == 1 and sk[0]["flags"] == ["refine_skipped:refine_empty"]


def test_malformed_requests_are_dropped_not_crash():
    s = make(refine=False)
    bus, asr = InMemoryBus(), FakeASR()
    good = None
    pipe = ChannelPipeline(s, s.channels[0], bus, EnergyVAD())
    feed(pipe, programme([[1, 2]])[0])
    pipe._flush()
    for bad in [{}, {"channel_id": "x"}, {"channel_id": "x", "utc_start": "abc"},
                {"channel_id": "x", "utc_start": 1.0, "pcm": b"\x00" * (2 * 16000 * 121)},
                {"kind": "stream", "channel_id": "x"}]:
        bus.publish(s.bus.stream_fast, bad)
    svc = AsrService(s, bus, asr, consumer="t")
    for _ in range(5):
        for mid, msg in bus.consume(s.bus.stream_fast, "asr", "t", count=100):
            svc._intake(mid, "fast", msg)
        svc.step()
    assert [ids_of(e["text_raw"]) for e in events(bus, s, "final")] == [[1, 2]]
    assert bus.depth(s.bus.stream_fast, "asr") == 0


def test_emit_failure_does_not_kill_service():
    s = make(refine=False)
    bus = InMemoryBus()
    pipe = ChannelPipeline(s, s.channels[0], bus, EnergyVAD())
    feed(pipe, programme([[1], [2]])[0])
    pipe._flush()
    svc = AsrService(s, bus, FakeASR(), consumer="t")
    calls = {"n": 0}
    real = svc._publish

    def flaky(ev):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("redis down")
        real(ev)

    svc._publish = flaky
    drain(svc)  # must not raise
    finals = events(bus, s, "final")
    assert len(finals) >= 1  # second segment processed despite the first failure


def test_fast_load_shedding_records_gap_events_and_acks():
    s = make(asr__fast_max_lag_s=5.0, asr__refine_enabled=False)
    bus, asr = InMemoryBus(), FakeASR()
    # worker clock 60 s in the past → every FAST job is already older than fast_max_lag_s
    pipe = ChannelPipeline(s, s.channels[0], bus, EnergyVAD(), wall=lambda: time.time() - 60)
    audio, _ = programme([[1, 2], [3, 4]])
    feed(pipe, audio)
    svc = AsrService(s, bus, asr, consumer="t")
    drain(svc)
    assert events(bus, s, "final") == []
    gaps = [e for e in bus.messages(s.bus.stream_system) if e["type"] == "fast_skipped"]
    assert len(gaps) == 2 and all(g["details"]["reason"] == "fast_lag" for g in gaps)
    assert all(g["details"]["duration_s"] > 0 for g in gaps)
    assert bus.depth(s.bus.stream_fast, "asr") == 0
