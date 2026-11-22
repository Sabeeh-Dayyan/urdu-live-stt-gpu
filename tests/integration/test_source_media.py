"""PyAV ingest against nasty real media generated with ffmpeg."""

import os
import shutil
import socket
import subprocess
import time

import numpy as np
import pytest

from live_stt.ingest.source import AudioChunk, AudioSource, SourceEvent
from live_stt.settings import AudioTrack, ClockConfig, IngestConfig

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
FF = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]


def ff(*args):
    subprocess.run(FF + list(args), check=True, timeout=120)


def collect(url, track=AudioTrack(), max_connects=1, cfg=None):
    src = AudioSource(url, track, cfg or IngestConfig(reconnect_min_s=0.05), ClockConfig(),
                      max_connects=max_connects)
    chunks, events = [], []
    for it in src:
        if isinstance(it, SourceEvent):
            events.append(it.type)
        else:
            chunks.append(it)
            events += [e.type for e in it.events]
    n = sum(len(c.samples) for c in chunks)
    return chunks, events, n / 16000, src


def assert_monotonic(chunks):
    """Starts strictly increase once the startup correction is applied retroactively to earlier
    chunks (exactly what the worker does with ring.shift_utc)."""
    t = []
    for c in chunks:
        for e in c.events:
            if e.type == "startup_correction":
                t = [x + e.details["shift_s"] for x in t]
        if len(c.samples):
            t.append(c.utc)
    assert all(b >= a - 1e-6 for a, b in zip(t, t[1:])), "UTC must be monotonic"
    for c in chunks:
        assert c.samples.dtype == np.float32 and np.all(np.isfinite(c.samples))


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    v = ["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25"]
    # 1) plain AAC stereo 48k, 6 s
    ff(*v, "-f", "lavfi", "-i", "sine=f=440:r=48000", "-t", "6", "-ac", "2", "-c:v", "libx264",
       "-preset", "ultrafast", "-c:a", "aac", "-f", "mpegts", str(d / "plain.ts"))
    # 2) format changes mid-stream: stereo48k → mono44.1k → 5.1 48k (concatenated TS)
    parts = []
    for i, (ac, ar) in enumerate([(2, 48000), (1, 44100), (6, 48000)]):
        p = d / f"p{i}.ts"
        ff("-f", "lavfi", "-i", f"sine=f={300 + 100 * i}:r={ar}", "-t", "3", "-ac", str(ac),
           "-ar", str(ar), "-c:a", "aac", "-f", "mpegts", str(p))
        parts.append(p.read_bytes())
    (d / "fmtchange.ts").write_bytes(b"".join(parts))
    # 3) corrupted: random garbage overwritten in the middle
    b = bytearray((d / "plain.ts").read_bytes())
    rng = np.random.default_rng(0)
    for off in range(len(b) // 3, len(b) // 3 + 60_000, 997):
        b[off:off + 200] = rng.integers(0, 255, 200, dtype=np.uint8).tobytes()
    (d / "corrupt.ts").write_bytes(bytes(b))
    # 4) truncated mid-packet
    (d / "trunc.ts").write_bytes((d / "plain.ts").read_bytes()[: len(b) // 2 + 77])
    # 5) video only
    ff(*v, "-t", "2", "-c:v", "libx264", "-preset", "ultrafast", "-f", "mpegts", str(d / "novid.ts"))
    # 6) two audio tracks with language tags, MP2 + AAC
    ff("-f", "lavfi", "-i", "sine=f=300:r=48000", "-f", "lavfi", "-i", "sine=f=900:r=48000",
       "-t", "3", "-map", "0:a", "-map", "1:a", "-c:a:0", "mp2", "-c:a:1", "aac",
       "-metadata:s:a:0", "language=eng", "-metadata:s:a:1", "language=urd",
       "-f", "mpegts", str(d / "multi.ts"))
    # 7) not media at all
    (d / "garbage.ts").write_bytes(os.urandom(200_000))
    # 8) empty file
    (d / "empty.ts").write_bytes(b"")
    return d


def test_plain(media):
    chunks, ev, dur, _ = collect(str(media / "plain.ts"))
    assert abs(dur - 6.0) < 0.2 and ev[:2] == ["stream_up", "anchor"]
    assert_monotonic(chunks)
    span = chunks[-1].utc + len(chunks[-1].samples) / 16000 - chunks[0].utc
    assert abs(span - dur) < 0.1, "virtual clock must track media time for files"


def test_format_change_mid_stream(media):
    chunks, ev, dur, _ = collect(str(media / "fmtchange.ts"))
    assert abs(dur - 9.0) < 0.5
    assert_monotonic(chunks)


def test_corrupt_stream_survives(media):
    chunks, ev, dur, src = collect(str(media / "corrupt.ts"))
    assert dur > 3.0
    assert_monotonic(chunks)


def test_truncated(media):
    chunks, ev, dur, _ = collect(str(media / "trunc.ts"))
    assert 1.0 < dur < 4.0
    assert ev[-1] == "stream_down"


@pytest.mark.parametrize("name", ["novid.ts", "garbage.ts", "empty.ts", "does_not_exist.ts"])
def test_unusable_inputs_retry_without_crash(media, name):
    chunks, ev, dur, src = collect(str(media / name), max_connects=3)
    assert dur == 0 and src.connects == 3 and "stream_up" not in ev


def test_track_selection_by_language_and_pid(media):
    def freq(chunks):
        x = np.concatenate([c.samples for c in chunks])[16000:32000]
        return np.argmax(np.abs(np.fft.rfft(x))) * 16000 / len(x)

    c_urd, *_ = collect(str(media / "multi.ts"), AudioTrack(language="urd"))
    c_eng, *_ = collect(str(media / "multi.ts"), AudioTrack(language="eng"))
    c_idx, *_ = collect(str(media / "multi.ts"), AudioTrack(index=1))
    assert abs(freq(c_urd) - 900) < 20 and abs(freq(c_eng) - 300) < 20
    assert abs(freq(c_idx) - 900) < 20


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_udp_live_loss_reconnect_and_loop_discontinuity(media):
    """Real UDP: looping sender (discontinuity per loop), garbage datagrams, sender killed →
    stream_down, restarted → stream_up again. Sender control runs on timer threads because the
    source yields nothing while the stream is down."""
    import threading

    port = _free_port()
    # Restart ffmpeg every loop → PTS jumps back each loop (encoder reset emulation).
    loop = (f"while true; do ffmpeg -hide_banner -loglevel quiet -re -i '{media / 'plain.ts'}' "
            f"-c copy -f mpegts 'udp://127.0.0.1:{port}?pkt_size=1316'; done")
    sender = ["bash", "-c", loop]

    def start():
        return subprocess.Popen(sender, start_new_session=True)

    procs = [start()]
    cfg = IngestConfig(timeout_us=2_000_000, reconnect_min_s=0.2, reconnect_max_s=1.0,
                       probesize=200_000, analyzeduration_us=500_000)
    src = AudioSource(f"udp://127.0.0.1:{port}", AudioTrack(), cfg, ClockConfig(warmup_s=1.0))

    def garbage():
        g = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for _ in range(200):
            g.sendto(os.urandom(1316), ("127.0.0.1", port))
            time.sleep(0.005)

    def kill():
        import signal
        os.killpg(procs[0].pid, signal.SIGKILL)
        procs[0].wait()

    timers = [threading.Timer(3, garbage), threading.Timer(14, kill),
              threading.Timer(19, lambda: procs.append(start())),
              threading.Timer(30, src.stop)]
    for t in timers:
        t.start()
    events, audio_s, t0 = [], 0.0, time.time()
    try:
        for it in src:
            if isinstance(it, SourceEvent):
                events.append((round(time.time() - t0, 1), it.type))
            else:
                audio_s += len(it.samples) / 16000
                events += [(round(time.time() - t0, 1), e.type) for e in it.events]
    finally:
        for t in timers:
            t.cancel()
        import signal
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    types = [t for _, t in events]
    assert types.count("stream_up") >= 2, events
    assert "stream_down" in types, events
    assert "discontinuity" in types, events  # 6 s file looped
    assert audio_s > 15, (audio_s, events)


def test_udp_timestamp_accuracy_converges():
    """Steady-state: chunk UTC vs. arrival wall clock within 100 ms after ~40 s, despite ffmpeg's
    gradual initial send-ahead (warmup correction + drift slew)."""
    from live_stt.settings import PROJECT_ROOT

    ts = PROJECT_ROOT / "testbed/media/fleurs_ur.ts"
    if not ts.exists():
        pytest.skip("make testbed/media/fleurs_ur.ts")
    port = _free_port()
    p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "quiet", "-re", "-i", str(ts),
                          "-c", "copy", "-f", "mpegts", f"udp://127.0.0.1:{port}?pkt_size=1316"])
    src = AudioSource(f"udp://127.0.0.1:{port}", AudioTrack(), IngestConfig(), ClockConfig())
    t0, offs = time.time(), []
    try:
        for it in src:
            if isinstance(it, SourceEvent) or not len(it.samples):
                continue
            el = time.time() - t0
            if el > 40:
                offs.append(time.time() - (it.utc + len(it.samples) / 16000))
            if el > 48:
                break
    finally:
        p.kill()
        src.stop()
    med = float(np.median(offs))
    assert abs(med) < 0.1, f"median arrival-vs-UTC offset {med * 1000:.0f} ms"
