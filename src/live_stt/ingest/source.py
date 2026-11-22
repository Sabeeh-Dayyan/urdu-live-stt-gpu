"""PyAV ingest: demux MPEG-TS (UDP or any URL/file), decode one audio track, resample to
16 kHz mono float32, timestamp with the PTS clock, reconnect forever.

Verified against PyAV 18.1.0:
* ``av.open(url, options=..., timeout=(open_s, read_s))``
* ``av.AudioResampler(format="flt", layout="mono", rate=16000)``; ``resample(frame)`` returns a
  list of frames; ``resample(None)`` flushes. Re-created when the input format/layout/rate changes
  mid-stream (ad breaks often switch stereo ↔ 5.1 or 48 ↔ 44.1 kHz).
* ``stream.id`` is the MPEG-TS PID; ``stream.language`` / ``metadata["language"]`` the ISO-639 tag.

Resampler delay accounting: swresample buffers a few samples, so the UTC of the first output
sample is ``utc(input) − (pending input samples expressed at 16 kHz)``; we track exactly how many
input samples went in vs. output samples came out.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np

from live_stt.ingest.clock import ClockEvent, PtsClock
from live_stt.logging import get_logger
from live_stt.settings import AudioTrack, ClockConfig, IngestConfig

log = get_logger(__name__)


class NoAudioStream(RuntimeError):
    pass


@dataclass
class AudioChunk:
    samples: np.ndarray  # float32 mono @ sample_rate
    utc: float  # UTC of samples[0]
    events: list[ClockEvent] = field(default_factory=list)


@dataclass
class SourceEvent:
    type: str  # "stream_up" | "stream_down"
    utc: float
    details: dict = field(default_factory=dict)


def build_url(url: str, cfg: IngestConfig) -> str:
    """Add ffmpeg udp protocol options to udp:// URLs unless already present."""
    parts = urlsplit(url)
    if parts.scheme != "udp":
        return url
    q = dict(parse_qsl(parts.query, keep_blank_values=True))
    q.setdefault("fifo_size", str(cfg.fifo_size))
    q.setdefault("overrun_nonfatal", "1")
    q.setdefault("timeout", str(cfg.timeout_us))
    q.setdefault("buffer_size", str(cfg.buffer_size))
    if cfg.localaddr:
        q.setdefault("localaddr", cfg.localaddr)
    return urlunsplit(parts._replace(query=urlencode(q)))


def select_audio_stream(streams, track: AudioTrack):
    audio = [s for s in streams if s.type == "audio"]
    if not audio:
        raise NoAudioStream("no audio streams in input")
    if track.pid is not None:
        for s in audio:
            if s.id == track.pid:
                return s
        raise NoAudioStream(f"audio PID {track.pid} not found (have {[s.id for s in audio]})")
    if track.language:
        want = track.language.lower()
        for s in audio:
            lang = (getattr(s, "language", None) or s.metadata.get("language") or "").lower()
            if lang == want:
                return s
        log.warning("audio_language_not_found", want=want, fallback_index=track.index)
    if track.index >= len(audio):
        raise NoAudioStream(f"audio index {track.index} out of range ({len(audio)} streams)")
    return audio[track.index]


class AudioSource:
    """Iterates ``AudioChunk`` / ``SourceEvent`` objects forever (until ``stop()``)."""

    def __init__(
        self,
        url: str,
        track: AudioTrack,
        ingest: IngestConfig,
        clock_cfg: ClockConfig,
        wallclock: Callable[[], float] = time.time,
        max_connects: int | None = None,
        on_packet: Callable[[int], None] | None = None,
        relay=None,
    ) -> None:
        self.url = build_url(url, ingest)
        self.track = track
        self.cfg = ingest
        self.clock_cfg = clock_cfg
        self.wallclock = wallclock
        self.sr = ingest.sample_rate
        self.max_connects = max_connects
        self.on_packet = on_packet
        self.relay = relay  # optional HlsRelay fed with every demuxed packet (video + audio)
        self._stop = False
        self.clock: PtsClock | None = None
        self.decode_errors = 0
        self.connects = 0
        self._container = None
        # File inputs decode faster than real time; a wall clock would see one huge "burst".
        # Use a virtual clock that advances with decoded media time (perfect real time).
        self.is_file = "://" not in url or url.startswith("file:")
        self._media_s = 0.0
        self._t0 = wallclock()
        if self.is_file:
            self.wallclock = lambda: self._t0 + self._media_s

    def stop(self) -> None:
        self._stop = True

    # ------------------------------------------------------------------ main
    def __iter__(self) -> Iterator[AudioChunk | SourceEvent]:
        delay = self.cfg.reconnect_min_s
        while not self._stop:
            if self.max_connects is not None and self.connects >= self.max_connects:
                return
            self.connects += 1
            got_audio = False
            try:
                for item in self._run_once():
                    if isinstance(item, AudioChunk) and not got_audio:
                        got_audio = True
                        delay = self.cfg.reconnect_min_s
                        yield SourceEvent("stream_up", item.utc, {"url": self._safe_url()})
                    yield item
                    if self._stop:
                        break
                reason = "eof"
            except Exception as e:  # noqa: BLE001 - any failure → reconnect
                reason = f"{type(e).__name__}: {e}"
            finally:
                self._close()
            if self._stop:
                return
            if got_audio:
                yield SourceEvent("stream_down", self.wallclock(), {"reason": reason[:500]})
            log.warning("stream_reconnect", reason=reason[:300], retry_in_s=round(delay, 2))
            if self.is_file and reason == "eof" and got_audio:
                delay = 0.0  # looping a file: restart immediately
            # jittered exponential backoff
            time.sleep(delay * random.uniform(0.8, 1.2))
            if self.is_file and delay == 0.0:
                delay = self.cfg.reconnect_min_s
                continue
            delay = min(delay * 2, self.cfg.reconnect_max_s)

    def _safe_url(self) -> str:
        return self.url.split("?", 1)[0]

    def _close(self) -> None:
        if self.relay is not None:
            self.relay.close()
        c, self._container = self._container, None
        if c is not None:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass

    def _run_once(self) -> Iterator[AudioChunk]:
        import av

        opts = {
            "fflags": "nobuffer+discardcorrupt",
            "probesize": str(self.cfg.probesize),
            "analyzeduration": str(self.cfg.analyzeduration_us),
        }
        tmo = self.cfg.timeout_us / 1e6
        self._container = container = av.open(self.url, options=opts, timeout=(tmo, tmo))
        stream = select_audio_stream(container.streams, self.track)
        stream.thread_type = "AUTO"
        demux = [stream]
        if self.relay is not None:  # also demux (never decode) the first video stream
            demux += [s for s in container.streams if s.type == "video"][:1]
            self.relay.open(demux)
        tb = stream.time_base
        self.clock = clock = PtsClock(tb, self.clock_cfg, wallclock=self.wallclock)
        log.info(
            "stream_open", url=self._safe_url(), pid=stream.id, codec=stream.codec_context.name,
            rate=stream.codec_context.sample_rate, time_base=str(tb),
        )
        resampler = None
        rs_key = None
        out_end_utc: float | None = None  # UTC just after the last emitted sample
        out_start_utc = float("-inf")
        in_total_16k = 0.0  # input samples fed, expressed at 16 kHz
        out_total = 0  # output samples produced

        for packet in container.demux(demux):
            if self._stop:
                return
            if packet.size == 0 and packet.pts is None:
                continue  # flush packet at EOF
            if packet.stream.index != stream.index:  # video: relay only
                self.relay.push(packet)
                continue
            if self.on_packet:
                self.on_packet(packet.size)
            try:
                frames = packet.decode()
            except av.error.FFmpegError as e:  # corrupt packet: count and continue
                self.decode_errors += 1
                if self.decode_errors % 100 == 1:
                    log.warning("decode_error", error=str(e), total=self.decode_errors)
                frames = []
            if self.relay is not None:  # after decoding: muxing re-bases the packet in place
                self.relay.push(packet)
            if not frames:
                continue
            for frame in frames:
                if frame.samples == 0 or frame.sample_rate <= 0:
                    continue
                key = (frame.format.name, frame.layout.name, frame.sample_rate)
                if key != rs_key:
                    if rs_key is not None:
                        log.info("audio_format_change", old=rs_key, new=key)
                    resampler = av.AudioResampler(format="flt", layout="mono", rate=self.sr)
                    rs_key = key
                    in_total_16k = float(out_total)  # reset pending accounting
                dur = frame.samples / frame.sample_rate
                pts = frame.pts
                if pts is None:  # synthesise from expected position
                    pts = clock.expected_pts if clock.expected_pts is not None else 0
                    pts = pts % clock.wrap_ticks
                utc_in, events = clock.map(pts, dur)
                self._media_s += dur
                for ev in events:  # intentional retroactive shift (ring buffer re-maps too)
                    if ev.type == "startup_correction" and out_end_utc is not None:
                        out_end_utc += ev.details["shift_s"]
                pending = in_total_16k - out_total  # samples inside the resampler
                in_total_16k += frame.samples * self.sr / frame.sample_rate
                try:
                    outs = resampler.resample(frame)
                except ValueError:  # setup mismatch the key did not catch → rebuild
                    resampler = av.AudioResampler(format="flt", layout="mono", rate=self.sr)
                    outs = resampler.resample(frame)
                    in_total_16k = out_total + frame.samples * self.sr / frame.sample_rate
                    pending = 0.0
                arrs = [o.to_ndarray().reshape(-1) for o in outs if o.samples]
                if not arrs:
                    if events:  # never lose clock events
                        yield AudioChunk(np.zeros(0, np.float32), utc_in, events)
                    continue
                x = np.concatenate(arrs).astype(np.float32, copy=False)
                out_total += len(x)
                utc_out = utc_in - max(pending, 0.0) / self.sr
                # Guarantee a monotonic output timeline (corrupt PTS, resampler resets): the ring
                # buffer's UTC→index mapping relies on it. Overlaps < 2 ms are allowed — that is
                # how the clock's rate-limited negative drift slew (≤ 0.5 ms per frame) shows up.
                if out_end_utc is not None and (utc_out < out_end_utc - 0.002
                                                or utc_out <= out_start_utc):
                    utc_out = out_end_utc
                out_start_utc = utc_out
                out_end_utc = utc_out + len(x) / self.sr
                yield AudioChunk(x, utc_out, events)
