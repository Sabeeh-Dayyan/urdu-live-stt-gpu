"""Rolling HLS copy of the received stream (video + audio, no re-encode) for the browser player.

Fed by ``AudioSource`` with the packets it already demuxes, so there is exactly one UDP receiver per
channel (a second receiver on the same unicast port could steal datagrams) and the video shares the
transcript's reception timeline. Each playlist entry carries ``EXT-X-PROGRAM-DATE-TIME`` (wall
clock), which the UI uses to seek to a sentence's UTC time.

Safety: the relay must never disturb transcription. Every failure is caught and logged; the muxer is
re-opened after ``retry_s``. Timestamps are re-based to stay monotonic across PTS jumps (encoder
resets / looping files), because the HLS muxer rejects non-monotonic DTS.

Verified against PyAV 18.1.0: ``OutputContainer.add_stream_from_template``, format ``hls`` with
options ``hls_time``, ``hls_list_size``, ``hls_flags``, ``hls_segment_filename`` (ffmpeg hlsenc).
"""

from __future__ import annotations

import math
import re
import time
from fractions import Fraction
from pathlib import Path

from live_stt.logging import get_logger

log = get_logger(__name__)

PLAYLIST = "index.m3u8"
SEGMENT_PATTERN = "seg_%08d.ts"
JUMP_BACK_S = 1.0  # backward DTS jump larger than this ⇒ re-base
JUMP_FWD_S = 30.0  # forward DTS jump larger than this ⇒ re-base
REBASE_GAP_S = 0.04


class TimestampRebaser:
    """Keeps output DTS/PTS monotonic per stream across input timestamp jumps (pure logic)."""

    def __init__(self) -> None:
        self.offset_s = 0.0
        self.last_s: float | None = None  # highest output DTS over all streams (seconds)
        self.last_by_stream: dict[int, float] = {}

    def map(self, stream_index: int, dts_s: float) -> float | None:
        """Return the offset (seconds) to add to this packet, or None to drop it."""
        out = dts_s + self.offset_s
        if self.last_s is not None and (out < self.last_s - JUMP_BACK_S
                                        or out > self.last_s + JUMP_FWD_S):
            self.offset_s = self.last_s + REBASE_GAP_S - dts_s
            out = dts_s + self.offset_s
        prev = self.last_by_stream.get(stream_index)
        if prev is not None and out <= prev:
            return None  # small backward glitch within a stream: drop, the muxer would reject it
        self.last_by_stream[stream_index] = out
        self.last_s = out if self.last_s is None else max(self.last_s, out)
        return self.offset_s


class HlsRelay:
    def __init__(self, outdir: str | Path, window_s: float = 900.0, segment_s: float = 2.0,
                 retry_s: float = 5.0) -> None:
        self.dir = Path(outdir)
        self.window_s = window_s
        self.segment_s = segment_s
        self.retry_s = retry_s
        self._templates: list = []
        self._out = None
        self._omap: dict[int, object] = {}
        self._rebase = TimestampRebaser()
        self._started = False
        self._has_video = False
        self._retry_at = 0.0
        self.errors = 0
        self.packets = 0

    # ----------------------------------------------------------------- lifecycle
    def open(self, in_streams: list) -> None:
        """(Re)start for a new connection. Appends to an existing playlist (discontinuity)."""
        self.close()
        self._templates = list(in_streams)
        self._has_video = any(s.type == "video" for s in in_streams)
        self._try_open()

    def _try_open(self) -> None:
        import av

        self._retry_at = 0.0
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            playlist = self.dir / PLAYLIST
            self._sweep(playlist)
            flags = ["delete_segments", "program_date_time", "independent_segments",
                     "omit_endlist"]
            if playlist.exists() and playlist.stat().st_size > 0:
                flags += ["append_list", "discont_start"]
            self._out = av.open(str(playlist), "w", format="hls", options={
                "hls_time": f"{self.segment_s:g}",
                "hls_list_size": str(max(3, math.ceil(self.window_s / self.segment_s))),
                "hls_flags": "+".join(flags),
                "hls_segment_filename": str(self.dir / SEGMENT_PATTERN),
            })
            self._omap = {s.index: self._out.add_stream_from_template(s)
                          for s in self._templates}
            self._rebase = TimestampRebaser()
            self._started = False
        except Exception as e:  # noqa: BLE001
            self._fail("relay_open_failed", e)

    def _sweep(self, playlist: Path) -> None:
        """Delete segment files the playlist no longer references. ffmpeg keeps one segment past
        the list (hls_delete_threshold) and a new muxer instance never learns about it, so every
        reconnect would otherwise leak one file. A playlist older than the window is discarded
        (its video is outside the replay window anyway)."""
        try:
            if playlist.exists() and time.time() - playlist.stat().st_mtime > self.window_s:
                playlist.unlink()
            keep = set(re.findall(r"seg_\d{8}\.ts", playlist.read_text())) \
                if playlist.exists() else set()
            for f in self.dir.glob("seg_*.ts"):
                if f.name not in keep:
                    f.unlink(missing_ok=True)
        except OSError as e:
            log.debug("relay_sweep_error", error=str(e))

    def close(self) -> None:
        out, self._out = self._out, None
        self._omap = {}
        if out is not None:
            try:
                out.close()
            except Exception as e:  # noqa: BLE001
                log.debug("relay_close_error", error=str(e))

    def _fail(self, what: str, e: Exception) -> None:
        self.errors += 1
        if self.errors <= 5 or self.errors % 100 == 0:
            log.warning(what, error=f"{type(e).__name__}: {e}", errors=self.errors,
                        dir=str(self.dir))
        self.close()
        self._retry_at = time.monotonic() + self.retry_s

    # ------------------------------------------------------------------- packets
    def push(self, packet) -> None:
        """Mux one demuxed input packet. Never raises."""
        try:
            if self._out is None:
                if not self._templates or not self._retry_at or time.monotonic() < self._retry_at:
                    return
                self._try_open()
                if self._out is None:
                    return
            if packet.dts is None or packet.size == 0:
                return
            ost = self._omap.get(packet.stream.index)
            if ost is None:
                return
            if not self._started:  # first segment must start on a video keyframe
                if self._has_video and not (packet.stream.type == "video" and packet.is_keyframe):
                    return
                self._started = True
            tb = packet.time_base or packet.stream.time_base or Fraction(1, 90000)
            off = self._rebase.map(packet.stream.index, float(packet.dts * tb))
            if off is None:
                return
            if off:
                ticks = round(off / tb)
                packet.dts += ticks
                if packet.pts is not None:
                    packet.pts += ticks
            packet.stream = ost
            self._out.mux(packet)
            self.packets += 1
        except Exception as e:  # noqa: BLE001
            self._fail("relay_mux_failed", e)
