"""Video archive: an index of every HLS segment the relay wrote, so any past time range can be
replayed (VOD playlist with EXT-X-PROGRAM-DATE-TIME) or downloaded (MPEG-TS / MP4, no re-encode).

Layout per channel (``<relay.dir>/<channel>/``)::

    index.m3u8                         rolling live playlist (ffmpeg hlsenc)
    a/YYYYMMDD/seg_<time>_<session>_<n>.ts   archived segments (kept ``archive_hours``)
    archive/YYYYMMDD.idx               one line per segment: start_epoch \\t duration \\t path \\t disc

The relay appends to the index whenever ffmpeg lists a new segment in the live playlist (a
segment is listed only after it is closed). Index days are UTC dates of the segment start.
Pure functions except the explicit file operations; used by the relay (writer) and the API.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

ARCHIVE_DIR = "a"
INDEX_DIR = "archive"
SEG_REL = re.compile(r"^a/\d{8}/seg_\d{8}T\d{6}_\d{1,8}_\d{4,8}\.ts$")
_PDT = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)(Z|[+-]\d{2}:?\d{2})?$")


@dataclass(frozen=True)
class Entry:
    t: float  # UTC epoch of the first frame (PROGRAM-DATE-TIME)
    d: float  # duration, seconds
    uri: str  # path relative to the channel directory
    disc: bool = False  # discontinuity before this segment (new muxer session)

    @property
    def end(self) -> float:
        return self.t + self.d


def parse_pdt(s: str) -> float:
    """``2026-11-24T09:29:36.724+0500`` (ffmpeg) / ``…Z`` / ``…+05:00`` → epoch seconds."""
    m = _PDT.match(s.strip())
    if not m:
        raise ValueError(f"bad PROGRAM-DATE-TIME {s!r}")
    base, off = m.group(1), m.group(2) or "Z"
    d = datetime.fromisoformat(base)
    if off == "Z":
        tz = timezone.utc
    else:
        sign = -1 if off[0] == "-" else 1
        hh, mm = int(off[1:3]), int(off[-2:])
        tz = timezone(sign * timedelta(hours=hh, minutes=mm))
    return d.replace(tzinfo=tz).timestamp()


def rel_uri(uri: str, chan_dir: Path) -> str:
    """ffmpeg writes the segment path as given (absolute for strftime names): make it relative."""
    p = Path(uri)
    if p.is_absolute():
        try:
            return p.relative_to(chan_dir).as_posix()
        except ValueError:
            return p.name
    return uri


def parse_playlist(text: str, chan_dir: Path | None = None) -> list[Entry]:
    out: list[Entry] = []
    pdt: float | None = None
    dur: float | None = None
    disc = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#EXTINF:"):
            try:
                dur = float(line[8:].split(",", 1)[0])
            except ValueError:
                dur = None
        elif line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            try:
                pdt = parse_pdt(line.split(":", 1)[1])
            except ValueError:
                pdt = None
        elif line.startswith("#EXT-X-DISCONTINUITY"):
            disc = True
        elif not line.startswith("#"):
            if pdt is not None and dur is not None:
                uri = rel_uri(line, chan_dir) if chan_dir is not None else line
                out.append(Entry(pdt, dur, uri, disc))
                pdt = pdt + dur  # next segment without its own PDT continues from here
            dur, disc = None, False
    return out


def rewrite_live_playlist(text: str, chan_dir: Path) -> str:
    """Serve ffmpeg's live playlist with URIs relative to ``/hls/<channel>/``."""
    return "\n".join(line if (not line.strip() or line.startswith("#"))
                     else rel_uri(line.strip(), chan_dir) for line in text.splitlines()) + "\n"


# --------------------------------------------------------------------------- index
def _day(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y%m%d")


class ArchiveIndex:
    def __init__(self, chan_dir: str | Path) -> None:
        self.dir = Path(chan_dir)
        self.idx_dir = self.dir / INDEX_DIR

    def last_t(self) -> float:
        files = sorted(self.idx_dir.glob("*.idx")) if self.idx_dir.is_dir() else []
        for f in reversed(files):
            try:
                with f.open("rb") as fh:
                    fh.seek(max(0, f.stat().st_size - 4096))
                    lines = fh.read().decode("utf-8", "replace").splitlines()
            except OSError:
                continue
            for line in reversed(lines):
                try:
                    return float(line.split("\t", 1)[0])
                except ValueError:
                    continue
        return 0.0

    def recent_uris(self, tail_bytes: int = 256 * 1024) -> set[str]:
        """URIs in the tail of the newest index files (covers any live playlist window), so a
        restarted relay does not index the segments ``append_list`` carries over twice."""
        out: set[str] = set()
        files = sorted(self.idx_dir.glob("*.idx")) if self.idx_dir.is_dir() else []
        for f in files[-2:]:
            try:
                with f.open("rb") as fh:
                    fh.seek(max(0, f.stat().st_size - tail_bytes))
                    lines = fh.read().decode("utf-8", "replace").splitlines()
            except OSError:
                continue
            for line in lines:
                p = line.split("\t")
                if len(p) >= 3:
                    out.add(p[2])
        return out

    def append(self, entries: list[Entry]) -> int:
        if not entries:
            return 0
        self.idx_dir.mkdir(parents=True, exist_ok=True)
        by_day: dict[str, list[str]] = {}
        for e in entries:
            by_day.setdefault(_day(e.t), []).append(
                f"{e.t:.3f}\t{e.d:.3f}\t{e.uri}\t{int(e.disc)}\n")
        for day, lines in by_day.items():
            with (self.idx_dir / f"{day}.idx").open("a", encoding="utf-8") as fh:
                fh.writelines(lines)
        return len(entries)

    def read(self, t0: float, t1: float, existing_only: bool = True) -> list[Entry]:
        """Segments overlapping [t0, t1), oldest first."""
        out: list[Entry] = []
        day = datetime.fromtimestamp(t0, timezone.utc).date() - timedelta(days=1)
        last = datetime.fromtimestamp(t1, timezone.utc).date()
        while day <= last:
            f = self.idx_dir / f"{day:%Y%m%d}.idx"
            day += timedelta(days=1)
            if not f.is_file():
                continue
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                p = line.split("\t")
                if len(p) < 3:
                    continue
                try:
                    e = Entry(float(p[0]), float(p[1]), p[2], len(p) > 3 and p[3] == "1")
                except ValueError:
                    continue
                if e.end > t0 and e.t < t1 and SEG_REL.match(e.uri):
                    out.append(e)
        out.sort(key=lambda e: e.t)
        dedup: list[Entry] = []
        for e in out:
            if dedup and abs(dedup[-1].t - e.t) < 1e-3:
                continue
            dedup.append(e)
        if existing_only:
            dedup = [e for e in dedup if (self.dir / e.uri).is_file()]
        return dedup


def coverage(entries: list[Entry], gap_s: float = 1.0) -> list[tuple[float, float]]:
    spans: list[list[float]] = []
    for e in entries:
        if spans and e.t - spans[-1][1] <= gap_s:
            spans[-1][1] = max(spans[-1][1], e.end)
        else:
            spans.append([e.t, e.end])
    return [(a, b) for a, b in spans]


def vod_playlist(entries: list[Entry], gap_s: float = 0.5) -> str:
    """Finished (VOD) playlist; URIs relative to ``/hls/<channel>/``."""
    target = max([math.ceil(e.d) for e in entries] or [2])
    lines = ["#EXTM3U", "#EXT-X-VERSION:6", f"#EXT-X-TARGETDURATION:{target}",
             "#EXT-X-MEDIA-SEQUENCE:0", "#EXT-X-PLAYLIST-TYPE:VOD", "#EXT-X-INDEPENDENT-SEGMENTS"]
    prev: Entry | None = None
    for e in entries:
        if prev is not None and (e.disc or abs(e.t - prev.end) > gap_s):
            lines.append("#EXT-X-DISCONTINUITY")
        pdt = datetime.fromtimestamp(e.t, timezone.utc).isoformat(timespec="milliseconds")
        lines += [f"#EXTINF:{e.d:.3f},", f"#EXT-X-PROGRAM-DATE-TIME:{pdt.replace('+00:00', 'Z')}",
                  e.uri]
        prev = e
    lines.append("#EXT-X-ENDLIST")
    return "\n".join(lines) + "\n"


def sweep_retention(chan_dir: str | Path, hours: float, now: float | None = None) -> int:
    """Delete archived segments older than ``hours`` and index days entirely before that."""
    if hours <= 0:
        return 0
    chan_dir = Path(chan_dir)
    now = time.time() if now is None else now
    cutoff = now - hours * 3600
    removed = 0
    adir = chan_dir / ARCHIVE_DIR
    if adir.is_dir():
        for day in sorted(adir.iterdir()):
            if not day.is_dir():
                continue
            for f in day.glob("seg_*.ts"):
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                        removed += 1
                except OSError:
                    pass
            try:  # never the directory ffmpeg is about to write into
                if day.stat().st_mtime < cutoff and not any(day.iterdir()):
                    day.rmdir()
            except OSError:
                pass
    idx = chan_dir / INDEX_DIR
    keep_from = _day(cutoff - 86400)
    if idx.is_dir():
        for f in idx.glob("*.idx"):
            if f.stem < keep_from:
                f.unlink(missing_ok=True)
    return removed


# --------------------------------------------------------------------------- clips
def iter_ts_bytes(files: list[Path], chunk: int = 1 << 20):
    """MPEG-TS segments concatenate into a playable TS file (players handle the PTS jumps)."""
    for f in files:
        try:
            with f.open("rb") as fh:
                while True:
                    b = fh.read(chunk)
                    if not b:
                        break
                    yield b
        except OSError:
            continue


def remux_mp4(files: list[Path], out_path: str | Path) -> int:
    """Stream-copy TS segments into one MP4 (faststart). Timestamps are re-based to stay
    monotonic across muxer sessions. Returns packets written; raises on failure."""
    import av

    from live_stt.ingest.relay import TimestampRebaser

    out = av.open(str(out_path), "w", format="mp4", options={"movflags": "+faststart"})
    omap: dict[tuple[str, int], object] = {}
    rebase = TimestampRebaser()
    n = 0
    origin: float | None = None
    try:
        for f in files:
            try:
                inp = av.open(str(f))
            except Exception:  # noqa: BLE001 - a damaged segment is skipped
                continue
            try:
                streams = [s for s in inp.streams if s.type in ("video", "audio")]
                for s in streams:
                    key = (s.type, sum(1 for x in streams[: streams.index(s)] if x.type == s.type))
                    if key not in omap and not n:
                        omap[key] = out.add_stream_from_template(s)
                for pkt in inp.demux(streams):
                    if pkt.dts is None or pkt.size == 0:
                        continue
                    s = pkt.stream
                    key = (s.type, sum(1 for x in streams[: streams.index(s)] if x.type == s.type))
                    ost = omap.get(key)
                    if ost is None:
                        continue
                    tb = pkt.time_base or s.time_base
                    t = float(pkt.dts * tb)
                    off = rebase.map(list(omap).index(key), t)
                    if off is None:
                        continue
                    if origin is None:
                        origin = t + off
                    shift = off - origin  # start the file at t = 0
                    ticks = round(shift / tb)
                    pkt.dts += ticks
                    if pkt.pts is not None:
                        pkt.pts += ticks
                    if pkt.dts < 0:
                        continue
                    pkt.stream = ost
                    out.mux(pkt)
                    n += 1
            finally:
                inp.close()
    finally:
        out.close()
    if n == 0:
        raise ValueError("no packets")
    return n


def remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def du_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total
