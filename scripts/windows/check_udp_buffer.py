"""Does PyAV's bundled FFmpeg support the UDP receive FIFO (``fifo_size`` / ``overrun_nonfatal``)?

FFmpeg implements that FIFO with a receiver thread that needs pthread cancellation; builds without
it (typical on Windows) ignore the options and print a warning, so packets are read only as fast
as the demuxer asks and bursts rely on the socket buffer (``ingest.buffer_size``) alone.

Usage: python scripts/windows/check_udp_buffer.py   (prints a short Markdown report)
"""

from __future__ import annotations

import platform
import socket

import av
import av.logging

from live_stt.ingest.source import build_url
from live_stt.settings import IngestConfig


def main() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    url = build_url(f"udp://127.0.0.1:{port}", IngestConfig(timeout_us=1_000_000))
    av.logging.set_level(av.logging.WARNING)
    with av.logging.Capture() as logs:
        try:
            av.open(url, timeout=3).close()
        except Exception:  # noqa: BLE001 - no sender: a timeout is expected
            pass
    msgs = [m for _lvl, _name, m in logs]
    unsupported = [m.strip() for m in msgs if "not supported on this build" in m]
    print(f"### UDP receive FIFO in PyAV {av.__version__} ({platform.system()} {platform.machine()})")
    print(f"- URL options: `{url.split('?', 1)[1]}`")
    if unsupported:
        print("- **FIFO not supported by this FFmpeg build**; the socket buffer "
              f"(`buffer_size`) carries bursts alone. FFmpeg said: {' / '.join(unsupported)}")
    else:
        print("- FIFO options accepted (no 'not supported' warning)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
