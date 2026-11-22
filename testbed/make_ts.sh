#!/usr/bin/env bash
# Convert any media file to broadcast-like MPEG-TS: H.264 (GOP 50, 25 fps), AAC 48 kHz stereo.
# Usage: testbed/make_ts.sh in.mp4 out.ts
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 in.(mp4|mkv|wav|...) out.ts" >&2; exit 2; }
in="$1"; out="$2"
if ffprobe -v error -select_streams v:0 -show_entries stream=index -of csv=p=0 "$in" | grep -q .; then
  vargs=(-map 0:v:0 -c:v libx264 -preset veryfast -profile:v main -pix_fmt yuv420p -r 25 -g 50 -keyint_min 50 -sc_threshold 0 -b:v 2M -maxrate 2M -bufsize 4M)
else
  # Audio-only input: add a test pattern video so the TS looks like TV.
  vargs=(-f lavfi -i "testsrc2=size=1280x720:rate=25" -map 1:v:0 -shortest -c:v libx264 -preset veryfast -pix_fmt yuv420p -g 50 -keyint_min 50 -sc_threshold 0 -b:v 1M)
fi
ffmpeg -hide_banner -y -i "$in" "${vargs[@]}" -map 0:a:0 -c:a aac -b:a 128k -ar 48000 -ac 2 \
  -metadata:s:a:0 language=urd -f mpegts -mpegts_service_id 1 -muxrate 0 "$out"
echo "wrote $out"
