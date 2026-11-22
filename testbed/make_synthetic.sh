#!/usr/bin/env bash
# Build a synthetic test clip (no external media needed): speech (macOS `say`), a music-like
# jingle (tones) and silence, so ingest, VAD and hallucination guards can be exercised.
# Real Urdu broadcast clips are required for accuracy benchmarking (Phase 3).
# Usage: testbed/make_synthetic.sh out.ts
set -euo pipefail
out="${1:-testbed/media/synthetic.ts}"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
mkdir -p "$(dirname "$out")"
say -o "$tmp/a.aiff" "Good evening. This is the nine o'clock news. The Prime Minister addressed the National Assembly in Islamabad today."
say -o "$tmp/b.aiff" "In other news, the weather in Lahore will remain hot and dry for the rest of the week."
ffmpeg -hide_banner -loglevel error -y \
  -i "$tmp/a.aiff" \
  -f lavfi -t 2 -i "anullsrc=r=48000:cl=stereo" \
  -f lavfi -t 6 -i "sine=frequency=440:sample_rate=48000,volume=0.3,apulsator=hz=2" \
  -f lavfi -t 1.5 -i "anullsrc=r=48000:cl=stereo" \
  -i "$tmp/b.aiff" \
  -f lavfi -t 3 -i "anullsrc=r=48000:cl=stereo" \
  -filter_complex "[0:a]aresample=48000,aformat=channel_layouts=stereo[s0];[2:a]aformat=channel_layouts=stereo[m];[4:a]aresample=48000,aformat=channel_layouts=stereo[s1];[s0][1:a][m][3:a][s1][5:a]concat=n=6:v=0:a=1[a]" \
  -map "[a]" "$tmp/audio.wav"
"$(dirname "$0")/make_ts.sh" "$tmp/audio.wav" "$out"
