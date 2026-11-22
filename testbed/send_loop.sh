#!/usr/bin/env bash
# Loop a TS file to UDP at real-time rate (like a live channel).
# Usage: testbed/send_loop.sh file.ts PORT [MULTICAST_GROUP] [N_CHANNELS]
#   N_CHANNELS>1 starts N senders on PORT, PORT+1, ... (each a separate ffmpeg).
# Note: ffmpeg's -stream_loop rewrites timestamps so PTS stays continuous across loops. Set RESET=1
# to restart ffmpeg on every loop instead: PTS then jumps back to the file start each loop, which
# emulates an encoder reset / playout switch (exercises the clock's discontinuity handling).
# BURST=<seconds> sets ffmpeg's -readrate_initial_burst (ffmpeg default 0.5 s sends the first half
# second instantly; BURST=0 behaves like a real encoder — use it for timestamp-accuracy runs).
# Each sender's start time is written to testbed/run/sender_<port>.start (epoch seconds) so the
# benchmark can compute stream position = (now - start) mod duration.
set -euo pipefail
[[ $# -ge 2 ]] || { echo "usage: $0 file.ts PORT [MCAST_GROUP|-] [N]" >&2; exit 2; }
file="$1"; port="$2"; group="${3:--}"; n="${4:-1}"
host="127.0.0.1"; [[ "$group" != "-" ]] && host="$group"
dir="$(cd "$(dirname "$0")" && pwd)"; mkdir -p "$dir/run"
rate=(-re)
[[ -n "${BURST:-}" ]] && rate+=(-readrate_initial_burst "$BURST")
pids=()
for ((i=0; i<n; i++)); do
  p=$((port + i))
  url="udp://${host}:${p}?pkt_size=1316"
  [[ "$group" != "-" ]] && url="${url}&ttl=1"
  python3 -c 'import time;print(f"{time.time():.6f}")' > "$dir/run/sender_${p}.start"
  if [[ "${RESET:-0}" == "1" ]]; then
    ( while true; do ffmpeg -hide_banner -loglevel warning "${rate[@]}" -i "$file" -c copy -f mpegts "$url" || sleep 1; done ) &
  else
    ffmpeg -hide_banner -loglevel warning "${rate[@]}" -stream_loop -1 -i "$file" -c copy -f mpegts "$url" &
  fi
  pids+=($!)
  echo "sender pid $! -> $url"
done
cleanup() {
  set +e   # never abort cleanup half-way (pgrep exits 1 when there are no children)
  # In RESET mode ffmpeg is a grandchild (child of the loop subshell): collect it before killing
  # the subshell, otherwise it is re-parented to launchd/init and keeps sending.
  local kids=""
  for p in "${pids[@]}"; do kids+=" $(pgrep -P "$p" 2>/dev/null | tr '\n' ' ')"; done
  kill "${pids[@]}" 2>/dev/null
  [[ -n "${kids// /}" ]] && kill $kids 2>/dev/null
  pkill -P $$ 2>/dev/null
  true
}
trap cleanup INT TERM EXIT
wait
