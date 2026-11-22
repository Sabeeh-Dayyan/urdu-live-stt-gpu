#!/usr/bin/env bash
# Live measurement run: looping Urdu programme → full system → `stt bench-live`.
# Usage: bench/live_run.sh <label> <seconds> [extra env assignments...]
#   e.g. bench/live_run.sh segment 360
#        bench/live_run.sh local_agreement 300 STT_STREAMING__MODE=local_agreement
# Leaves nothing running when it exits.
set -u
label="$1"; secs="$2"; shift 2
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD"
for kv in "$@"; do export "$kv"; done
mkdir -p bench/out testbed/run
# Measurement hygiene: start from empty request streams (a previous run may have left a backlog;
# FAST requests are never dropped by design, so they would distort this run's latency).
docker compose exec -T redis redis-cli DEL asr.requests.fast asr.requests.refine >/dev/null
testbed/send_loop.sh testbed/media/fleurs_ur.ts 5101 - 1 > "bench/out/sender_$label.log" 2>&1 &
snd=$!
.venv/bin/python -m live_stt.cli supervise -c config/dev-mac-cpu.yaml > "bench/out/supervisor_$label.log" 2>&1 &
sup=$!
trap 'kill -TERM $sup 2>/dev/null; wait $sup 2>/dev/null; kill $snd 2>/dev/null; pkill -P $snd 2>/dev/null' EXIT
sleep 45   # model load + clock warmup
.venv/bin/python -m live_stt.cli bench-live -c config/dev-mac-cpu.yaml --seconds "$secs" \
  --out "bench/out/live_$label.json"
