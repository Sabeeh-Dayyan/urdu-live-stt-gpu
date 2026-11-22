#!/usr/bin/env bash
# Soak: looping Urdu programme with a PTS reset every loop (RESET=1) → full system → monitor.
# Usage: bench/soak_run.sh <hours>      (results: bench/out/soak.csv, soak_summary.json)
set -u
hours="${1:-24}"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD"
mkdir -p bench/out
docker compose exec -T redis redis-cli DEL asr.requests.fast asr.requests.refine >/dev/null
RESET=1 testbed/send_loop.sh testbed/media/fleurs_ur.ts 5101 - 1 > bench/out/soak_sender.log 2>&1 &
snd=$!
.venv/bin/python -m live_stt.cli supervise -c config/dev-mac-cpu.yaml > bench/out/soak_supervisor.log 2>&1 &
sup=$!
trap 'kill -TERM $sup 2>/dev/null; wait $sup 2>/dev/null; kill $snd 2>/dev/null; pkill -P $snd 2>/dev/null' EXIT
sleep 60
.venv/bin/python -m bench.soak --config config/dev-mac-cpu.yaml --hours "$hours" --interval 60
