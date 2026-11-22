#!/usr/bin/env bash
# Run the whole system detached (survives closing the terminal / editor).
#   scripts/live.sh start [config]   default config: config/dev-mac-cpu-broadcast.yaml
#   scripts/live.sh stop | restart [config] | status | logs
# State: run/live.pid, run/live.config   Logs: var/log/live.log
# Starts nothing on the sender side: the UDP senders/feeds must already be running.
set -u
cd "$(dirname "$0")/.."
ROOT="$PWD"
PIDF="$ROOT/run/live.pid"; CFGF="$ROOT/run/live.config"; LOG="$ROOT/var/log/live.log"
mkdir -p run var/log

running() { [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

start() {
  local cfg="${1:-config/dev-mac-cpu-broadcast.yaml}"
  if running; then echo "already running (pid $(cat "$PIDF"), config $(cat "$CFGF" 2>/dev/null))"; return 0; fi
  [[ -f "$cfg" ]] || { echo "no such config: $cfg" >&2; return 2; }
  # Another copy (e.g. `make run`) would compete for the CPU and for port 8000.
  local others; others="$(pgrep -fl 'live_stt.cli supervise' || true)"
  if [[ -n "$others" && "${FORCE:-0}" != "1" ]]; then
    echo "another live-stt supervisor is already running:" >&2; echo "$others" >&2
    echo "stop it first (Ctrl-C in its terminal), or FORCE=1 to start anyway" >&2
    return 1
  fi
  docker compose up -d --wait postgres redis >/dev/null && docker compose up -d prometheus grafana >/dev/null \
    || { echo "docker compose failed (is Docker running?)" >&2; return 1; }
  PYTHONPATH="$ROOT/src" .venv/bin/python -m live_stt.cli show-config -c "$cfg" >/dev/null \
    || { echo "invalid config: $cfg" >&2; return 2; }
  echo "=== $(date -u +%FT%TZ) start $cfg" >> "$LOG"
  PYTHONPATH="$ROOT/src" nohup .venv/bin/python -m live_stt.cli supervise -c "$cfg" >> "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDF"; echo "$cfg" > "$CFGF"
  disown 2>/dev/null || true
  for _ in $(seq 1 45); do
    curl -sf localhost:8000/health >/dev/null 2>&1 && break
    running || { echo "supervisor exited — see $LOG" >&2; tail -20 "$LOG" >&2; return 1; }
    sleep 1
  done
  status
}

stop() {
  if ! running; then echo "not running"; rm -f "$PIDF"; return 0; fi
  local pid; pid="$(cat "$PIDF")"
  kill -TERM "$pid"
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && { echo "still running after 30 s, sending SIGKILL"; kill -KILL "$pid"; }
  rm -f "$PIDF"; echo "stopped"
}

status() {
  if running; then
    echo "running: pid $(cat "$PIDF"), config $(cat "$CFGF" 2>/dev/null)"
    curl -s localhost:8000/health | .venv/bin/python -c '
import json, sys
try: h = json.load(sys.stdin)
except Exception: print("  api: not answering yet"); sys.exit()
print("  status", h.get("status"), " redis", h.get("redis"), " db", h.get("db"))
for k, v in sorted(h.get("components", {}).get("supervisor", {}).get("children", {}).items()):
    print("  %-14s alive=%s restarts=%s" % (k, v.get("alive"), v.get("restarts")))'
    echo "  UI       http://localhost:8000"
    echo "  Grafana  http://localhost:3000/d/live-stt"
    echo "  logs     $LOG"
  else
    echo "not running"
  fi
}

case "${1:-status}" in
  start) start "${2:-}";;
  stop) stop;;
  restart) stop; start "${2:-$(cat "$CFGF" 2>/dev/null)}";;
  status) status;;
  logs) tail -n 100 -f "$LOG";;
  *) echo "usage: $0 start [config] | stop | restart [config] | status | logs" >&2; exit 2;;
esac
