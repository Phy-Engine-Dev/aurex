#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
mkdir -p .aurex
pidfile="$root/.aurex/web.pid"
config="$root/.config/aurex3.json"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "$root/.config/web-token" ]]; then
  AUREX_WEB_TOKEN=$(< "$root/.config/web-token")
  export AUREX_WEB_TOKEN
fi
case "${1:-status}" in
  start)
    if [[ -f "$pidfile" ]] && kill -0 "$(< "$pidfile")" 2>/dev/null; then
      echo "Aurex already running"; exit 0
    fi
    [[ -f "$config" ]] || { echo "Run scripts/configure-aurex3.py first" >&2; exit 1; }
    shift || true
    nohup setsid "$root/.venv/bin/python" -m aurex web --config "$config" --login "$@" >"$root/.aurex/web.log" 2>&1 </dev/null &
    printf '%s\n' "$!" > "$pidfile"
    echo "Aurex starting; log: $root/.aurex/web.log"
    ;;
  stop)
    if [[ -f "$pidfile" ]]; then
      pid=$(< "$pidfile")
      command=$(ps -p "$pid" -o args= || true)
      if [[ "$command" == *"$root/.venv/bin/python -m aurex web --config $config"* ]]; then
        kill "$pid"
        for ((attempt=0; attempt<40; attempt++)); do
          kill -0 "$pid" 2>/dev/null || break
          sleep 0.25
        done
        if kill -0 "$pid" 2>/dev/null; then
          echo "Aurex has not exited yet; refusing to start a duplicate" >&2
          exit 1
        fi
        echo "Stopped Aurex; database and artifacts retained"
      else
        echo "No matching Aurex process; not stopping another process"
      fi
    fi
    ;;
  status)
    [[ ! -f "$pidfile" ]] || ps -p "$(< "$pidfile")" -o pid=,etime=,args=
    curl --noproxy '*' --fail --silent http://127.0.0.1:4097/health
    echo
    ;;
  *) echo "Usage: $0 start [--poll] | stop | status" >&2; exit 2 ;;
esac
