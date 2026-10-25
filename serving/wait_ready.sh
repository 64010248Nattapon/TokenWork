#!/usr/bin/env bash
# Wait until the server answers /health, then print the served models.
#   serving/wait_ready.sh [timeout-seconds] [log-file-to-tail-on-failure]
set -euo pipefail
source "$(dirname "$0")/env.sh"
TIMEOUT=${1:-900}
LOG=${2:-}
deadline=$(( $(date +%s) + TIMEOUT ))
until curl -sf --max-time 3 "http://localhost:$PORT/health" >/dev/null; do
  if (( $(date +%s) > deadline )); then
    echo "error: no healthy server on port $PORT after ${TIMEOUT}s" >&2
    [[ -n $LOG && -f $LOG ]] && tail -n 30 "$LOG" >&2
    exit 1
  fi
  sleep 5
done
echo "ready on port $PORT:"
curl -s "http://localhost:$PORT/v1/models"
echo
