#!/usr/bin/env bash
# GPU etiquette for one GPU shared by the whole team (see CLAUDE.md).
#   serving/gpu_lock.sh show
#   serving/gpu_lock.sh claim <name> <job> <expected end, e.g. 11:30>
#   serving/gpu_lock.sh release <name>
set -euo pipefail
LOCK=${GPU_LOCK_FILE:-${WORK:-$HOME/work}/GPU_LOCK}

case ${1:-show} in
  show)
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
    if [[ -s $LOCK ]]; then echo "LOCKED: $(cat "$LOCK")"; else echo "lock: free"; fi
    ;;
  claim)
    (( $# == 4 )) || { echo "usage: $0 claim <name> <job> <expected end>" >&2; exit 2; }
    mkdir -p "$(dirname "$LOCK")"
    # noclobber makes the create atomic: two people claiming at once cannot both win.
    if ! (set -o noclobber; echo "$2 | $3 | $(date +%H:%M) | $4" > "$LOCK") 2>/dev/null; then
      echo "GPU already held: $(cat "$LOCK")" >&2
      exit 1
    fi
    echo "claimed: $(cat "$LOCK")"
    ;;
  release)
    (( $# == 2 )) || { echo "usage: $0 release <name>" >&2; exit 2; }
    [[ -s $LOCK ]] || { echo "lock already free"; exit 0; }
    owner=$(cut -d'|' -f1 "$LOCK" | xargs)
    if [[ $owner != "$2" ]]; then
      echo "lock belongs to '$owner', not '$2'; ask them first." >&2
      exit 1
    fi
    rm -f "$LOCK"
    echo "released"
    ;;
  *)
    echo "usage: $0 show | claim <name> <job> <end> | release <name>" >&2
    exit 2
    ;;
esac
