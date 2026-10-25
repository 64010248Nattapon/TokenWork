#!/usr/bin/env bash
# GPU power limit and clock locks. Changing them needs root AND the organizers' (and team's) OK: CLAUDE.md.
# Every change refuses to run unless ALLOW_GPU_TUNING=1 is set, so it can't happen by accident.
#   serving/gpu_tune.sh show
#   ALLOW_GPU_TUNING=1 serving/gpu_tune.sh power 150        # watts (A100: 100-400)
#   ALLOW_GPU_TUNING=1 serving/gpu_tune.sh clocks 1095      # lock graphics clocks to MHz (nvidia-smi -lgc)
#   ALLOW_GPU_TUNING=1 serving/gpu_tune.sh clocks 210,900   # or a min,max range (lets the idle GPU clock down)
#   ALLOW_GPU_TUNING=1 serving/gpu_tune.sh reset            # default power limit, unlock clocks
# Set it after the server has started (with persistence mode off the driver may reset it when the GPU is idle),
# and always reset afterwards; scripts should `trap 'serving/gpu_tune.sh reset' EXIT`.
set -euo pipefail
smi() { if [[ $(id -u) == 0 ]]; then nvidia-smi "$@"; else sudo -n nvidia-smi "$@"; fi; }  # APEX: passwordless sudo
show() { nvidia-smi --query-gpu=power.limit,power.default_limit,clocks.sm,clocks.max.sm,clocks.applications.graphics --format=csv; }
change() { [[ ${ALLOW_GPU_TUNING:-0} == 1 ]] || { echo "refusing: set ALLOW_GPU_TUNING=1 (needs the organizers' OK)" >&2; exit 3; }; }
case ${1:-show} in
  show) show ;;
  power) change; smi -pl "${2:?watts}" >/dev/null; show ;;
  clocks) change; c=${2:?MHz}; [[ $c == *,* ]] || c=$c,$c; smi -lgc "$c" >/dev/null; show ;;
  reset) change
    default=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1)
    smi -pl "${default%.*}" >/dev/null; smi -rgc >/dev/null 2>&1 || true; show ;;
  *) echo "usage: serving/gpu_tune.sh show | power <W> | clocks <MHz> | reset" >&2; exit 2 ;;
esac
