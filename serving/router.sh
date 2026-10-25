#!/usr/bin/env bash
# Thinking router in front of a running server (api/router.py). Only if RULES.md lets us choose thinking per request.
#   serving/router.sh                       # budget forcing at 512 tokens, :8001 -> :8000
#   ROUTER_MODE=ask serving/router.sh       # think only when the model says it needs to (mixed workloads)
#   ROUTER_PORT=8000 UPSTREAM=http://localhost:8100 serving/router.sh   # router on the official port, vLLM moved to
#                                           # 8100 (start it with: PORT=8100 serving/candidate.sh ...)
set -euo pipefail
cd "$(dirname "$0")/.."
source serving/env.sh
mode=${ROUTER_MODE:-budget}
args=(--mode "$mode" --port "${ROUTER_PORT:-8001}" --upstream "${UPSTREAM:-http://localhost:$PORT}" --tokenizer "$MODEL_BF16")
case $mode in
  budget) args+=(--think-budget "${THINK_BUDGET:-512}" --answer-max-tokens "${ANSWER_MAX_TOKENS:-1024}") ;;
  ask) args+=(--ask-threshold "${ASK_THRESHOLD:-0.84}") ;;
  cascade) args+=(--cascade-metric "${CASCADE_METRIC:-min}" --cascade-threshold "${CASCADE_THRESHOLD:--1.38}") ;;
esac
[[ -n ${ROUTER_LOG:-} ]] && args+=(--log "$ROUTER_LOG")
exec python api/router.py "${args[@]}" "$@"
