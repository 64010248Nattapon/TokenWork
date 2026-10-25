#!/usr/bin/env bash
# Baseline: BF16 weights, vLLM, text-only (vision tower not loaded).
# Every improvement is measured against this. Extra flags are passed through:
#   ./serving/vllm_bf16.sh --max-num-seqs 64
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec $VLLM_CMD serve "$MODEL_BF16" \
  --served-model-name "$SERVED_NAME" --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --language-model-only \
  $REASONING_FLAGS \
  $TOOL_FLAGS \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  "$@"
