#!/usr/bin/env bash
# W4A16 (4-bit weights, BF16 activations) through vLLM's Marlin kernels.
# The quantization scheme is read from the checkpoint's config.json; no
# --quantization flag is needed for compressed-tensors checkpoints.
#   ./serving/vllm_w4a16.sh                                   # $MODEL_W4A16 from env.sh
#   MODEL_W4A16=$WORK/models/mango-w4a16 ./serving/vllm_w4a16.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec $VLLM_CMD serve "$MODEL_W4A16" \
  --served-model-name "$SERVED_NAME" --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --language-model-only \
  $REASONING_FLAGS \
  $TOOL_FLAGS \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  "$@"
