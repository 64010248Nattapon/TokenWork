#!/usr/bin/env bash
# Baseline + MTP self-speculative decoding (the model's own NextN head drafts
# NUM_SPEC tokens, the full model verifies them in one step).
#   NUM_SPEC=1 ./serving/vllm_mtp.sh
#   MODEL=$MODEL_W4A16 ./serving/vllm_mtp.sh                 # on the W4A16 checkpoint
#   SPEC_METHOD=qwen3_next_mtp ./serving/vllm_mtp.sh         # if this vLLM rejects "mtp"
# Read the draft acceptance rate from the server log ("SpecDecoding metrics").
set -euo pipefail
source "$(dirname "$0")/env.sh"
MODEL=${MODEL:-$MODEL_BF16}
NUM_SPEC=${NUM_SPEC:-3}
SPEC_METHOD=${SPEC_METHOD:-mtp}
exec $VLLM_CMD serve "$MODEL" \
  --served-model-name "$SERVED_NAME" --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --language-model-only \
  $REASONING_FLAGS \
  $TOOL_FLAGS \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --speculative-config "{\"model\": \"$MODEL\", \"method\": \"$SPEC_METHOD\", \"num_speculative_tokens\": $NUM_SPEC}" \
  "$@"
