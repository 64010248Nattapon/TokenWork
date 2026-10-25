#!/usr/bin/env bash
# SGLang on the same BF16 weights, for the engine comparison (not validated by CMKL).
#   ./serving/sglang_bf16.sh
#   MTP=1 ./serving/sglang_bf16.sh          # NextN speculative decoding
set -euo pipefail
source "$(dirname "$0")/env.sh"
MEM_FRACTION=${MEM_FRACTION:-0.80}
spec=()
if [[ ${MTP:-0} == 1 ]]; then
  spec=(--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1
        --speculative-num-draft-tokens 4)
fi
exec python -m sglang.launch_server --model-path "$MODEL_BF16" \
  --served-model-name "$SERVED_NAME" --port "$PORT" \
  --context-length "$MAX_MODEL_LEN" \
  --mem-fraction-static "$MEM_FRACTION" \
  --reasoning-parser qwen3 \
  "${spec[@]}" "$@"
