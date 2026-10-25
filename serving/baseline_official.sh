#!/usr/bin/env bash
# The organizers' untuned baseline (RULES.md [2], [7], [14]): default `vllm serve` on BF16 MANGO plus the two
# tool-calling flags the grader needs. It scores S = 0 by definition; serve it to see the grader's baseline numbers.
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec $VLLM_CMD serve "$MODEL_BF16" --served-model-name "$SERVED_NAME" --port "$PORT" $TOOL_FLAGS "$@"
