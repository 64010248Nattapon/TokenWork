#!/usr/bin/env bash
# The submitted solution. Graded history (all S 99.9, 299/300 on time):
#   #16/#17  RTN W8A8 (mango-w8a8)               THINK_BUDGET=2048   R 1.040  140 J/req  η 4.38×
#   #19      Nattapon248/mango15-sq-gptq-w8a8     THINK_BUDGET=2048   R 1.011  137 J/req  η 4.47×
#
#   serving/final.sh                                   # = #16/#17 exactly (RTN W8A8, budget 2048)
#   THINK_BUDGET=4096 serving/final.sh                 # same, longer thinking cap (see below)
#   FINAL_MODEL=Nattapon248/mango15-sq-gptq-w8a8 serving/final.sh   # our GPTQ+SmoothQuant W8A8 instead
#   SERVED_NAME=shogun serving/final.sh                # must match the grader's model dropdown
#
# Why THINK_BUDGET=4096 is the next thing to try:
#   * S is capped at 100 and s_E is already saturated (η ≥ 3× gives full marks; we are at 4.4×), so energy has
#     ~45 % headroom that earns nothing. The tie-break is accuracy retention (R), then p95 TTFT.
#   * A 2048-token cap forces an answer early on the hardest reasoning/maths items; budget forcing loses accuracy
#     when the cap is tight (s1, arXiv 2501.19393). A forced early answer can also keep "reasoning" in the content
#     and run into the 8,192-token limit — a candidate for the one request that fails in every run.
#   * 4096 still leaves 4,096 tokens for the answer, so nothing is cut that would not be cut anyway.
#   Check after the run: η must stay ≥ ~3.3× (s_E = 1 with margin); if it drops below, go back to 2048 or 3072.
#
# Unchanged from the graded runs: prefix caching, BF16 DeltaNet SSM state, 8192-token steps, long prompts chunked
# at 2048 tokens, --reasoning-parser qwen3, qwen3_coder tool calling, usage in every stream, default clocks/power.
set -euo pipefail
here=$(dirname "$0")
export THINK_BUDGET=${THINK_BUDGET:-2048} PREFIX=1 BATCHED=8192 LONG_PREFILL=2048 REASONING_FLAGS='--reasoning-parser qwen3'
source "$here/env.sh"
echo "final.sh: THINK_BUDGET=$THINK_BUDGET model=${FINAL_MODEL:-$WORK/models/mango-w8a8} served-name=$SERVED_NAME"
if [[ -n ${FINAL_MODEL:-} ]]; then
  # any compressed-tensors checkpoint (local dir or HF id) with exactly the same flags as the w8a8 candidate
  export MODEL_W4A16=$FINAL_MODEL
  exec "$here/candidate.sh" w4a16-cmkl "$@"     # this branch serves $MODEL_W4A16 with the quantized-candidate flags
fi
[[ -f $WORK/models/mango-w8a8/config.json ]] || bash "$here/../quant/make_candidates.sh" w8a8
exec "$here/candidate.sh" w8a8 "$@"