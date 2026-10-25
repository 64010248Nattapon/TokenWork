#!/usr/bin/env bash
# Serve one of the measured candidates with the flags that won on Colab (docs/PLAYBOOK.md).
#   serving/candidate.sh <bf16 | fp8 | w8a8 | w8a8-gdn | w8a16 | w4a16-cmkl> [extra vllm args]
# Env toggles (all off by default):
#   PREFIX=1        --enable-prefix-caching   (only helps when prompts share > ~528 tokens)
#   MAX_SEQS=512    --max-num-seqs            (needed above c=256)
#   BATCHED=16384   --max-num-batched-tokens  (long prompts: -3..8% J/token)
#   LONG_PREFILL=2048 --long-prefill-token-threshold (caps one long prompt's chunk per step so short ones still start)
#   MTP=1           MTP self-speculative decoding with 1 draft token (the head stays BF16 in our checkpoints)
#   SPEC='{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4}'   any --speculative-config
#   SSM_BF16=0      keep the DeltaNet SSM state in FP32 (default for the quantized candidates: BF16, -3..6% J/token)
# Quantized checkpoints come from quant/make_candidates.sh; w4a16-cmkl is CMKL's (only if external checkpoints are allowed).
set -euo pipefail
here=$(dirname "$0")
source "$here/env.sh"
name=${1:?usage: serving/candidate.sh <bf16|fp8|w8a8|w8a8-gdn|w8a16|w4a16-cmkl> [extra vllm args]}
shift
flags=()
[[ ${PREFIX:-0} == 1 ]] && flags+=(--enable-prefix-caching)
[[ -n ${MAX_SEQS:-} ]] && flags+=(--max-num-seqs "$MAX_SEQS")
[[ -n ${BATCHED:-} ]] && flags+=(--max-num-batched-tokens "$BATCHED")
[[ -n ${LONG_PREFILL:-} ]] && flags+=(--long-prefill-token-threshold "$LONG_PREFILL")
[[ -n ${MTP:-} ]] && flags+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $MTP}")
[[ -n ${SPEC:-} ]] && flags+=(--speculative-config "$SPEC")
ssm=(--mamba-ssm-cache-dtype bfloat16)
[[ ${SSM_BF16:-1} == 0 ]] && ssm=()
case $name in
  bf16) exec "$here/vllm_bf16.sh" "${flags[@]}" "$@" ;;   # the baseline stays as measured: FP32 SSM state
  fp8)  exec "$here/vllm_bf16.sh" --quantization fp8 "${ssm[@]}" "${flags[@]}" "$@" ;;
  w8a8|w8a8-gdn|w8a16|w8a8-gdn-head|w4a16-gdn)
    model=$WORK/models/mango-$name
    [[ -f $model/config.json ]] || { echo "missing $model: run quant/make_candidates.sh $name" >&2; exit 1; }
    MODEL_W4A16=$model exec "$here/vllm_w4a16.sh" "${ssm[@]}" "${flags[@]}" "$@" ;;
  w4a16-cmkl) exec "$here/vllm_w4a16.sh" "${ssm[@]}" "${flags[@]}" "$@" ;;
  *) echo "unknown candidate '$name'" >&2; exit 2 ;;
esac
