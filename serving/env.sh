# Shared settings for the serving scripts. Sourced, not executed.
# Override any of these per run, e.g.  MAX_MODEL_LEN=8192 ./serving/vllm_bf16.sh
# $WORK: /content/work on Colab, ~/work on APEX (see CLAUDE.md "GPU environments").
export WORK=${WORK:-$HOME/work}
export HF_HOME=${HF_HOME:-$WORK/hf}
export MODEL_BF16=${MODEL_BF16:-CMKL/MANGO1.5-Qwen3.5-9B}
# Pre-quantized checkpoint: on event days use it only if the rules allow external
# checkpoints. Otherwise point this at our own output from quant/, e.g. $WORK/models/mango-w4a16.
export MODEL_W4A16=${MODEL_W4A16:-CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp}
export SERVED_NAME=${SERVED_NAME:-mango}
export PORT=${PORT:-8000}
# Longest prompt + max output of the workload: prompts up to 32K + max_tokens 8192 (RULES.md [5]), with headroom.
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-49152}
# The grader needs tool calling with tool_choice auto (RULES.md [13], [14]).
# --enable-force-include-usage: usage in the last streamed chunk even when the client does not ask for it
# (the grader needs "streaming and token usage"; teammate nut's graded config had this flag).
export TOOL_FLAGS=${TOOL_FLAGS:---enable-auto-tool-choice --tool-call-parser qwen3_coder --enable-force-include-usage}
# REASONING_FLAGS= (empty) streams the thinking as `content`, like the organizers' baseline that the grader timed
# (its TTFT counts that first thinking token). Until the organizers confirm the grader also reads the `reasoning`
# field, submissions should run with REASONING_FLAGS= .
export REASONING_FLAGS=${REASONING_FLAGS---reasoning-parser qwen3}
# THINK_BUDGET=<n>: cap thinking at n tokens on the server (serving/vllm_launch.py); 0/unset = no cap.
if [[ -n ${THINK_BUDGET:-} && ${THINK_BUDGET} != 0 ]]; then
  export VLLM_CMD="python $(dirname "${BASH_SOURCE[0]}")/vllm_launch.py"
  [[ $REASONING_FLAGS == *--reasoning-config* ]] || REASONING_FLAGS="$REASONING_FLAGS --reasoning-config {\"reasoning_start_str\":\"<think>\",\"reasoning_end_str\":\"</think>\"}"
fi
export VLLM_CMD=${VLLM_CMD:-vllm}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
# APEX has no CUDA toolkit, so FlashInfer's JIT-built sampler fails at startup ("Could not find nvcc").
# The grader uses temperature 0 (greedy), so the PyTorch sampler costs nothing.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
