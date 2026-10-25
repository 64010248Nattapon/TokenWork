#!/usr/bin/env bash
# Environment setup on the GPU machine (Colab or APEX). Run each step in tmux;
# nothing here is interactive. Layout under $WORK as in CLAUDE.md.
#   serving/setup_vm.sh check            # GPU, disk, Python, what is already installed
#   serving/setup_vm.sh venv             # $WORK/.venv with vLLM + lm-eval + nvidia-ml-py
#   serving/setup_vm.sh venv-quant       # $WORK/.venv-quant with llm-compressor (pins an older torch)
#   serving/setup_vm.sh download         # BF16 model into $HF_HOME (~18 GB)
#   serving/setup_vm.sh download-w4a16   # pre-quantized checkpoint (~8 GB); event days only if the rules allow it
set -euo pipefail
source "$(dirname "$0")/env.sh"
VLLM_VERSION=${VLLM_VERSION:-0.27.1}         # model card: validated on 0.24.0-0.27.1
LLMCOMPRESSOR_VERSION=${LLMCOMPRESSOR_VERSION:-0.12.0}   # what CMKL used
VENV=${VENV:-$WORK/.venv}
VENV_QUANT=${VENV_QUANT:-$WORK/.venv-quant}

need_uv() {
  command -v uv >/dev/null || python3 -m pip install -q uv
}

case ${1:-check} in
  check)
    mkdir -p "$WORK/logs" "$HF_HOME" "$WORK/models"
    nvidia-smi || echo "WARNING: nvidia-smi failed"
    df -h "$WORK"
    python3 --version
    [[ -x $VENV/bin/python ]] && "$VENV/bin/python" -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null \
      || echo "vllm: not installed in $VENV"
    command -v uv >/dev/null && echo "uv: $(uv --version)" || echo "uv: not installed"
    curl -sI --max-time 10 https://huggingface.co >/dev/null && echo "huggingface.co: reachable" || echo "huggingface.co: NOT reachable"
    ;;
  venv)
    need_uv
    [[ -x $VENV/bin/python ]] || uv venv --python 3.12 "$VENV"
    # --python, not VIRTUAL_ENV: UV_SYSTEM_PYTHON (set on Colab) would win over it.
    # --torch-backend=auto picks the torch build that matches the installed CUDA driver.
    uv pip install --python "$VENV/bin/python" --torch-backend=auto \
      "vllm==$VLLM_VERSION" nvidia-ml-py "lm_eval[api]"
    "$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    echo "Activate with: source $VENV/bin/activate"
    ;;
  venv-quant)
    # Separate venv: llm-compressor 0.12.0 resolves to an older torch than vLLM 0.27.1
    # (2.12 vs 2.13 on 2026-10-23), so installing it into $VENV would break vLLM.
    need_uv
    [[ -x $VENV_QUANT/bin/python ]] || uv venv --python 3.12 "$VENV_QUANT"
    uv pip install --python "$VENV_QUANT/bin/python" --torch-backend=auto "llmcompressor==$LLMCOMPRESSOR_VERSION"
    "$VENV_QUANT/bin/python" -c "import llmcompressor, transformers, torch; print('llmcompressor', llmcompressor.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)"
    ;;
  download|download-w4a16)
    model=$MODEL_BF16
    [[ $1 == download-w4a16 ]] && model=$MODEL_W4A16
    mkdir -p "$HF_HOME"
    df -h "$HF_HOME"
    if command -v hf >/dev/null; then
      hf download "$model"
    elif [[ -x $VENV/bin/hf ]]; then
      "$VENV/bin/hf" download "$model"
    else
      huggingface-cli download "$model"
    fi
    ;;
  *)
    echo "usage: $0 check | venv | venv-quant | download | download-w4a16" >&2
    exit 2
    ;;
esac
