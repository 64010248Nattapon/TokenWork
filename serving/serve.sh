#!/usr/bin/env bash
# Serve the quantized MANGO model with vLLM (OpenAI-compatible API on port 8000).
#
#   bash serve.sh start     start in the background (log: vllm_server.log)
#   bash serve.sh status    is it up? + GPU usage
#   bash serve.sh logs      follow the log (Ctrl+C to stop following)
#   bash serve.sh stop      stop the server and free the GPU
#
# Edit the settings below; use the best values found by sweep_serve.py.

# ======================= Settings =======================
ENV=~/envs/ohm                                    # venv that has vllm installed
MODEL=~/nut/hf/hub/models--Nattapon248--mango15-sq-gptq-w8a8/snapshots/1ce063a2d48c1b4cfcc7981730d0429297e5af58         # quantized model folder
NAME=nico007                                      # model name clients send in requests
PORT=8001
MAX_MODEL_LEN=40000                               # max prompt + answer tokens per request
MAX_NUM_SEQS=256                                 # max requests processed together
MAX_BATCHED_TOKENS=8196                        # tokensd per engine step (prefill budget)
LONG_PREFILL=2048                                  # >0: cap prefill chunk of long prompts (e.g. 2048)
GPU_MEM=0.90                                      # share of GPU memory vLLM may reserve
KV_FP8=0                                         # 1 = FP8 KV cache (more batch room)
MTP=0                                             # 1 = speculative decoding with MTP layers
# ========================================================

LOG=vllm_server.log
PIDFILE=vllm_server.pid

start() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
        echo "Already running (PID $(cat $PIDFILE))"; return
    fi
    source "$ENV/bin/activate"
    ARGS=(
        "$MODEL"
        --served-model-name "$NAME"
        --port "$PORT"
        --language-model-only
        --max-model-len "$MAX_MODEL_LEN"
        --max-num-seqs "$MAX_NUM_SEQS"
        --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
        --gpu-memory-utilization "$GPU_MEM"
        --reasoning-parser qwen3
	--reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "\n\nI have to give the solution based on the reasoning directly now.\n</think>"}'
        --mamba-ssm-cache-dtype bfloat16
	--enable-auto-tool-choice
	--tool-call-parser qwen3_coder
        --default-chat-template-kwargs '{"enable_thinking": true}'
   )
    [ "$LONG_PREFILL" -gt 0 ] && ARGS+=(--long-prefill-token-threshold "$LONG_PREFILL")
    [ "$KV_FP8" = 1 ] && ARGS+=(--kv-cache-dtype fp8)
    [ "$MTP" = 1 ] && ARGS+=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}')

    # No CUDA toolkit (nvcc) on this server -> FlashInfer can't JIT-compile its kernels.
    # Use vLLM's built-in PyTorch/Triton sampler instead of FlashInfer's.
    export VLLM_USE_FLASHINFER_SAMPLER=0

    echo "Starting: vllm serve ${ARGS[*]}"
    nohup vllm serve "${ARGS[@]}" > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"

    echo -n "Waiting for the server"
    for _ in $(seq 1 180); do
        if curl -s "localhost:$PORT/v1/models" > /dev/null; then
            echo -e "\nReady: http://localhost:$PORT/v1  (model name: $NAME)"; return
        fi
        if ! kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
            echo -e "\nServer exited. Likely root cause from $LOG:"
            grep -E "Error|error:|Exception|CUDA out of memory|Free memory|not supported|No such file" "$LOG" \
                | grep -v -E "Engine core initialization failed|wait_for_engine_startup" | tail -n 15
            echo "(full log: $LOG)"; return 1
        fi
        echo -n "."; sleep 5
    done
    echo -e "\nStill not ready after 15 min, check: tail -f $LOG"
}

stop() {
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        kill -INT "$PID" 2>/dev/null
        for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
        kill -9 "$PID" 2>/dev/null
        rm -f "$PIDFILE"
        echo "Stopped."
    else
        echo "No PID file; not started by this script."
    fi
}

status() {
    if curl -s "localhost:$PORT/v1/models" > /dev/null; then
        echo "UP on port $PORT"
    else
        echo "DOWN"
    fi
    nvidia-smi --query-gpu=memory.used,memory.total,power.draw,utilization.gpu --format=csv
}

case "$1" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    logs) tail -f "$LOG" ;;
    *) echo "Usage: bash serve.sh {start|stop|status|logs}" ;;
esac
