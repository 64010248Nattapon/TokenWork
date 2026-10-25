#!/usr/bin/env bash
# Serve one config for the grader (https://greenmind.cmkl.ai/) in tmux, run the grader's checks against it, and record
# the exact config in results/<YYYYMMDD-HHMM>_apex_grader-<name>/ for the tracker sheet. Run it on the VM, from
# ~/work/repo (from a laptop: ssh apex 'cd ~/work/repo && git pull -q && <command>').
#   serving/grader.sh start baseline                                   # the organizers' baseline (baseline_official.sh)
#   PREFIX=1 serving/grader.sh start w8a8-gdn                          # any candidate.sh name + its env toggles
#   serving/grader.sh start ~/models/MANGO1.5-Qwen3.5-9B-W4A16 --max-num-seqs 128   # any model dir or HF id (has a /)
#   SERVED_NAME=mango-w4 serving/grader.sh start w8a8-gdn              # the model name the grader's dropdown shows
#   POWER=150 serving/grader.sh start w8a8-gdn                         # or CLOCKS=900: held until stop, then reset
#   serving/grader.sh check | status | stop
# A model path goes through vllm_w4a16.sh as is (--language-model-only + REASONING_FLAGS + TOOL_FLAGS, nothing else);
# extra vllm args after the name go to `vllm serve` and the last copy of a repeated flag wins.
# Env: OWNER [shogun] for $WORK/GPU_LOCK; PORT [8000]; SERVED_NAME [mango]; REASONING_FLAGS [empty: the thinking streams
# as `content`, like the graded baseline; REASONING_FLAGS="--reasoning-parser qwen3" moves it to `reasoning`].
# start refuses while another process holds GPU memory or someone else holds the lock (no other GPU work in graded runs).
set -euo pipefail
cd "$(dirname "$0")/.."
export REASONING_FLAGS=${REASONING_FLAGS-}
source serving/env.sh
OWNER=${OWNER:-shogun}
SESSION=grader-serve
STATE=$WORK/grader.current
LOCK=$WORK/GPU_LOCK
endpoint="http://$(hostname -I | awk '{print $1}'):$PORT/v1"
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
lock_owner() { [[ -s $LOCK ]] && cut -d'|' -f1 "$LOCK" | xargs || true; }

check() {  # the grader's contract: /v1/models, streamed chat with token usage, a tool call with tool_choice auto
  python3 - "$PORT" "$SERVED_NAME" <<'EOF'
import json, sys, time, urllib.request
port, model = sys.argv[1], sys.argv[2]
base = f"http://localhost:{port}/v1"

def stream(body):
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0, **body}
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, first, fields, usage, calls, finish = time.time(), None, set(), None, {}, None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for ch in chunk.get("choices", []):
                d = ch.get("delta", {})
                finish = ch.get("finish_reason") or finish
                for f in ("content", "reasoning", "reasoning_content"):
                    if d.get(f):
                        fields.add(f)
                        first = first or (time.time() - t0, f)
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc["index"], {"name": "", "args": ""})
                    c["name"] += tc.get("function", {}).get("name") or ""
                    c["args"] += tc.get("function", {}).get("arguments") or ""
    return first, fields, usage, calls, finish

ok = True
ids = [m["id"] for m in json.load(urllib.request.urlopen(base + "/models", timeout=10))["data"]]
print(f"models: {ids}")
ok &= model in ids
first, fields, usage, _, finish = stream({"max_tokens": 64, "messages": [{"role": "user", "content": "สวัสดีครับ แนะนำตัวสั้นๆ"}]})
print(f"stream: first token after {first[0]:.2f} s in `{first[1]}`, fields {sorted(fields)}, usage {usage}, finish {finish}"
      if first else "stream: NO tokens")
ok &= bool(first) and bool(usage)
tool = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
_, _, usage, calls, finish = stream({"max_tokens": 2048, "tools": [tool], "tool_choice": "auto",
                                     "messages": [{"role": "user", "content": "What is the weather in Bangkok right now?"}]})
print(f"tool call: {[c['name'] + '(' + c['args'] + ')' for c in calls.values()]}, finish {finish}, usage {usage}")
ok &= any(c["name"] == "get_weather" for c in calls.values())
print("CHECK OK" if ok else "CHECK FAILED")
sys.exit(0 if ok else 1)
EOF
}

case ${1:-status} in
  start)
    name=${2:?usage: serving/grader.sh start <baseline | candidate.sh name | model path> [extra vllm args]}
    shift 2
    if tmux has-session -t "$SESSION" 2>/dev/null; then echo "already serving: $(cat "$STATE" 2>/dev/null)" >&2; exit 1; fi
    owner=$(lock_owner)
    if [[ -n $owner && $owner != "$OWNER" ]]; then echo "GPU locked: $(cat "$LOCK")" >&2; exit 1; fi
    if (( $(gpu_mib) > 1000 )); then
      echo "GPU busy ($(gpu_mib) MiB used); free it first:" >&2
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read -r pid mem; do
        echo "  $mem  $(ps -o user=,lstart=,cmd= -p "$pid" | cut -c1-120)" >&2; done
      exit 1
    fi
    case $name in
      baseline) cmd=(serving/baseline_official.sh) ;;
      */*)
        if [[ -d $name ]]; then
          [[ -f $name/config.json ]] || { echo "no config.json in $name" >&2; exit 1; }
          name=$(cd "$name" && pwd)
        fi
        export MODEL_W4A16=$name
        name=$(basename "$name")
        cmd=(serving/vllm_w4a16.sh) ;;
      *) cmd=(serving/candidate.sh "$name") ;;
    esac
    ts=$(date +%Y%m%d-%H%M)
    run=results/${ts}_apex_grader-$name
    log=$WORK/logs/serve-$name-$ts.log
    mkdir -p "$run" "$WORK/logs"
    {  # run.sh replays this exact server: every env toggle that was set, then the command
      echo '#!/usr/bin/env bash'
      echo "# $name for the grader, started $(date '+%F %T') by $OWNER"
      echo "cd $PWD"
      for v in PREFIX MAX_SEQS BATCHED LONG_PREFILL MTP SSM_BF16 MODEL_BF16 MODEL_W4A16 REASONING_FLAGS TOOL_FLAGS \
               MAX_MODEL_LEN GPU_MEM_UTIL PORT SERVED_NAME HF_HOME VLLM_USE_FLASHINFER_SAMPLER POWER CLOCKS; do
        [[ -v $v ]] && printf 'export %s=%q\n' "$v" "${!v}"
      done
      echo "source $WORK/.venv/bin/activate"
      printf 'exec'; printf ' %q' "${cmd[@]}" "$@"; echo
    } > "$run/run.sh"
    echo "$OWNER | grader serve $name | $(date +%H:%M) | until serving/grader.sh stop" > "$LOCK"
    printf 'name=%q\nrun=%q\nlog=%q\nstarted=%q\ntuned=%q\n' "$name" "$run" "$log" "$(date '+%F %T')" \
      "${POWER:+power $POWER W}${CLOCKS:+clocks $CLOCKS MHz}" > "$STATE"
    tmux new -d -s "$SESSION" "bash $PWD/$run/run.sh > $log 2>&1"
    echo "starting $name (log $log) ..."
    for _ in $(seq 1 180); do
      curl -sf -o /dev/null --max-time 3 "localhost:$PORT/health" && break
      if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "server exited during startup:" >&2; tail -n 25 "$log" >&2; rm -f "$STATE" "$LOCK"; exit 1
      fi
      sleep 5
    done
    curl -sf -o /dev/null --max-time 3 "localhost:$PORT/health" || { echo "not healthy after 15 min: $log" >&2; exit 1; }
    [[ -n ${POWER:-} ]] && ALLOW_GPU_TUNING=1 serving/gpu_tune.sh power "$POWER" > /dev/null
    [[ -n ${CLOCKS:-} ]] && ALLOW_GPU_TUNING=1 serving/gpu_tune.sh clocks "$CLOCKS" > /dev/null
    (
      "$WORK/.venv/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
      nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.sm,clocks.max.sm,persistence_mode --format=csv
      "$HOME/.local/bin/uv" pip freeze --python "$WORK/.venv/bin/python" 2>/dev/null
    ) > "$run/env.txt" || true
    grep -m1 -o "non-default args: .*" "$log" > "$run/config.txt" || true
    check | tee "$run/check.txt" || true
    echo
    echo "config:   $(cat "$run/config.txt")"
    echo "gpu:      $(nvidia-smi --query-gpu=power.limit,clocks.sm --format=csv,noheader)"
    echo "endpoint: $endpoint   model: $SERVED_NAME"
    echo "next:     grader Settings -> endpoint + model, Health check, Submit; then serving/grader.sh stop"
    ;;
  check) check ;;
  status)
    [[ -f $STATE ]] && cat "$STATE" || echo "no grader server recorded"
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux $SESSION: running" || echo "tmux $SESSION: not running"
    curl -sf -o /dev/null --max-time 3 "localhost:$PORT/health" && echo "health: ok ($endpoint)" || echo "health: down"
    nvidia-smi --query-gpu=memory.used,power.draw,power.limit,clocks.sm --format=csv,noheader
    echo "lock: $(cat "$LOCK" 2>/dev/null || echo free)"
    ;;
  stop)
    [[ -f $STATE ]] && source "$STATE"
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "no $SESSION session"
    for _ in $(seq 1 30); do (( $(gpu_mib) < 1000 )) && break; sleep 2; done
    [[ -n ${tuned:-} ]] && ALLOW_GPU_TUNING=1 serving/gpu_tune.sh reset > /dev/null && echo "GPU power/clocks reset"
    if [[ -n ${log:-} && -f $log && -n ${run:-} ]]; then  # what the grader sent, for the sheet's notes
      {
        echo "requests: $(grep -cE '"POST /v1/chat/completions' "$log" || true) chat completions"
        { grep -oE 'Running: [0-9]+ reqs, Waiting: [0-9]+ reqs' "$log" || true; } | awk '{r = $2 > r ? $2 : r; w = $5 > w ? $5 : w}
          END {print "max running:", r + 0, " max waiting:", w + 0}'
        { grep -oE 'Prefix cache hit rate: [0-9.]+%' "$log" || true; } | tail -1
      } | tee "$run/served.txt"
    fi
    [[ $(lock_owner) == "$OWNER" ]] && rm -f "$LOCK"
    rm -f "$STATE"
    echo "GPU: $(gpu_mib) MiB used"
    ;;
  *) echo "usage: serving/grader.sh start <name> [vllm args] | check | status | stop" >&2; exit 2 ;;
esac
