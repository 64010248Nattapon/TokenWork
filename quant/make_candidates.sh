#!/usr/bin/env bash
# Build our quantized candidate checkpoints into $WORK/models (data-free, about 1 min each, ~12 GB each).
#   quant/make_candidates.sh                 # w8a8-gdn (the quality-safe INT8 default)
#   quant/make_candidates.sh w8a8 w8a8-gdn w8a16
# w8a8      INT8 weights + activations, everything quantized          (fastest; -1.2 pts lm-eval on Colab)
# w8a8-gdn  W8A8 with the Gated DeltaNet projections kept in BF16      (+0.3 pts; -20..29% J/token vs BF16)
# w8a16     INT8 weights only                                          (+0.3 pts; gains only at c <= 16)
# w8a8-gdn-head  w8a8-gdn with lm_head INT8 too (248K-vocab head: ~1 GB less read per decode step)
# w4a16-gdn INT4 weights (round-to-nearest, group 128) except DeltaNet: fewer bytes for the low-batch phases
# Needs $WORK/.venv-quant (serving/setup_vm.sh venv-quant) and the BF16 model in $HF_HOME.
set -euo pipefail
cd "$(dirname "$0")/.."
source serving/env.sh
source "$WORK/.venv-quant/bin/activate"
mkdir -p "$WORK/models"
for name in "${@:-w8a8-gdn}"; do
  out=$WORK/models/mango-$name
  if [[ -f $out/config.json ]]; then echo "$name: $out exists, skipping"; continue; fi
  free_gb=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc 0-9)
  if (( free_gb < 15 )); then echo "$name: only ${free_gb} GB free on $WORK, need ~15 GB" >&2; exit 1; fi
  case $name in
    w8a8)     args=(--scheme W8A8) ;;
    w8a8-gdn) args=(--scheme W8A8 --keep-bf16 're:.*linear_attn.*') ;;
    w8a16)    args=(--scheme W8A16) ;;
    w8a8-gdn-head) args=(--scheme W8A8 --keep-bf16 're:.*linear_attn.*' --quant-lm-head) ;;
    w4a16-gdn) args=(--scheme W4A16 --keep-bf16 're:.*linear_attn.*') ;;   # RTN, no calibration
    *) echo "unknown candidate '$name' (w8a8 | w8a8-gdn | w8a16 | w8a8-gdn-head | w4a16-gdn)" >&2; exit 2 ;;
  esac
  echo "$name: quantizing to $out"
  python quant/w8a8_int8.py --model "$MODEL_BF16" --out "$out" "${args[@]}"
done
