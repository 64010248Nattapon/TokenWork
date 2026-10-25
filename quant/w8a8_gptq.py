#!/usr/bin/env python3
"""W8A8 INT8 with calibration: SmoothQuant + GPTQ (llm-compressor's standard INT8 recipe).

The data-free `w8a8_int8.py` lost about 4 points on Thai math (mgsm th) on Colab.
This tries to win that back. SmoothQuant moves activation outliers into the weights,
and GPTQ rounds weights to minimise each layer's output error on Thai/English
calibration data. The SmoothQuant mappings are written out for both layer types,
like CMKL's AWQ recipe, because the default ones assume a plain transformer.

  serving/setup_vm.sh venv-quant && source $WORK/.venv-quant/bin/activate && source serving/env.sh
  python quant/w8a8_gptq.py --out $WORK/models/mango-w8a8-gptq
  python quant/w8a8_gptq.py --out ... --no-smoothquant      # GPTQ only, if SmoothQuant fails
  python quant/w8a8_gptq.py --out $WORK/models/mango-sqgptq-nut --nut   # nut's recipe (quant_meta.json on the VM)
  MODEL_W4A16=$WORK/models/mango-w8a8-gptq ./serving/vllm_w4a16.sh
"""
from __future__ import annotations

import argparse
import sys

from w4a16_awq import quantize

ATTN = r"(3|7|11|15|19|23|27|31)"                               # full-attention layers
GDN = r"(0|1|2|4|5|6|8|9|10|12|13|14|16|17|18|20|21|22|24|25|26|28|29|30)"   # Gated DeltaNet layers
IGNORE = ["lm_head", "re:.*visual.*", "re:.*mtp.*"]


def recipe(smoothquant: bool, strength: float, extra_ignore: tuple = (), actorder: str | None = None) -> list:
    try:  # module paths moved between llm-compressor releases
        from llmcompressor.modifiers.gptq import GPTQModifier
    except ImportError:
        from llmcompressor.modifiers.quantization import GPTQModifier
    gptq = {"actorder": actorder} if actorder else {}
    mods = [GPTQModifier(targets="Linear", scheme="W8A8", ignore=IGNORE + list(extra_ignore), **gptq)]
    if smoothquant:
        try:
            from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
        except ImportError:
            from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
        mappings = [
            [["re:.*self_attn.q_proj$", "re:.*self_attn.k_proj$", "re:.*self_attn.v_proj$"],
             rf"re:.*layers\.{ATTN}\.input_layernorm$"],
            [["re:.*linear_attn.in_proj_qkv$", "re:.*linear_attn.in_proj_z$",
              "re:.*linear_attn.in_proj_b$", "re:.*linear_attn.in_proj_a$"],
             rf"re:.*layers\.{GDN}\.input_layernorm$"],
            [["re:.*mlp.gate_proj$", "re:.*mlp.up_proj$"], "re:.*post_attention_layernorm$"],
        ]
        mods.insert(0, SmoothQuantModifier(smoothing_strength=strength, mappings=mappings, ignore=IGNORE))
    return mods


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="CMKL/MANGO1.5-Qwen3.5-9B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--thai-share", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-smoothquant", action="store_true")
    ap.add_argument("--smoothing-strength", type=float, default=0.8)
    ap.add_argument("--keep-bf16", nargs="*", default=[], help="extra GPTQ ignore patterns (kept BF16)")
    ap.add_argument("--actorder", choices=["static", "group", "weight"], help="GPTQ activation ordering")
    ap.add_argument("--nut", action="store_true",
                    help="teammate nut's Final recipe (S 92.3): 384 x 4096 samples, seed 42, in_proj_a/b BF16, actorder static")
    args = ap.parse_args()
    if args.nut:
        args.num_samples, args.max_seq_len, args.seed, args.actorder = 384, 4096, 42, "static"
        args.keep_bf16 += [r"re:.*linear_attn\.in_proj_a$", r"re:.*linear_attn\.in_proj_b$"]
    return quantize(args.model, args.out,
                    recipe(not args.no_smoothquant, args.smoothing_strength, tuple(args.keep_bf16), args.actorder),
                    args.num_samples, args.max_seq_len, args.thai_share, args.seed)


if __name__ == "__main__":
    sys.exit(main())
