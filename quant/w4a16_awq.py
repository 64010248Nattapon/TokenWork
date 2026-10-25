#!/usr/bin/env python3
"""Reproduce CMKL's W4A16 AWQ checkpoint with our own calibration data.

Use this when the rules forbid downloading pre-quantized checkpoints. Recipe:
quant/recipe_w4a16_awq.yaml (CMKL's, verbatim). Calibration: Thai and English
chat data at the model's 65/35 training mix, formatted with its chat template:
  English: HuggingFaceH4/ultrachat_200k (train_sft, MIT)
  Thai:    airesearch/WangchanThaiInstruct (train, CC BY-SA 4.0)
CMKL calibrated on 512 samples of their private SFT mix at 2048 tokens.

Run in its own venv (llm-compressor pins torch/transformers versions that can
clash with vLLM), on an idle GPU, inside tmux:
  serving/setup_vm.sh venv-quant           # $WORK/.venv-quant: llmcompressor 0.12.0 (CMKL's version)
  source $WORK/.venv-quant/bin/activate && source serving/env.sh    # HF_HOME=$WORK/hf
  python quant/w4a16_awq.py --out $WORK/models/mango-w4a16
Smoke test of the whole pipeline (minutes, not for real use):
  python quant/w4a16_awq.py --out /tmp/mango-w4a16-smoke --num-samples 32 --max-seq-len 512

Then serve it:  MODEL_W4A16=$WORK/models/mango-w4a16 ./serving/vllm_w4a16.sh
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def calibration_set(tokenizer, n: int, thai_share: float, max_len: int, seed: int):
    from datasets import concatenate_datasets, load_dataset

    n_th = round(n * thai_share)
    n_en = n - n_th
    en = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=seed).select(range(n_en))
    th = load_dataset("airesearch/WangchanThaiInstruct", split="train").shuffle(seed=seed).select(range(n_th))

    def en_text(ex):
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}

    def th_text(ex):
        user = ex["Instruction"]
        if ex["Input"] and ex["Input"].strip() not in ("", "None"):
            user += "\n\n" + ex["Input"]
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": ex["Output"]}]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

    ds = concatenate_datasets([
        en.map(en_text, remove_columns=en.column_names),
        th.map(th_text, remove_columns=th.column_names),
    ]).shuffle(seed=seed)
    return ds.map(lambda ex: tokenizer(ex["text"], max_length=max_len, truncation=True, add_special_tokens=False),
                  remove_columns=["text"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="CMKL/MANGO1.5-Qwen3.5-9B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipe", default=str(HERE / "recipe_w4a16_awq.yaml"))
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--thai-share", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mtp", action="store_true", help="skip copying the MTP head into the output")
    args = ap.parse_args()
    return quantize(args.model, args.out, args.recipe, args.num_samples, args.max_seq_len,
                    args.thai_share, args.seed, mtp=not args.no_mtp)


def quantize(model_id: str, out: str, recipe, num_samples: int, max_seq_len: int,
             thai_share: float, seed: int, mtp: bool = True) -> int:
    """Calibrate with llm-compressor `oneshot`, save compressed, splice the MTP head back in.
    `recipe` is a recipe file path or a list of modifiers. Also used by w8a8_gptq.py."""
    from huggingface_hub import snapshot_download
    from llmcompressor import oneshot
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ds = calibration_set(tokenizer, num_samples, thai_share, max_seq_len, seed)
    # The full multimodal class, like CMKL: the recipe's layer names and the saved
    # weight names (model.language_model.*) match what vLLM expects.
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, dtype="auto")
    # Pass the tokenizer as the processor: the model's AutoProcessor includes a
    # video processor that needs torchvision, which text calibration does not.
    oneshot(model=model, processor=tokenizer, dataset=ds, recipe=recipe,
            max_seq_length=max_seq_len, num_calibration_samples=num_samples)
    model.save_pretrained(out, save_compressed=True)
    tokenizer.save_pretrained(out)
    # Processor/preprocessor configs and the chat template, copied as-is.
    src = Path(model_id) if Path(model_id).is_dir() else Path(snapshot_download(model_id))
    for f in src.iterdir():
        if f.suffix in (".json", ".jinja") and "safetensors" not in f.name and not (Path(out) / f.name).exists():
            shutil.copy(f, Path(out) / f.name)
    print(f"saved {out} in {(time.time() - t0) / 60:.1f} min", flush=True)

    if mtp:
        return subprocess.call([sys.executable, str(HERE / "splice_mtp.py"), "--src", model_id, "--dst", out])
    return 0


if __name__ == "__main__":
    sys.exit(main())
