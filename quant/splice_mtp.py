#!/usr/bin/env python3
"""Copy the MTP (NextN) head from the BF16 checkpoint into a quantized one.

transformers does not load the `mtp.*` weights, so a checkpoint saved by
llm-compressor has no MTP head and vLLM's MTP speculative decoding cannot use
it. This copies the 15 `mtp.*` tensors unchanged (BF16) into
<dst>/model-mtp.safetensors, adds them to the weight index, and adds
"re:.*mtp.*" to quantization_config.ignore so vLLM builds the MTP layers as
plain (unquantized) Linear layers. Same fix as CMKL's -mtp checkpoint.

  python quant/splice_mtp.py --src CMKL/MANGO1.5-Qwen3.5-9B --dst $WORK/models/mango-w4a16

Stdlib only: it copies raw tensor bytes, so no torch or safetensors needed.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

MTP_FILE = "model-mtp.safetensors"


def read_header(path: Path) -> tuple[dict, int]:
    """Return (header, byte offset where tensor data starts)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def resolve(src: str) -> Path:
    p = Path(src).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download  # only needed for a Hub id
    return Path(snapshot_download(src, allow_patterns=["*.json", "*.safetensors"]))


def weight_map(model_dir: Path) -> dict[str, str]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    single = model_dir / "model.safetensors"
    if single.exists():
        header, _ = read_header(single)
        return {k: single.name for k in header if k != "__metadata__"}
    raise FileNotFoundError(f"no safetensors weights in {model_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="BF16 model dir or Hub id")
    ap.add_argument("--dst", required=True, help="quantized model dir (modified in place)")
    args = ap.parse_args()
    src, dst = resolve(args.src), Path(args.dst).expanduser()

    src_map = weight_map(src)
    mtp = sorted(k for k in src_map if k.startswith("mtp."))
    if not mtp:
        print(f"error: no mtp.* tensors in {src}", file=sys.stderr)
        return 1
    dst_map = weight_map(dst)
    if any(k.startswith("mtp.") for k in dst_map):
        print(f"{dst} already has mtp.* tensors; nothing to do")
        return 0

    # Gather raw bytes per tensor from the source shards.
    tensors: dict[str, tuple[dict, bytes]] = {}
    for shard in sorted({src_map[k] for k in mtp}):
        header, data_start = read_header(src / shard)
        with open(src / shard, "rb") as f:
            for k in mtp:
                if src_map[k] != shard:
                    continue
                meta = header[k]
                start, end = meta["data_offsets"]
                f.seek(data_start + start)
                tensors[k] = ({"dtype": meta["dtype"], "shape": meta["shape"]}, f.read(end - start))

    # Write them as one new safetensors file.
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for k, (meta, raw) in tensors.items():
        header[k] = {**meta, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    hbytes = json.dumps(header, separators=(",", ":")).encode()
    hbytes += b" " * (-len(hbytes) % 8)  # keep tensor data 8-byte aligned
    with open(dst / MTP_FILE, "wb") as f:
        f.write(struct.pack("<Q", len(hbytes)) + hbytes)
        for _, raw in tensors.values():
            f.write(raw)

    # Index: create one if the checkpoint was a single file, then add the MTP tensors.
    total = sum((dst / name).stat().st_size for name in set(dst_map.values()))
    total += (dst / MTP_FILE).stat().st_size
    new_map = {**dst_map, **{k: MTP_FILE for k in tensors}}
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": dict(sorted(new_map.items()))}, indent=2))

    cfg_path = dst / "config.json"
    cfg = json.loads(cfg_path.read_text())
    qc = cfg.get("quantization_config")
    if qc is not None and "re:.*mtp.*" not in qc.setdefault("ignore", []):
        qc["ignore"].append("re:.*mtp.*")
        cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"spliced {len(tensors)} mtp tensors ({offset / 2**20:.0f} MiB) into {dst / MTP_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
