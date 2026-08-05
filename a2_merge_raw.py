#!/usr/bin/env python3
"""Full-rank task-arithmetic merge, done directly on safetensors -- no mergekit.

    W_out[k] = W_fable[k] + lambda * (W_tc[k] - W_base[k])   for every tensor k

We do this ourselves because (a) mergekit hit a pydantic/arch-forward-ref bug on the novel
qwen3_5 architecture, and (b) the operation is trivial and the three checkpoints share an
IDENTICAL tensor keyset (verified: 1199 common, 0 extra on any side), so a per-tensor stream
needs no architecture understanding at all. Arithmetic is done in fp32 and cast back to the
tensor's original dtype; tensors are streamed one at a time (peak RAM = the largest single
tensor x3, ~15 GB for the embedding), so a 27B x3 merge fits in 64 GB.

Non-weight files (config.json, tokenizer, chat template, generation config) are copied from
FABLE -- the merge keeps Fable's identity/tokenizer; only the weights move toward concision.

    python a2_merge_raw.py --base .../fp16/base --tc .../fp16/tc --fable .../fp16/fable \
                           --lam 0.7 --out .../merges/fable-tc-l0.7-fp16
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SHARD_BYTES = 5 * 1024**3   # ~5 GB shards, matching typical HF sharding


def key_index(model_dir: str) -> dict[str, str]:
    """map tensor-key -> shard file, across all *.safetensors in a dir."""
    idx: dict[str, str] = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                idx[k] = f
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tc", required=True)
    ap.add_argument("--fable", required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ib = key_index(args.base)
    it = key_index(args.tc)
    iff = key_index(args.fable)
    keys = list(iff)                     # iterate Fable's keys (the identity we keep)
    missing = [k for k in keys if k not in ib or k not in it]
    if missing:
        raise SystemExit(f"{len(missing)} Fable keys absent from base/tc, e.g. {missing[:3]} "
                         f"-- task arithmetic undefined for those; aborting")

    # cache one open handle per shard file to avoid reopening
    handles: dict[str, object] = {}

    def get(idx: dict[str, str], k: str) -> torch.Tensor:
        f = idx[k]
        h = handles.get(f)
        if h is None:
            h = handles[f] = safe_open(f, framework="pt").__enter__()
        return h.get_tensor(k)

    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    shard_no = 0
    weight_map: dict[str, str] = {}
    total = len(keys)
    lam = args.lam

    def flush():
        nonlocal shard, shard_bytes, shard_no
        if not shard:
            return
        shard_no += 1
        name = f"model-{shard_no:05d}.safetensors"
        save_file(shard, str(out / name), metadata={"format": "pt"})
        for kk in shard:
            weight_map[kk] = name
        print(f"  wrote {name}  ({shard_bytes/1024**3:.1f} GB, {len(shard)} tensors)", flush=True)
        shard = {}
        shard_bytes = 0

    for i, k in enumerate(keys):
        wf = get(iff, k)
        wt = get(it, k)
        wb = get(ib, k)
        orig_dtype = wf.dtype
        # fp32 arithmetic; delta only where shapes agree (they do -- identical keyset/shapes).
        merged = (wf.float() + lam * (wt.float() - wb.float())).to(orig_dtype).contiguous()
        shard[k] = merged
        shard_bytes += merged.numel() * merged.element_size()
        if shard_bytes >= SHARD_BYTES:
            flush()
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{total} tensors", flush=True)
    flush()

    # sharded-safetensors index
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(
            os.path.getsize(out / n) for n in set(weight_map.values()))},
         "weight_map": weight_map}, indent=2))

    # carry Fable's config/tokenizer/template so convert_hf_to_gguf has what it needs
    src = pathlib.Path(args.fable)
    for f in src.iterdir():
        if f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
            if f.is_file():
                shutil.copy2(f, out / f.name)
    print(f"merged {total} tensors into {shard_no} shards + config -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
