#!/usr/bin/env python3
"""Run official hybrid forwards and retain every next-token logits tensor."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time


EXPECTED_CHECKPOINT = {
    "config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "model.safetensors-00001-of-00001.safetensors": "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c",
    "model.safetensors.index.json": "ce9a885efdf27d3664fdef5d512ad365216f1074051ef840c7cd8e5431495d0a",
    "tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    "tokenizer_config.json": "e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de",
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor) -> str:
    raw = tensor.detach().float().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def classify_layer(has_linear: bool, has_full: bool) -> str:
    if has_linear and not has_full:
        return "linear_attention"
    if has_full and not has_linear:
        return "full_attention"
    return "ambiguous"


def run(args) -> dict:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = {}
    for name, expected in EXPECTED_CHECKPOINT.items():
        path = args.model / name
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"checkpoint drift: {name}: {actual} != {expected}")
        checkpoint[name] = {"sha256": actual, "bytes": path.stat().st_size}

    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    text_config = config["text_config"]
    declared = text_config["layer_types"]
    if len(declared) != 24:
        raise ValueError(f"unexpected declared layer count: {len(declared)}")

    rows = []
    seen = set()
    for line in args.prompts.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["task_id"] in seen:
            continue
        seen.add(row["task_id"])
        rows.append(row)
        if len(rows) == 24:
            break
    if len(rows) != 24:
        raise ValueError("frozen panel has fewer than 24 unique prompts")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    physical = []
    for index, (layer, declared_type) in enumerate(zip(model.model.layers, declared, strict=True)):
        has_linear = hasattr(layer, "linear_attn") and layer.linear_attn is not None
        has_full = hasattr(layer, "self_attn") and layer.self_attn is not None
        actual = classify_layer(has_linear, has_full)
        physical.append({
            "index": index,
            "declared": declared_type,
            "actual": actual,
            "match": declared_type == actual,
            "layer_class": type(layer).__name__,
            "attention_class": type(layer.linear_attn if has_linear else layer.self_attn).__name__ if actual != "ambiguous" else None,
        })

    samples = []
    retained = {}
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for index, row in enumerate(rows):
        encoded = tokenizer(row["prompt"], add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to("cuda")
        attention_mask = encoded["attention_mask"].to("cuda")
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, -1, :]
        cpu_logits = logits.detach().contiguous().cpu()
        key = f"logits_{index:03d}"
        retained[key] = cpu_logits
        samples.append({
            "task_id": row["task_id"],
            "prompt_tokens": int(input_ids.shape[1]),
            "logits_key": key,
            "logits_shape": list(cpu_logits.shape),
            "logits_dtype": str(cpu_logits.dtype),
            "argmax_token": int(cpu_logits.argmax(dim=-1).item()),
            "logits_sha256": tensor_sha256(cpu_logits),
            "finite": bool(torch.isfinite(cpu_logits).all().item()),
        })
    save_file(retained, str(args.logits_output), metadata={"schema": "slx11-retained-logits-v1"})
    elapsed = time.monotonic() - started
    return {
        "schema": "slx11-official-hybrid-worker-v2",
        "model": str(args.model),
        "checkpoint": checkpoint,
        "prompt_file_sha256": sha256_file(args.prompts),
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "text_model_type": text_config.get("model_type"),
        "full_attention_interval": text_config.get("full_attention_interval"),
        "physical_layers": physical,
        "samples": samples,
        "logits_bundle": {
            "path": str(args.logits_output),
            "sha256": sha256_file(args.logits_output),
            "bytes": args.logits_output.stat().st_size,
            "tensors": len(retained),
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
            "forward_elapsed_seconds": elapsed,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--prompts", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--logits-output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(result["samples"]), "logits_bundle": result["logits_bundle"], "hardware": result["hardware"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
