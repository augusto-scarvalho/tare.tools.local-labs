#!/usr/bin/env python3
"""Fresh physical topology and forward checks for official Qwen3.5 hybrid model."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor) -> str:
    return hashlib.sha256(tensor.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def run(args) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_path = args.model / "config.json"
    tensor_path = args.model / "model.safetensors-00001-of-00001.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
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
        str(args.model), local_files_only=True, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="sdpa"
    ).eval()
    layers = model.model.layers
    physical = []
    for index, (layer, declared_type) in enumerate(zip(layers, declared, strict=True)):
        has_linear = hasattr(layer, "linear_attn") and layer.linear_attn is not None
        has_full = hasattr(layer, "self_attn") and layer.self_attn is not None
        actual = "linear_attention" if has_linear and not has_full else "full_attention" if has_full and not has_linear else "ambiguous"
        physical.append({
            "index": index,
            "declared": declared_type,
            "actual": actual,
            "match": declared_type == actual,
            "layer_class": type(layer).__name__,
            "attention_class": type(layer.linear_attn if has_linear else layer.self_attn).__name__ if actual != "ambiguous" else None,
        })

    samples = []
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for row in rows:
        encoded = tokenizer(row["prompt"], add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to("cuda")
        attention_mask = encoded["attention_mask"].to("cuda")
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, -1, :]
        finite = bool(torch.isfinite(logits).all().item())
        samples.append({
            "task_id": row["task_id"],
            "prompt_tokens": int(input_ids.shape[1]),
            "argmax_token": int(logits.argmax(dim=-1).item()),
            "logits_sha256": tensor_sha256(logits),
            "finite": finite,
        })
    elapsed = time.monotonic() - started
    return {
        "schema": "slx11-official-hybrid-worker-v1",
        "model": str(args.model),
        "config_sha256": sha256_file(config_path),
        "tensor_sha256": sha256_file(tensor_path),
        "prompt_file_sha256": sha256_file(args.prompts),
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "text_model_type": text_config.get("model_type"),
        "full_attention_interval": text_config.get("full_attention_interval"),
        "physical_layers": physical,
        "samples": samples,
        "metrics": {
            "official_checkpoint_identified": int(config.get("model_type") == "qwen3_5" and text_config.get("model_type") == "qwen3_5_text"),
            "hybrid_layer_types_verified": sum(row["match"] for row in physical),
            "physical_recurrent_layers": sum(row["actual"] == "linear_attention" for row in physical),
            "physical_full_attention_layers": sum(row["actual"] == "full_attention" for row in physical),
            "successful_live_forwards": len(samples),
            "finite_output_rate": sum(row["finite"] for row in samples) / len(samples),
        },
        "hardware": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda, "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3, "forward_elapsed_seconds": elapsed},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--prompts", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": result["metrics"], "hardware": result["hardware"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
