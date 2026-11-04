#!/usr/bin/env python3
"""Bounded qualification of the official RWKV7 1.5B release runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


PROMPTS = [
    "Reply with only the integer result of 17 + 25.",
    "Write a Python function named add that returns the sum of two numbers.",
    "In one sentence, explain why a recurrent model can use constant-size state.",
    "Reply with exactly OK.",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_release(model_dir: Path):
    package_name = "_rwkv7_qualification_release"
    package = ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = [str(model_dir / "inference")]
    sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.model_loader")


def tensors(value: Any, seen: set[int] | None = None) -> list[torch.Tensor]:
    if seen is None:
        seen = set()
    if value is None or id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        out: list[torch.Tensor] = []
        for item in value.values():
            out.extend(tensors(item, seen))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(tensors(item, seen))
        return out
    if hasattr(value, "__dict__"):
        return tensors(vars(value), seen)
    return []


def storage_bytes(value: Any) -> int:
    unique: dict[tuple[str, int], int] = {}
    for tensor in tensors(value):
        storage = tensor.untyped_storage()
        unique[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(unique.values())


def fresh_forward(model: Any, ids: torch.Tensor):
    mask = torch.ones_like(ids)
    return model(input_ids=ids, attention_mask=mask, use_cache=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="auto", choices=("auto", "torch", "tilelang"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.model / "release-manifest.json").read_text())
    required = ("model.safetensors", "config.json", "tokenizer.json")
    identities = {}
    for name in required:
        actual = sha256(args.model / name)
        expected = manifest["files"][name]["sha256"]
        identities[name] = {"actual": actual, "expected": expected, "match": actual == expected}
    identity_pass = all(item["match"] for item in identities.values())
    if not identity_pass:
        raise RuntimeError("artifact identity gate failed")

    torch.manual_seed(33377335)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    started = time.perf_counter()
    loader = load_release(args.model)
    model, tokenizer = loader.load_model_and_tokenizer(
        str(args.model), device="cuda", dtype=torch.bfloat16,
        backend=args.backend, state_dtype="float32"
    )
    model.set_kernel_backend(args.backend)
    if args.backend == "tilelang":
        model.prepare_inference_weights()
    load_seconds = time.perf_counter() - started
    free_after, _ = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    peak_allocated = torch.cuda.max_memory_allocated()

    backend_status = None
    try:
        backend_status = model.kernel_backend_status()
    except (AttributeError, TypeError):
        backend_status = {"requested": args.backend}

    base_text = "RWKV recurrent state qualification. " * 2000
    long_ids = tokenizer(base_text, return_tensors="pt").input_ids[:, :1024].to("cuda")
    if long_ids.shape[1] < 1024:
        raise RuntimeError("tokenizer did not produce 1024 tokens")

    state_sizes = {}
    for length in (32, 256, 1024):
        result = fresh_forward(model, long_ids[:, :length])
        state_sizes[str(length)] = storage_bytes(result.past_key_values)
        del result
    constant_state_pass = len(set(state_sizes.values())) == 1

    parity_ids = long_ids[:, :128]
    with torch.inference_mode():
        full = fresh_forward(model, parity_ids)
        first = fresh_forward(model, parity_ids[:, :64])
        second = model(
            input_ids=parity_ids[:, 64:],
            attention_mask=torch.ones_like(parity_ids[:, 64:]),
            past_key_values=first.past_key_values,
            use_cache=True,
        )
        fresh_again = fresh_forward(model, parity_ids)
    full_suffix = full.logits[:, 64:].float()
    split_diff = float((full_suffix - second.logits.float()).abs().max().item())
    reset_diff = float((full.logits.float() - fresh_again.logits.float()).abs().max().item())

    contamination_prompt = tokenizer("Unrelated sequence about astronomy.", return_tensors="pt").input_ids.to("cuda")
    contam_seed = fresh_forward(model, contamination_prompt)
    contaminated = model(
        input_ids=parity_ids[:, 64:],
        attention_mask=torch.ones_like(parity_ids[:, 64:]),
        past_key_values=contam_seed.past_key_values,
        use_cache=True,
    )
    contamination_delta = float((second.logits.float() - contaminated.logits.float()).abs().max().item())

    behavior = []
    for prompt in PROMPTS:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, thinking=False, return_tensors="pt"
        )
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids.input_ids
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        generated = model.generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=128,
            do_sample=False,
            eos_token_id=0,
            pad_token_id=0,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        completion_ids = generated[0, prompt_ids.shape[1]:]
        behavior.append({
            "prompt": prompt,
            "completion": tokenizer.decode(completion_ids, skip_special_tokens=True).strip(),
            "new_tokens": int(completion_ids.numel()),
            "natural_eos": bool(completion_ids.numel() and int(completion_ids[-1]) == 0),
            "elapsed_seconds": elapsed,
            "tokens_per_second": float(completion_ids.numel() / elapsed),
        })

    split_pass = split_diff <= 5e-2
    reset_pass = reset_diff <= 5e-2 and contamination_delta > 5e-2
    fit_pass = free_after >= 4 * 1024**3
    mechanism_pass = identity_pass and fit_pass and constant_state_pass and split_pass and reset_pass
    report = {
        "schema_version": 1,
        "model": str(args.model),
        "backend_requested": args.backend,
        "backend_status": str(backend_status),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "total_vram_bytes": total,
        },
        "identity": identities,
        "load": {
            "seconds": load_seconds, "free_before_bytes": free_before,
            "free_after_bytes": free_after, "allocated_bytes": allocated,
            "peak_allocated_bytes": peak_allocated, "fit_pass": fit_pass,
        },
        "state_bytes_by_length": state_sizes,
        "constant_state_pass": constant_state_pass,
        "split_max_abs_diff": split_diff,
        "split_pass": split_pass,
        "fresh_reset_max_abs_diff": reset_diff,
        "contaminated_state_max_abs_diff": contamination_delta,
        "isolation_pass": reset_pass,
        "behavior": behavior,
        "nonempty_rate": sum(bool(item["completion"]) for item in behavior) / len(behavior),
        "natural_eos_rate": sum(item["natural_eos"] for item in behavior) / len(behavior),
        "mechanism_pass": mechanism_pass,
        "decision": "QUALIFIED_MECHANISM" if mechanism_pass else "REJECT_MECHANISM",
        "deployment_license": "BLOCKED_FOR_DEPLOYMENT_UNASSERTED_WEIGHT_LICENSE",
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not mechanism_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
