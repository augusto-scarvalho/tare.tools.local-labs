#!/usr/bin/env python3
"""Frozen normal-question quality gate for the official RWKV7 release runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType

import torch

from normal_qa_ab import grade, load_tasks


EXPECTED_WEIGHT_SHA256 = "84ccbb857c84e00cefc48b233937ada79c411e491df25fb21aed23237f39a14f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_release(model_dir: Path):
    package_name = "_rwkv7_quality_release"
    package = ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = [str(model_dir / "inference")]
    sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.model_loader")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="auto", choices=("auto", "torch", "tilelang"))
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    task_sha = sha256(args.tasks)
    weight_sha = sha256(args.model / "model.safetensors")
    if weight_sha != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(f"weight identity mismatch: {weight_sha}")

    prior = []
    if args.output.exists():
        document = json.loads(args.output.read_text(encoding="utf-8"))
        if document.get("tasks_sha256") != task_sha or document.get("weight_sha256") != weight_sha:
            raise RuntimeError("resume identity mismatch")
        prior = document.get("results", [])
    done = {row["id"] for row in prior}

    torch.manual_seed(20260824)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    loader = load_release(args.model)
    started = time.perf_counter()
    model, tokenizer = loader.load_model_and_tokenizer(
        str(args.model), device="cuda", dtype=torch.bfloat16,
        backend=args.backend, state_dtype="float32"
    )
    model.set_kernel_backend(args.backend)
    if args.backend == "tilelang":
        model.prepare_inference_weights()
    load_seconds = time.perf_counter() - started

    rows = list(prior)
    for index, task in enumerate(tasks, start=1):
        if task["id"] in done:
            continue
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": task["prompt"]}], tokenize=True,
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
            max_new_tokens=task.get("max_tokens", 256),
            do_sample=False,
            eos_token_id=0,
            pad_token_id=0,
        )
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        completion_ids = generated[0, prompt_ids.shape[1]:]
        answer = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        passed, detail = grade(task, answer)
        natural_eos = bool(completion_ids.numel() and int(completion_ids[-1]) == 0)
        rows.append({
            "id": task["id"], "category": task["category"], "prompt": task["prompt"],
            "answer": answer, "pass": passed, "grade_detail": detail,
            "new_tokens": int(completion_ids.numel()), "natural_eos": natural_eos,
            "wall_s": round(wall, 3),
        })
        by_category = {}
        for category in sorted({row["category"] for row in rows}):
            group = [row for row in rows if row["category"] == category]
            by_category[category] = {"pass": sum(row["pass"] for row in group), "n": len(group)}
        report = {
            "schema_version": 1,
            "model": str(args.model),
            "weight_sha256": weight_sha,
            "tasks": str(args.tasks),
            "tasks_sha256": task_sha,
            "backend": args.backend,
            "load_seconds": load_seconds,
            "summary": {
                "pass": sum(row["pass"] for row in rows), "n": len(rows),
                "natural_eos": sum(row["natural_eos"] for row in rows),
                "by_category": by_category,
                "median_wall_s": statistics.median(row["wall_s"] for row in rows),
            },
            "results": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{index:02d}/{len(tasks)} {task['id']} {'PASS' if passed else 'FAIL'} "
              f"eos={natural_eos} {wall:.2f}s", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
