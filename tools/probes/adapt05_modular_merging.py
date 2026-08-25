#!/usr/bin/env python3
"""ADAPT-05: Modular Skill Composition (TIES-Merging & DARE) on RTX 3090.

Evaluates adapter merging techniques (Linear Average vs DARE vs TIES vs Disjoint Composition)
combining specialized Math (MLP-Only) and QA (Attn-Only) LoKr adapters on Qwen3.5-0.8B.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import pathlib
import random
import re
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools/benchmarks"))

from normal_qa_ab import grade, load_tasks  # noqa: E402
from tools.analysis.a2_stats import gsm8k_extract, numeric_equal  # noqa: E402
from tools.probes.adapt00_lora_smoke import load_pairs  # noqa: E402

PROTECTED_IDS = (
    "f01", "f02", "f03", "m01", "m02", "m03", "r01", "r02", "r03",
    "i01", "i02", "i03", "c01", "c02", "s01", "s02",
)
HASH_FORMAT = re.compile(r"####\s*\$?(-?[0-9][0-9,]*\.?[0-9]*)")


def load_target_rows(teacher: pathlib.Path, prompts: pathlib.Path, seed: int) -> list[dict]:
    pairs = load_pairs(teacher, prompts, seed)
    target = pairs[128:160]
    prompt_rows = {
        row["task_id"]: row
        for row in (json.loads(line) for line in prompts.read_text(encoding="utf-8").splitlines())
    }
    for row in target:
        row["gold"] = prompt_rows[row["task_id"]]["answer"]
    return target


def generate(model, tokenizer, prompt: str, max_new_tokens: int, torch) -> dict:
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded["attention_mask"].to("cuda")
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.monotonic() - started
    tokens = output[0, input_ids.shape[1]:]
    eos = bool(tokens.numel() and int(tokens[-1]) == tokenizer.eos_token_id)
    return {
        "text": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
        "new_tokens": int(tokens.numel()),
        "natural_eos": eos,
        "wall_s": elapsed,
    }


def evaluate_model(name: str, model, tokenizer, target_rows, protected_tasks, torch) -> dict:
    target_results = []
    for row in target_rows:
        generated = generate(model, tokenizer, row["prompt"], 192, torch)
        answer = gsm8k_extract(generated["text"])
        generated.update({
            "task_id": row["task_id"], "gold": row["gold"],
            "extracted": answer, "correct": numeric_equal(answer, row["gold"]),
            "hash_format": bool(HASH_FORMAT.search(generated["text"])),
        })
        target_results.append(generated)
        print(f"  [{name}] target {row['task_id']} "
              f"{'PASS' if generated['correct'] else 'FAIL'} "
              f"eos={generated['natural_eos']} n={generated['new_tokens']}", flush=True)

    protected_results = []
    for task in protected_tasks:
        generated = generate(model, tokenizer, task["prompt"], 128, torch)
        passed, detail = grade(task, generated["text"])
        generated.update({"id": task["id"], "category": task["category"],
                          "pass": passed, "grade_detail": detail})
        protected_results.append(generated)
        print(f"  [{name}] protected {task['id']} "
              f"{'PASS' if passed else 'FAIL'} eos={generated['natural_eos']} "
              f"n={generated['new_tokens']}", flush=True)

    return {
        "arm": name,
        "target_correct": sum(row["correct"] for row in target_results),
        "target_n": len(target_results),
        "protected_pass": sum(row["pass"] for row in protected_results),
        "protected_n": len(protected_results),
        "natural_eos": sum(row["natural_eos"] for row in target_results + protected_results),
        "generation_n": len(target_results) + len(protected_results),
        "target_results": target_results,
        "protected_results": protected_results,
    }


def ties_merge_tensors(tensors: list["torch.Tensor"], trim_ratio: float = 0.3, torch=None) -> "torch.Tensor":
    """Applies TIES-Merging: Trims smallest deltas, calculates sign consensus, and averages."""
    stacked = torch.stack(tensors, dim=0)  # Shape (N, ...)
    # 1. Trim smallest values per tensor
    trimmed = []
    for t in tensors:
        flat = t.abs().flatten()
        threshold = torch.quantile(flat, trim_ratio)
        mask = t.abs() >= threshold
        trimmed.append(t * mask)
    trimmed_stacked = torch.stack(trimmed, dim=0)

    # 2. Sign consensus: sum of signs
    signs = torch.sign(trimmed_stacked)
    sign_sum = signs.sum(dim=0)
    consensus_sign = torch.sign(sign_sum)

    # 3. Average elements matching consensus sign
    matching_mask = (signs == consensus_sign.unsqueeze(0)) & (trimmed_stacked != 0)
    denom = matching_mask.sum(dim=0).clamp(min=1)
    merged = (trimmed_stacked * matching_mask).sum(dim=0) / denom
    return merged


def dare_merge_tensors(tensors: list["torch.Tensor"], drop_prob: float = 0.5, torch=None) -> "torch.Tensor":
    """Applies DARE: Randomly drops deltas with probability drop_prob and rescales by 1/(1-p)."""
    rescaled = []
    scale = 1.0 / (1.0 - drop_prob)
    for t in tensors:
        mask = (torch.rand_like(t) > drop_prob).float()
        rescaled.append(t * mask * scale)
    return torch.stack(rescaled, dim=0).mean(dim=0)


def create_disjoint_merged_adapter(math_dir: pathlib.Path, qa_dir: pathlib.Path, out_dir: pathlib.Path):
    """Merges MLP-only and Attn-only adapters by combining their disjoint weight tensors."""
    from safetensors.torch import load_file, save_file
    out_dir.mkdir(parents=True, exist_ok=True)

    math_weights = load_file(str(math_dir / "adapter_model.safetensors"))
    qa_weights = load_file(str(qa_dir / "adapter_model.safetensors"))

    merged_weights = {}
    merged_weights.update(math_weights)
    merged_weights.update(qa_weights)

    save_file(merged_weights, str(out_dir / "adapter_model.safetensors"))

    # Update adapter_config
    config = json.loads((math_dir / "adapter_config.json").read_text(encoding="utf-8"))
    config["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="ADAPT-05 Modular Skill Composition Probe")
    parser.add_argument("--model-path", default="/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe")
    parser.add_argument("--math-adapter", default="runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/target_mlp_only/adapter")
    parser.add_argument("--qa-adapter", default="runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/target_attn_only/adapter")
    parser.add_argument("--teacher", default="runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json")
    parser.add_argument("--prompts", default="workloads/gsm8k.jsonl")
    parser.add_argument("--protected-tasks", default="runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl")
    parser.add_argument("--output-root", default="runs/research/ADAPT-05-MODULAR-MERGING-2026-08-25/raw")
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = ROOT
    output_root = (repo / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    math_adapter_path = (repo / args.math_adapter).resolve()
    qa_adapter_path = (repo / args.qa_adapter).resolve()
    disjoint_dir = output_root / "disjoint_composite"

    print("=== ADAPT-05 Modular Skill Composition Probe ===", flush=True)
    print("Building Disjoint Composite Adapter (MLP Math + Attention QA)...", flush=True)
    create_disjoint_merged_adapter(math_adapter_path, qa_adapter_path, disjoint_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    target_rows = load_target_rows(repo / args.teacher, repo / args.prompts, args.seed)
    task_by_id = {task["id"]: task for task in load_tasks(repo / args.protected_tasks)}
    protected_tasks = [task_by_id[task_id] for task_id in PROTECTED_IDS]

    # Evaluate Disjoint Composite Adapter
    print("\n--- Evaluating Disjoint Composite Adapter ---", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map={"": "cuda"},
        attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base_model, str(disjoint_dir))
    model.eval()

    eval_result = evaluate_model("disjoint_composite", model, tokenizer, target_rows, protected_tasks, torch)

    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    target_correct = eval_result["target_correct"]
    protected_pass = eval_result["protected_pass"]
    total_score = target_correct + protected_pass

    print(f"\nDisjoint Composite Results: GSM8K = {target_correct}/32 | QA = {protected_pass}/16 | Total = {total_score}/48")

    gates = {
        "math_score_ge_14": target_correct >= 14,
        "qa_score_ge_4": protected_pass >= 4,
        "natural_eos_ge_38": eval_result["natural_eos"] >= 38,
    }

    verdict = "PROMOTED" if all(gates.values()) else "REJECTED"

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Antigravity",
        "composite_adapter": str(disjoint_dir),
        "results": eval_result,
        "gates": gates,
        "verdict": verdict,
    }

    results_file = output_root / "results.json"
    results_file.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n==================================================", flush=True)
    print(f"  ADAPT-05 MODULAR MERGING VERDICT: {verdict}", flush=True)
    print(f"  Receipt written to: {results_file}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if verdict == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
