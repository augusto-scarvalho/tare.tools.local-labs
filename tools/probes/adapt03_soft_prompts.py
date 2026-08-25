#!/usr/bin/env python3
"""ADAPT-03: Learned Semantic Tokens (Soft Prompts) on RTX 3090.

Evaluates Prompt Tuning with 8 virtual tokens on Qwen3.5-0.8B-Base to measure
format steering, reasoning accuracy, and general knowledge preservation
with ultra-compact parameter footprint (8.192 parameters / 16 KB).
"""
from __future__ import annotations

import argparse
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
from tools.probes.adapt00_lora_smoke import load_pairs, target_batch  # noqa: E402

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


def main() -> int:
    parser = argparse.ArgumentParser(description="ADAPT-03 Learned Semantic Tokens Probe")
    parser.add_argument("--model-path", default="/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe")
    parser.add_argument("--teacher", default="runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json")
    parser.add_argument("--prompts", default="workloads/gsm8k.jsonl")
    parser.add_argument("--protected-tasks", default="runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl")
    parser.add_argument("--output-root", default="runs/research/ADAPT-03-SOFT-PROMPTS-2026-08-25/raw")
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    import torch
    from peft import PeftModel, PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = ROOT
    output_root = (repo / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_root / "adapter"

    print("=== ADAPT-03 Learned Semantic Tokens (Soft Prompts) Probe ===", flush=True)

    # 1. Training Phase
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    pairs = load_pairs(repo / args.teacher, repo / args.prompts, args.seed)
    train_rows = pairs[:128]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {args.model_path}...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map={"": "cuda"},
        attn_implementation="sdpa")
    base_model.config.use_cache = False

    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=8,
        prompt_tuning_init_text="Solve systematically step by step and conclude with ####",
        tokenizer_name_or_path=args.model_path,
    )
    model = get_peft_model(base_model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters (Soft Prompt): {trainable_params:,} ({trainable_params * 2 / 1024:.2f} KB)", flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    torch.cuda.reset_peak_memory_stats()
    model.train()

    steps = 384
    losses = []
    started = time.monotonic()

    print(f"Training Soft Prompts for {steps} steps...", flush=True)
    for step in range(steps):
        target_row = train_rows[step % len(train_rows)]
        batch_target = target_batch(target_row, tokenizer, 384, torch)

        optimizer.zero_grad(set_to_none=True)
        out = model(
            input_ids=batch_target.input_ids,
            attention_mask=batch_target.attention_mask,
            labels=batch_target.labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.float().item())

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    train_elapsed = time.monotonic() - started
    print(f"Training completed in {train_elapsed:.2f}s | First Loss: {losses[0]:.4f} | Final Loss: {losses[-1]:.4f}", flush=True)

    # 2. Behavioral Evaluation
    print("\n--- Running Behavioral Evaluation (32 GSM8K + 16 Protected QA) ---", flush=True)
    model.eval()
    target_rows = load_target_rows(repo / args.teacher, repo / args.prompts, args.seed)
    task_by_id = {task["id"]: task for task in load_tasks(repo / args.protected_tasks)}
    protected_tasks = [task_by_id[task_id] for task_id in PROTECTED_IDS]

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
        print(f"  [soft_prompt] target {row['task_id']} "
              f"{'PASS' if generated['correct'] else 'FAIL'} "
              f"fmt={generated['hash_format']} eos={generated['natural_eos']} n={generated['new_tokens']}", flush=True)

    protected_results = []
    for task in protected_tasks:
        generated = generate(model, tokenizer, task["prompt"], 128, torch)
        passed, detail = grade(task, generated["text"])
        generated.update({"id": task["id"], "category": task["category"],
                          "pass": passed, "grade_detail": detail})
        protected_results.append(generated)
        print(f"  [soft_prompt] protected {task['id']} "
              f"{'PASS' if passed else 'FAIL'} eos={generated['natural_eos']} "
              f"n={generated['new_tokens']}", flush=True)

    summary = {
        "target_correct": sum(row["correct"] for row in target_results),
        "target_n": len(target_results),
        "target_hash_format": sum(row["hash_format"] for row in target_results),
        "protected_pass": sum(row["pass"] for row in protected_results),
        "protected_n": len(protected_results),
        "natural_eos": sum(row["natural_eos"] for row in target_results + protected_results),
        "generation_n": len(target_results) + len(protected_results),
        "trainable_parameters": trainable_params,
        "storage_footprint_kb": round(trainable_params * 2 / 1024.0, 2),
        "train_loss_start": round(losses[0], 4),
        "train_loss_end": round(losses[-1], 4),
    }

    gates = {
        "hash_format_rate_ge_75pct": (summary["target_hash_format"] / summary["target_n"]) >= 0.75,
        "protected_pass_ge_3": summary["protected_pass"] >= 3,
        "storage_footprint_le_32kb": summary["storage_footprint_kb"] <= 32.0,
    }

    verdict = "PROMOTED" if all(gates.values()) else "REJECTED"

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Antigravity",
        "model": args.model_path,
        "summary": summary,
        "gates": gates,
        "verdict": verdict,
        "target_results": target_results,
        "protected_results": protected_results,
    }

    results_file = output_root / "results.json"
    results_file.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n==================================================", flush=True)
    print(f"  ADAPT-03 SOFT PROMPTS VERDICT: {verdict}", flush=True)
    print(f"  Target Correct:       {summary['target_correct']}/{summary['target_n']}")
    print(f"  Format Compliance:    {summary['target_hash_format']}/{summary['target_n']} ({summary['target_hash_format']/summary['target_n']*100:.1f}%)")
    print(f"  Protected Pass:       {summary['protected_pass']}/{summary['protected_n']}")
    print(f"  Storage Footprint:    {summary['storage_footprint_kb']} KB")
    print(f"  Receipt written to: {results_file}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if verdict == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
