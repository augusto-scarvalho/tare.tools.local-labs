#!/usr/bin/env python3
"""ADAPT-01A: LoKr scaling and budget exploration on Qwen3.5-0.8B Base.

Tests whether increasing training steps / epochs (128 -> 384 -> 640) and tuning
learning rate enables LoKr to cross the 16/32 GSM8K correctness threshold and
pass the natural EOS gate without deteriorating protected QA retention.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pathlib
import re
import statistics
import subprocess
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
        "new_tokens": int(tokens.numel()), "natural_eos": eos,
        "wall_s": elapsed,
    }


def arm_summary(name: str, model, tokenizer, target_rows: list[dict],
                protected_tasks: list[dict], teacher_median: float, torch) -> dict:
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

    median_tokens = statistics.median(row["new_tokens"] for row in target_results)
    return {
        "arm": name,
        "summary": {
            "target_correct": sum(row["correct"] for row in target_results),
            "target_n": len(target_results),
            "target_hash_format": sum(row["hash_format"] for row in target_results),
            "protected_pass": sum(row["pass"] for row in protected_results),
            "protected_n": len(protected_results),
            "natural_eos": sum(row["natural_eos"] for row in target_results + protected_results),
            "generation_n": len(target_results) + len(protected_results),
            "median_target_tokens": median_tokens,
            "teacher_median_target_tokens": teacher_median,
            "target_teacher_length_ratio": median_tokens / teacher_median,
            "peak_allocated_vram_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
            "elapsed_seconds": sum(row["wall_s"] for row in target_results + protected_results),
        },
        "target_results": target_results,
        "protected_results": protected_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ADAPT-01A LoKr Scaling & Behavioral Benchmark")
    parser.add_argument("--model-path", default="/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe")
    parser.add_argument("--revision", default="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
    parser.add_argument("--teacher", default="runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json")
    parser.add_argument("--prompts", default="workloads/gsm8k.jsonl")
    parser.add_argument("--protected-tasks", default="runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl")
    parser.add_argument("--output-root", default="runs/research/ADAPT-01A-LOKR-SCALE-2026-08-25/raw")
    parser.add_argument("--python", default="/home/augus/.venvs/adapt00-20260824/bin/python")
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    repo = ROOT
    arm_script = repo / "tools/probes/adapt00_lora_smoke.py"
    output_root = (repo / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Training Phase: LoKr across budgets
    train_configs = [
        ("lokr_1ep", 128, 2e-4),
        ("lokr_3ep", 384, 2e-4),
        ("lokr_5ep", 640, 2e-4),
        ("lokr_3ep_lr1e4", 384, 1e-4),
    ]

    trained_arms: list[tuple[str, pathlib.Path | None]] = [("base", None)]
    for arm_name, steps, lr in train_configs:
        arm_dir = output_root / arm_name
        adapter_dir = arm_dir / "adapter"
        metrics_file = arm_dir / "metrics.json"
        if adapter_dir.exists() and metrics_file.exists():
            print(f"\n[REUSING ALREADY TRAINED ARM]: {arm_name}", flush=True)
            trained_arms.append((arm_name, adapter_dir))
            continue

        print(f"\n==================================================", flush=True)
        print(f"  TRAINING ARM: {arm_name} (steps={steps}, lr={lr})", flush=True)
        print(f"==================================================", flush=True)
        cmd = [
            args.python, str(arm_script),
            "--method", "lokr",
            "--model", "Qwen/Qwen3.5-0.8B-Base",
            "--model-path", args.model_path,
            "--revision", args.revision,
            "--teacher", str(repo / args.teacher),
            "--prompts", str(repo / args.prompts),
            "--protected-file", str(repo / "README.md"),
            "--output", str(arm_dir),
            "--steps", str(steps),
            "--learning-rate", str(lr),
            "--seed", str(args.seed),
        ]
        res = subprocess.run(cmd, text=True, capture_output=True, check=False)
        print(res.stdout[-2000:] if res.stdout else "")
        if res.returncode != 0:
            print(f"ERROR in {arm_name}: {res.stderr[-2000:]}", file=sys.stderr, flush=True)
        else:
            trained_arms.append((arm_name, adapter_dir))

    # 2. Behavioral Evaluation Phase
    import torch
    import transformers
    import peft
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n==================================================", flush=True)
    print("  BEHAVIORAL EVALUATION PANEL (32 GSM8K + 16 QA)", flush=True)
    print("==================================================", flush=True)

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    target_rows = load_target_rows(repo / args.teacher, repo / args.prompts, args.seed)
    task_by_id = {task["id"]: task for task in load_tasks(repo / args.protected_tasks)}
    protected_tasks = [task_by_id[task_id] for task_id in PROTECTED_IDS]
    teacher_counts = [len(tokenizer(row["completion"], add_special_tokens=False)["input_ids"])
                      for row in target_rows]
    teacher_median = statistics.median(teacher_counts)

    eval_results = []
    for name, adapter in trained_arms:
        print(f"\n--- EVALUATING ARM: {name} ---", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=torch.bfloat16, device_map={"": "cuda"},
            attn_implementation="sdpa")
        if adapter is not None and adapter.exists():
            model = PeftModel.from_pretrained(model, str(adapter))
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        result = arm_summary(name, model, tokenizer, target_rows,
                             protected_tasks, teacher_median, torch)
        eval_results.append(result)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Gate Verification & Decision
    base_summary = eval_results[0]["summary"]
    final_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Antigravity",
        "base_control": base_summary,
        "teacher_median_tokens": teacher_median,
        "arms": eval_results,
    }

    for res in eval_results:
        s = res["summary"]
        gates = {
            "target_correct_ge_16": s["target_correct"] >= 16,
            "target_gain_over_base_ge_3": s["target_correct"] - base_summary["target_correct"] >= 3,
            "protected_no_worse_than_base_minus_1": s["protected_pass"] >= base_summary["protected_pass"] - 1,
            "natural_eos_ge_40": s["natural_eos"] >= 40,
            "length_ratio_le_1_25": s["target_teacher_length_ratio"] <= 1.25,
        }
        res["gates"] = gates
        res["promoted"] = all(gates.values())
        print(f"\nARM [{res['arm']}] SUMMARY:", flush=True)
        print(f"  Target Correct: {s['target_correct']}/{s['target_n']} (Pass: {gates['target_correct_ge_16']})")
        print(f"  Protected Pass: {s['protected_pass']}/{s['protected_n']} (Pass: {gates['protected_no_worse_than_base_minus_1']})")
        print(f"  Natural EOS:    {s['natural_eos']}/{s['generation_n']} (Pass: {gates['natural_eos_ge_40']})")
        print(f"  Teacher Ratio:  {s['target_teacher_length_ratio']:.2f}x (Pass: {gates['length_ratio_le_1_25']})")
        print(f"  PROMOTION VERDICT: {'PROMOTED' if res['promoted'] else 'REJECTED'}")

    results_file = output_root / "results.json"
    results_file.write_text(json.dumps(final_output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[DONE] Results written to {results_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
