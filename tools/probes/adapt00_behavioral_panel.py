#!/usr/bin/env python3
"""ADAPT-00C: behavioral gate for preregistered PEFT role representatives."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pathlib
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


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        print(f"  {name} target {row['task_id']} "
              f"{'PASS' if generated['correct'] else 'FAIL'} "
              f"eos={generated['natural_eos']} n={generated['new_tokens']}", flush=True)

    protected_results = []
    for task in protected_tasks:
        generated = generate(model, tokenizer, task["prompt"], 128, torch)
        passed, detail = grade(task, generated["text"])
        generated.update({"id": task["id"], "category": task["category"],
                          "pass": passed, "grade_detail": detail})
        protected_results.append(generated)
        print(f"  {name} protected {task['id']} "
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--teacher", type=pathlib.Path, required=True)
    parser.add_argument("--prompts", type=pathlib.Path, required=True)
    parser.add_argument("--protected-tasks", type=pathlib.Path, required=True)
    parser.add_argument("--lora-adapter", type=pathlib.Path, required=True)
    parser.add_argument("--lokr-adapter", type=pathlib.Path, required=True)
    parser.add_argument("--ia3-adapter", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    import torch
    import transformers
    import peft
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(20260824)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    target_rows = load_target_rows(args.teacher, args.prompts, 20260824)
    task_by_id = {task["id"]: task for task in load_tasks(args.protected_tasks)}
    protected_tasks = [task_by_id[task_id] for task_id in PROTECTED_IDS]
    teacher_counts = [len(tokenizer(row["completion"], add_special_tokens=False)["input_ids"])
                      for row in target_rows]
    teacher_median = statistics.median(teacher_counts)
    arms = [
        ("base", None), ("lora", args.lora_adapter),
        ("lokr", args.lokr_adapter), ("ia3", args.ia3_adapter),
    ]
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, adapter in arms:
        print(f"=== ADAPT-00C {name} ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=torch.bfloat16, device_map={"": "cuda"},
            attn_implementation="sdpa")
        if adapter is not None:
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        result = arm_summary(name, model, tokenizer, target_rows,
                             protected_tasks, teacher_median, torch)
        results.append(result)
        partial = {"status": "RUNNING", "arms": results}
        args.output.write_text(json.dumps(partial, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    base = results[0]["summary"]
    for result in results:
        summary = result["summary"]
        gates = {
            "target_correct_ge_16": summary["target_correct"] >= 16,
            "target_gain_over_base_ge_3":
                summary["target_correct"] - base["target_correct"] >= 3,
            "protected_no_worse_than_base_minus_1":
                summary["protected_pass"] >= base["protected_pass"] - 1,
            "natural_eos_ge_46_of_48": summary["natural_eos"] >= 46,
            "median_length_le_1_25x_teacher":
                summary["target_teacher_length_ratio"] <= 1.25,
        }
        result["gates"] = gates
        result["promotion_verdict"] = (
            "PASS" if result["arm"] != "base" and all(gates.values()) else
            "CONTROL" if result["arm"] == "base" else "FAIL")
    report = {
        "schema_version": 1, "status": "COMPLETE",
        "model": "Qwen/Qwen3.5-0.8B-Base", "revision": args.revision,
        "versions": {"torch": torch.__version__,
                     "transformers": transformers.__version__, "peft": peft.__version__},
        "inputs": {
            "teacher_sha256": sha256(args.teacher), "prompts_sha256": sha256(args.prompts),
            "protected_tasks_sha256": sha256(args.protected_tasks),
            "protected_ids": list(PROTECTED_IDS),
        },
        "promoted_arms": [result["arm"] for result in results
                          if result["promotion_verdict"] == "PASS"],
        "arms": results,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "promoted_arms": report["promoted_arms"],
        "summaries": {result["arm"]: result["summary"] for result in results},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
