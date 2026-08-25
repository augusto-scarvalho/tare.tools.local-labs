#!/usr/bin/env python3
"""ADAPT-02: Hybrid Module Targeting Bakeoff on RTX 3090.

Compares module targeting subsets in LoKr (All-Linear vs Attention-Only vs MLP-Only vs QV-Gate)
to isolate functional role efficiency and determine whether selective targeting preserves
general knowledge with lower parameter footprint.
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


def train_targeting_arm(name: str, target_modules: list[str], steps: int, lr: float, out_dir: pathlib.Path, args, repo, torch) -> tuple[pathlib.Path, int]:
    from peft import LoKrConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = out_dir / "adapter"
    if adapter_dir.exists() and (out_dir / "metrics.json").exists():
        print(f"[REUSING ALREADY TRAINED ARM]: {name}", flush=True)
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        return adapter_dir, metrics.get("trainable_parameters", 0)

    print(f"\n==================================================", flush=True)
    print(f"  TRAINING ARM: {name} (targets={target_modules}, steps={steps})", flush=True)
    print(f"==================================================", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    pairs = load_pairs(repo / args.teacher, repo / args.prompts, args.seed)
    train_rows = pairs[:128]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map={"": "cuda"},
        attn_implementation="sdpa")
    model.config.use_cache = False

    peft_config = LoKrConfig(
        task_type="CAUSAL_LM",
        r=8,
        alpha=16,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters for {name}: {trainable_params:,}", flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    torch.cuda.reset_peak_memory_stats()
    model.train()

    losses = []
    started = time.monotonic()

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

    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    metrics = {
        "arm": name,
        "target_modules": target_modules,
        "trainable_parameters": trainable_params,
        "steps": steps,
        "lr": lr,
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return adapter_dir, trainable_params


def main() -> int:
    parser = argparse.ArgumentParser(description="ADAPT-02 Hybrid Module Targeting Probe")
    parser.add_argument("--model-path", default="/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe")
    parser.add_argument("--teacher", default="runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json")
    parser.add_argument("--prompts", default="workloads/gsm8k.jsonl")
    parser.add_argument("--protected-tasks", default="runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl")
    parser.add_argument("--output-root", default="runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw")
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = ROOT
    output_root = (repo / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Training Phase: LoKr across target module subsets (3 epochs = 384 steps)
    targeting_configs = [
        ("target_all_linear", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        ("target_attn_only", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        ("target_mlp_only", ["gate_proj", "up_proj", "down_proj"]),
        ("target_qv_gate", ["q_proj", "v_proj", "gate_proj"]),
    ]

    trained_arms: list[tuple[str, pathlib.Path | None, int]] = [("base", None, 0)]
    for arm_name, modules in targeting_configs:
        arm_dir = output_root / arm_name
        adapter_path, p_count = train_targeting_arm(arm_name, modules, 384, 2e-4, arm_dir, args, repo, torch)
        trained_arms.append((arm_name, adapter_path, p_count))

    # 2. Behavioral Evaluation Panel
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
    for name, adapter, p_count in trained_arms:
        print(f"\n--- EVALUATING ARM: {name} (Params: {p_count:,}) ---", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=torch.bfloat16, device_map={"": "cuda"},
            attn_implementation="sdpa")
        if adapter is not None and adapter.exists():
            model = PeftModel.from_pretrained(model, str(adapter))
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        result = arm_summary(name, model, tokenizer, target_rows,
                             protected_tasks, teacher_median, torch)
        result["trainable_parameters"] = p_count
        eval_results.append(result)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    base_summary = eval_results[0]["summary"]
    final_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Antigravity",
        "base_control": base_summary,
        "teacher_median_tokens": teacher_median,
        "arms": eval_results,
    }

    all_linear_acc = next(r["summary"]["target_correct"] for r in eval_results if r["arm"] == "target_all_linear")

    for res in eval_results:
        s = res["summary"]
        p = res.get("trainable_parameters", 0)
        gates = {
            "target_correct_ge_12": s["target_correct"] >= 12,
            "protected_pass_ge_3": s["protected_pass"] >= 3,
            "natural_eos_ge_36": s["natural_eos"] >= 36,
        }
        res["gates"] = gates
        res["efficiency_score"] = round((s["target_correct"] / max(1, p / 1000.0)), 3) if p > 0 else 0
        print(f"\nARM [{res['arm']}] (Params: {p:,}) SUMMARY:", flush=True)
        print(f"  Target Correct:   {s['target_correct']}/{s['target_n']}")
        print(f"  Protected Pass:   {s['protected_pass']}/{s['protected_n']}")
        print(f"  Natural EOS:      {s['natural_eos']}/{s['generation_n']}")
        print(f"  Efficiency Score: {res['efficiency_score']} corr/kParam")

    results_file = output_root / "results.json"
    results_file.write_text(json.dumps(final_output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[DONE] Results written to {results_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
