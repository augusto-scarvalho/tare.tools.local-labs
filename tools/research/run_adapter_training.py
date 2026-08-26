#!/usr/bin/env python3
"""Training reproduction runner for adapter finalists (BACKLOG-ADAPT-TRAIN-01)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)
from tools.research.run_adapter_requalification import (
    FROZEN_GSM8K_IDS,
    FROZEN_QA_IDS,
    extract_gsm8k_gold,
    extract_gsm8k_pred,
    grade_qa,
    is_gsm8k_correct,
    load_math_panel,
    load_qa_panel,
)

BASE_MODEL_WSL = "/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe"
DEFAULT_TEACHER_PATH = ROOT / "runs" / "a2" / "market-r0__thinkingcap-27b-q4__gsm8k.json"
DEFAULT_MATH_PATH = ROOT / "workloads" / "gsm8k.jsonl"
DEFAULT_QA_PATH = ROOT / "runs" / "requalification" / "QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23" / "tasks.jsonl"
SOURCE_RESULT_PATH = ROOT / "runs" / "research" / "TRAIN-00B-GALORE-3090-2026-08-25" / "RESULT.md"

PROMPT_TEMPLATE = (
    "Solve the problem. Show your reasoning, then on the final line write only:\n"
    "#### <answer>\nwhere <answer> is the final number.\n\n{prompt}"
)

SEEDS = [20260824, 20260825]


def load_training_pairs(teacher_path: pathlib.Path, prompt_path: pathlib.Path, seed: int) -> list[dict[str, str]]:
    prompts: dict[str, str] = {}
    with prompt_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                prompts[row["task_id"]] = row["prompt"]

    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    pairs = []
    for row in teacher:
        task_id = row.get("task_id")
        completion = (row.get("completion") or "").strip()
        if task_id in FROZEN_GSM8K_IDS:
            continue
        if row.get("ok") and task_id in prompts and completion:
            pairs.append({
                "task_id": task_id,
                "prompt": PROMPT_TEMPLATE.format(prompt=prompts[task_id]),
                "completion": completion,
            })
    pairs.sort(key=lambda row: row["task_id"])
    random.Random(seed).shuffle(pairs)
    return pairs[:128]


def execute_training_worker(output_json_wsl: str, seeds: list[int], steps: int = 60) -> None:
    """GPU worker executed inside WSL environment."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel

    base_path = BASE_MODEL_WSL
    print(f"[WSL Worker] Loading base model from {base_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    math_tasks = load_math_panel(DEFAULT_MATH_PATH, FROZEN_GSM8K_IDS)
    qa_tasks = load_qa_panel(DEFAULT_QA_PATH, FROZEN_QA_IDS)

    worker_result = {
        "seeds": seeds,
        "runs": [],
        "base_eval": None,
    }

    # Evaluate base model once
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    base_model.eval()

    def generate_eval(model, prompt_text: str, max_new_tokens: int) -> tuple[str, int, bool, float]:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        t0 = time.monotonic()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.monotonic() - t0
        output_ids = outputs[0][input_len:]
        new_tokens = len(output_ids)
        natural_eos = (outputs[0][-1].item() == tokenizer.eos_token_id)
        decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return decoded, new_tokens, natural_eos, round(elapsed, 4)

    print("[WSL Worker] Evaluating base model control...", flush=True)
    base_math_samples = []
    for t in math_tasks:
        ans, ntok, eos, el = generate_eval(base_model, t["prompt"], max_new_tokens=192)
        pred = extract_gsm8k_pred(ans)
        base_math_samples.append({
            "task_id": t["task_id"], "prompt": t["prompt"], "gold": t["gold"],
            "extracted": pred, "correct": is_gsm8k_correct(pred, t["gold"]),
            "output_text": ans, "new_tokens": ntok, "natural_eos": eos, "elapsed_s": el
        })
    base_qa_samples = []
    for q in qa_tasks:
        ans, ntok, eos, el = generate_eval(base_model, q["prompt"], max_new_tokens=128)
        ok, det = grade_qa(q, ans)
        base_qa_samples.append({
            "task_id": q["id"], "prompt": q["prompt"], "correct": ok,
            "grade_detail": det, "output_text": ans, "new_tokens": ntok, "natural_eos": eos, "elapsed_s": el
        })
    worker_result["base_eval"] = {
        "math_samples": base_math_samples,
        "qa_samples": base_qa_samples,
        "math_correct": sum(1 for s in base_math_samples if s["correct"]),
        "math_total": len(base_math_samples),
        "qa_correct": sum(1 for s in base_qa_samples if s["correct"]),
        "qa_total": len(base_qa_samples),
    }

    del base_model
    torch.cuda.empty_cache()

    for s_idx, seed in enumerate(seeds):
        print(f"\n[WSL Worker] === Starting Training Run (Seed {seed}) ===", flush=True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

        pairs = load_training_pairs(DEFAULT_TEACHER_PATH, DEFAULT_MATH_PATH, seed)
        print(f"[WSL Worker] Loaded {len(pairs)} training pairs.", flush=True)

        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        model.config.use_cache = False

        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, lora_cfg)
        peft_model.train()

        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4, weight_decay=0.01)

        trace = []
        max_len = 384
        t_start = time.monotonic()

        for step in range(1, steps + 1):
            row = pairs[(step - 1) % len(pairs)]
            prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
            comp_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
            eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

            prompt_ids = prompt_ids[-(max_len // 2):]
            comp_room = max(1, max_len - len(prompt_ids) - len(eos))
            comp_ids = comp_ids[-comp_room:]
            input_ids = prompt_ids + comp_ids + eos
            comp_n = len(comp_ids) + len(eos)
            labels = [-100] * (len(input_ids) - comp_n) + input_ids[-comp_n:]

            inp_t = torch.tensor([input_ids], dtype=torch.long, device="cuda:0")
            att_t = torch.ones_like(inp_t)
            lab_t = torch.tensor([labels], dtype=torch.long, device="cuda:0")

            optimizer.zero_grad()
            out = peft_model(input_ids=inp_t, attention_mask=att_t, labels=lab_t)
            loss = out.loss
            loss_val = float(loss.item())

            if torch.isnan(loss) or torch.isinf(loss) or loss_val > 100.0:
                raise RuntimeError(f"Loss diverged at step {step}: {loss_val}")

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
            optimizer.step()

            mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            trace.append({
                "step": step,
                "loss": round(loss_val, 6),
                "grad_norm": round(float(grad_norm), 6),
                "memory_mb": round(mem_mb, 2),
                "elapsed_s": round(time.monotonic() - t_start, 4),
            })
            if step % 20 == 0 or step == steps:
                print(f"[WSL Worker] Seed {seed} | Step {step:02d}/{steps} | Loss: {loss_val:.4f} | Mem: {mem_mb:.1f} MB", flush=True)

        ckpt_dir = pathlib.Path(output_json_wsl).parent / f"checkpoint_seed_{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(str(ckpt_dir))
        print(f"[WSL Worker] Saved checkpoint to {ckpt_dir}", flush=True)

        peft_model.eval()
        trained_math_samples = []
        for t in math_tasks:
            ans, ntok, eos, el = generate_eval(peft_model, t["prompt"], max_new_tokens=192)
            pred = extract_gsm8k_pred(ans)
            trained_math_samples.append({
                "task_id": t["task_id"], "prompt": t["prompt"], "gold": t["gold"],
                "extracted": pred, "correct": is_gsm8k_correct(pred, t["gold"]),
                "output_text": ans, "new_tokens": ntok, "natural_eos": eos, "elapsed_s": el
            })
        trained_qa_samples = []
        for q in qa_tasks:
            ans, ntok, eos, el = generate_eval(peft_model, q["prompt"], max_new_tokens=128)
            ok, det = grade_qa(q, ans)
            trained_qa_samples.append({
                "task_id": q["id"], "prompt": q["prompt"], "correct": ok,
                "grade_detail": det, "output_text": ans, "new_tokens": ntok, "natural_eos": eos, "elapsed_s": el
            })

        math_corr = sum(1 for s in trained_math_samples if s["correct"])
        qa_corr = sum(1 for s in trained_qa_samples if s["correct"])
        print(f"[WSL Worker] Seed {seed} eval: GSM8K={math_corr}/{len(math_tasks)}, QA={qa_corr}/{len(qa_tasks)}", flush=True)

        worker_result["runs"].append({
            "seed": seed,
            "steps": steps,
            "final_loss": trace[-1]["loss"],
            "trace": trace,
            "checkpoint_dir": str(ckpt_dir),
            "math_samples": trained_math_samples,
            "qa_samples": trained_qa_samples,
            "math_correct": math_corr,
            "math_total": len(math_tasks),
            "qa_correct": qa_corr,
            "qa_total": len(qa_tasks),
        })

        del peft_model
        del model
        torch.cuda.empty_cache()

    with open(output_json_wsl, "w", encoding="utf-8") as f:
        json.dump(worker_result, f, indent=2, ensure_ascii=False)
    print(f"[WSL Worker] Finished all training seeds! Output written to {output_json_wsl}", flush=True)


def run_training_experiment(outdir: pathlib.Path) -> dict:
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()

    raw_dir = outdir / "raw"
    preexisting_files = list(raw_dir.glob("*")) if raw_dir.exists() else []
    preexisting_count = len(preexisting_files)

    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_ledger = {
        "teacher_dataset": {
            "path": str(DEFAULT_TEACHER_PATH.relative_to(ROOT).as_posix()),
            "bytes": DEFAULT_TEACHER_PATH.stat().st_size,
            "sha256": sha256_file(DEFAULT_TEACHER_PATH),
        },
        "math_panel": {
            "path": str(DEFAULT_MATH_PATH.relative_to(ROOT).as_posix()),
            "bytes": DEFAULT_MATH_PATH.stat().st_size,
            "sha256": sha256_file(DEFAULT_MATH_PATH),
        },
        "qa_panel": {
            "path": str(DEFAULT_QA_PATH.relative_to(ROOT).as_posix()),
            "bytes": DEFAULT_QA_PATH.stat().st_size,
            "sha256": sha256_file(DEFAULT_QA_PATH),
        },
        "source_result": {
            "path": str(SOURCE_RESULT_PATH.relative_to(ROOT).as_posix()),
            "bytes": SOURCE_RESULT_PATH.stat().st_size,
            "sha256": sha256_file(SOURCE_RESULT_PATH),
        },
    }
    (raw_dir / "dataset_hashes.json").write_text(
        json.dumps(dataset_ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    base_model_ledger = {
        "model_path": BASE_MODEL_WSL,
        "revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        "weights_sha256": "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c",
        "config_sha256": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
        "tokenizer_sha256": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    }
    (raw_dir / "model_hash.json").write_text(
        json.dumps(base_model_ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    seed_ledger = {"seeds": SEEDS, "count": len(SEEDS)}
    (raw_dir / "seed.json").write_text(
        json.dumps(seed_ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    intermediate_json = raw_dir / "_intermediate_training.json"
    intermediate_wsl = "/mnt/c/" + str(intermediate_json.resolve())[3:].replace("\\", "/")

    if platform.system() == "Windows":
        print("[HOST] Launching GPU training worker inside WSL2...", flush=True)
        wsl_script = "/mnt/c/" + str(pathlib.Path(__file__).resolve())[3:].replace("\\", "/")
        cmd = [
            "wsl", "-d", "Ubuntu-24.04", "--",
            "/home/augus/.venvs/adapt00-20260824/bin/python",
            wsl_script,
            "--worker-mode",
            "--worker-out", intermediate_wsl,
        ]
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            raise RuntimeError(f"Training worker failed with exit code {completed.returncode}")
        print(completed.stdout)
    else:
        execute_training_worker(str(intermediate_json), SEEDS)

    worker_result = json.loads(intermediate_json.read_text(encoding="utf-8"))

    checkpoint_ledger = {}
    for run_entry in worker_result["runs"]:
        seed_num = run_entry["seed"]
        ckpt_dir = raw_dir / f"checkpoint_seed_{seed_num}"
        cfg_p = ckpt_dir / "adapter_config.json"
        st_p = ckpt_dir / "adapter_model.safetensors"
        checkpoint_ledger[f"seed_{seed_num}"] = {
            "seed": seed_num,
            "config_sha256": sha256_file(cfg_p) if cfg_p.exists() else None,
            "safetensors_sha256": sha256_file(st_p) if st_p.exists() else None,
            "final_loss": run_entry["final_loss"],
        }
    (raw_dir / "checkpoint_hashes.json").write_text(
        json.dumps(checkpoint_ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    traces = {f"seed_{r['seed']}": r["trace"] for r in worker_result["runs"]}
    (raw_dir / "training_trace.json").write_text(
        json.dumps(traces, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    samples_path = raw_dir / "samples.jsonl"
    all_samples = []
    with open(samples_path, "w", encoding="utf-8") as sf:
        base_math = worker_result["base_eval"]["math_samples"]
        for s in base_math:
            row = copy.deepcopy(s)
            row["arm"] = "base"
            row["panel"] = "math"
            all_samples.append(row)
            sf.write(json.dumps(row, ensure_ascii=False) + "\n")
        base_qa = worker_result["base_eval"]["qa_samples"]
        for s in base_qa:
            row = copy.deepcopy(s)
            row["arm"] = "base"
            row["panel"] = "qa"
            all_samples.append(row)
            sf.write(json.dumps(row, ensure_ascii=False) + "\n")

        for r in worker_result["runs"]:
            arm_name = f"lora_mlp_seed_{r['seed']}"
            for s in r["math_samples"]:
                row = copy.deepcopy(s)
                row["arm"] = arm_name
                row["panel"] = "math"
                all_samples.append(row)
                sf.write(json.dumps(row, ensure_ascii=False) + "\n")
            for s in r["qa_samples"]:
                row = copy.deepcopy(s)
                row["arm"] = arm_name
                row["panel"] = "qa"
                all_samples.append(row)
                sf.write(json.dumps(row, ensure_ascii=False) + "\n")

    qa_tasks = load_qa_panel(DEFAULT_QA_PATH, FROZEN_QA_IDS)
    indep_scores = {}
    independent_match = True
    for s in all_samples:
        arm = s["arm"]
        if arm not in indep_scores:
            indep_scores[arm] = {"math_correct": 0, "math_total": 0, "qa_correct": 0, "qa_total": 0}
        if s["panel"] == "math":
            indep_scores[arm]["math_total"] += 1
            pred = extract_gsm8k_pred(s["output_text"])
            ok = is_gsm8k_correct(pred, s["gold"])
            if ok != s["correct"]:
                independent_match = False
            if ok:
                indep_scores[arm]["math_correct"] += 1
        elif s["panel"] == "qa":
            indep_scores[arm]["qa_total"] += 1
            q_task = next(q for q in qa_tasks if q["id"] == s["task_id"])
            ok, _ = grade_qa(q_task, s["output_text"])
            if ok != s["correct"]:
                independent_match = False
            if ok:
                indep_scores[arm]["qa_correct"] += 1

    independent_eval = {
        "independent_scorer_match": independent_match,
        "arm_scores": indep_scores,
        "total_samples": len(all_samples),
    }
    (raw_dir / "independent_evaluation.json").write_text(
        json.dumps(independent_eval, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    successful_seeds_count = len([r for r in worker_result["runs"] if r["final_loss"] < 1.0])
    base_math_acc = indep_scores["base"]["math_correct"] / indep_scores["base"]["math_total"]
    base_qa_acc = indep_scores["base"]["qa_correct"] / indep_scores["base"]["qa_total"]

    seed_math_accs = [
        indep_scores[f"lora_mlp_seed_{r['seed']}"]["math_correct"] / indep_scores[f"lora_mlp_seed_{r['seed']}"]["math_total"]
        for r in worker_result["runs"]
    ]
    seed_qa_accs = [
        indep_scores[f"lora_mlp_seed_{r['seed']}"]["qa_correct"] / indep_scores[f"lora_mlp_seed_{r['seed']}"]["qa_total"]
        for r in worker_result["runs"]
    ]

    min_math_gain = min(acc - base_math_acc for acc in seed_math_accs)
    max_qa_regression = max(max(0.0, base_qa_acc - acc) for acc in seed_qa_accs)

    receipt_inputs = [
        raw_dir / "checkpoint_hashes.json",
        raw_dir / "dataset_hashes.json",
        raw_dir / "model_hash.json",
        raw_dir / "seed.json",
        raw_dir / "training_trace.json",
        raw_dir / "samples.jsonl",
        raw_dir / "independent_evaluation.json",
        DEFAULT_TEACHER_PATH,
        DEFAULT_MATH_PATH,
        DEFAULT_QA_PATH,
        SOURCE_RESULT_PATH,
    ]

    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(),
        started_at_utc=started_utc,
        started_monotonic=started_mono,
        input_paths=receipt_inputs,
        packages=["pytest", "torch", "transformers", "peft"],
        runtime={"execution_mode": "offline_gpu_adapter_training", "seeds": SEEDS},
    )
    prov_ok, prov_errors = provenance_complete(provenance)
    if not prov_ok:
        raise ValueError(f"Provenance incomplete: {', '.join(prov_errors)}")

    gates = {
        "fresh_output": {
            "metric": "preexisting_output_files",
            "operator": "eq",
            "threshold": 0,
            "actual": preexisting_count,
            "pass": (preexisting_count == 0),
        },
        "repeatability": {
            "metric": "successful_repeated_seeds",
            "operator": "ge",
            "threshold": 2,
            "actual": successful_seeds_count,
            "pass": (successful_seeds_count >= 2),
        },
        "behavioral_gain": {
            "metric": "heldout_gain_over_base",
            "operator": "gt",
            "threshold": 0.0,
            "actual": round(min_math_gain, 4),
            "pass": (min_math_gain > 0.0),
        },
        "retention": {
            "metric": "protected_regression",
            "operator": "le",
            "threshold": 0.05,
            "actual": round(max_qa_regression, 4),
            "pass": (max_qa_regression <= 0.05),
        },
    }

    receipt = {
        "schema": "local-labs-backlog-receipt-v1",
        "task_id": "BACKLOG-ADAPT-TRAIN-01",
        "provenance": provenance,
        "provenance_complete": prov_ok,
        "gates": gates,
        "evidence": {
            "acceptance_gates": "raw/receipt.json",
            "checkpoint_hashes": "raw/checkpoint_hashes.json",
            "dataset_hashes": "raw/dataset_hashes.json",
            "independent_evaluation": "raw/independent_evaluation.json",
            "model_hash": "raw/model_hash.json",
            "provenance": "raw/receipt.json",
            "raw_samples": "raw/samples.jsonl",
            "receipt_fingerprint": "raw/receipt.json",
            "seed": "raw/seed.json",
            "training_trace": "raw/training_trace.json",
        },
    }
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)

    (raw_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[HOST] Successfully written receipt to {raw_dir / 'receipt.json'}!", flush=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Adapter training reproduction runner")
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs" / "research" / "BACKLOG-ADAPT-TRAIN-01")
    parser.add_argument("--worker-mode", action="store_true", help="Run GPU worker inside WSL")
    parser.add_argument("--worker-out", type=str, help="Path for worker output JSON")
    args = parser.parse_args()

    if args.worker_mode:
        if not args.worker_out:
            parser.error("--worker-out is required in worker mode")
        execute_training_worker(args.worker_out, SEEDS)
        return 0

    receipt = run_training_experiment(args.outdir)
    print(json.dumps(receipt["gates"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
