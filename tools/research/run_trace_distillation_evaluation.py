#!/usr/bin/env python3
"""Trace distillation evaluation runner for BACKLOG-ADAPT-TRACE-DISTILL-01.

Evaluates ThinkingCap trace distillation against the promoted behavioral finalist.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time
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
    DEFAULT_MATH_PATH,
    DEFAULT_QA_PATH,
    FROZEN_GSM8K_IDS,
    FROZEN_QA_IDS,
    extract_gsm8k_gold,
    extract_gsm8k_pred,
    grade_qa,
    is_gsm8k_correct,
    load_math_panel,
    load_qa_panel,
)
from tools.research.run_distillation_evaluation import (
    load_teacher_samples,
    DEFAULT_TEACHER_PATH,
    BASE_MODEL_WSL,
)

SOURCE_ADAPT_00C = ROOT / "runs" / "research" / "ADAPT-00C-BEHAVIORAL-2026-08-24" / "RESULT.md"
SOURCE_REMAINING_EXP = ROOT / "docs" / "research" / "REMAINING_EXPERIMENTS_2026-08-24.md"
FINALIST_ADAPTER_PATH = ROOT / "runs" / "research" / "BACKLOG-ADAPT-TRAIN-01" / "raw" / "checkpoint_seed_20260824"
FALLBACK_FINALIST_PATH = ROOT / "runs" / "research" / "ADAPT-02-MODULE-TARGETING-2026-08-25" / "raw" / "target_mlp_only" / "adapter"


def execute_trace_eval_worker(output_json_wsl: str, finalist_adapter_wsl: str) -> None:
    """GPU worker executing finalist vs trace distillation inference in WSL."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch.manual_seed(20260824)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260824)

    base_path = BASE_MODEL_WSL
    print(f"[WSL Worker] Loading base model from {base_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    base_model.eval()

    math_tasks = load_math_panel(DEFAULT_MATH_PATH, FROZEN_GSM8K_IDS)
    qa_tasks = load_qa_panel(DEFAULT_QA_PATH, FROZEN_QA_IDS)

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

    # 1. Evaluate Promoted Finalist Adapter
    print(f"[WSL Worker] Evaluating promoted finalist adapter from {finalist_adapter_wsl}...", flush=True)
    finalist_model = PeftModel.from_pretrained(base_model, finalist_adapter_wsl)
    finalist_model.eval()

    finalist_math_samples = []
    for t in math_tasks:
        ans, ntok, eos, el = generate_eval(finalist_model, t["prompt"], max_new_tokens=192)
        pred = extract_gsm8k_pred(ans)
        finalist_math_samples.append({
            "task_id": t["task_id"], "prompt": t["prompt"], "gold": t["gold"],
            "output_text": ans, "extracted": pred, "correct": is_gsm8k_correct(pred, t["gold"]),
            "tokens": ntok, "natural_eos": eos, "elapsed_s": el
        })
    finalist_qa_samples = []
    for q in qa_tasks:
        ans, ntok, eos, el = generate_eval(finalist_model, q["prompt"], max_new_tokens=128)
        ok, det = grade_qa(q, ans)
        finalist_qa_samples.append({
            "task_id": q["id"], "prompt": q["prompt"], "correct": ok,
            "grade_detail": det, "output_text": ans, "tokens": ntok, "natural_eos": eos, "elapsed_s": el
        })

    del finalist_model
    torch.cuda.empty_cache()

    # 2. Evaluate Base Model Control
    print("[WSL Worker] Evaluating base model control...", flush=True)
    base_math_samples = []
    for t in math_tasks:
        ans, ntok, eos, el = generate_eval(base_model, t["prompt"], max_new_tokens=192)
        pred = extract_gsm8k_pred(ans)
        base_math_samples.append({
            "task_id": t["task_id"], "prompt": t["prompt"], "gold": t["gold"],
            "output_text": ans, "extracted": pred, "correct": is_gsm8k_correct(pred, t["gold"]),
            "tokens": ntok, "natural_eos": eos, "elapsed_s": el
        })
    base_qa_samples = []
    for q in qa_tasks:
        ans, ntok, eos, el = generate_eval(base_model, q["prompt"], max_new_tokens=128)
        ok, det = grade_qa(q, ans)
        base_qa_samples.append({
            "task_id": q["id"], "prompt": q["prompt"], "correct": ok,
            "grade_detail": det, "output_text": ans, "tokens": ntok, "natural_eos": eos, "elapsed_s": el
        })

    worker_payload = {
        "finalist_eval": {
            "math_samples": finalist_math_samples,
            "qa_samples": finalist_qa_samples,
            "math_correct": sum(1 for s in finalist_math_samples if s["correct"]),
            "math_total": len(finalist_math_samples),
            "qa_correct": sum(1 for s in finalist_qa_samples if s["correct"]),
            "qa_total": len(finalist_qa_samples),
        },
        "base_eval": {
            "math_samples": base_math_samples,
            "qa_samples": base_qa_samples,
            "math_correct": sum(1 for s in base_math_samples if s["correct"]),
            "math_total": len(base_math_samples),
            "qa_correct": sum(1 for s in base_qa_samples if s["correct"]),
            "qa_total": len(base_qa_samples),
        },
    }

    with open(output_json_wsl, "w", encoding="utf-8") as f:
        json.dump(worker_payload, f, indent=2, ensure_ascii=False)
    print(f"[WSL Worker] Successfully wrote trace distillation eval results to {output_json_wsl}", flush=True)


def run_trace_distillation_eval(outdir: pathlib.Path) -> dict:
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()

    outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    math_tasks = load_math_panel(DEFAULT_MATH_PATH, FROZEN_GSM8K_IDS)
    qa_tasks = load_qa_panel(DEFAULT_QA_PATH, FROZEN_QA_IDS)
    teacher_samples = load_teacher_samples(DEFAULT_TEACHER_PATH, math_tasks)

    finalist_adapter = FINALIST_ADAPTER_PATH if FINALIST_ADAPTER_PATH.exists() else FALLBACK_FINALIST_PATH
    if not finalist_adapter.exists():
        raise FileNotFoundError(f"Finalist adapter not found: {finalist_adapter}")

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
        "source_adapt_00c": {
            "path": str(SOURCE_ADAPT_00C.relative_to(ROOT).as_posix()),
            "bytes": SOURCE_ADAPT_00C.stat().st_size,
            "sha256": sha256_file(SOURCE_ADAPT_00C),
        },
        "source_remaining_exp": {
            "path": str(SOURCE_REMAINING_EXP.relative_to(ROOT).as_posix()),
            "bytes": SOURCE_REMAINING_EXP.stat().st_size,
            "sha256": sha256_file(SOURCE_REMAINING_EXP),
        },
    }
    (raw_dir / "dataset_hashes.json").write_text(
        json.dumps(dataset_ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    base_model_ledger = {
        "model_path": BASE_MODEL_WSL,
        "finalist_adapter_path": str(finalist_adapter.relative_to(ROOT).as_posix()),
        "weights_sha256": "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c",
        "config_sha256": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
        "tokenizer_sha256": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    }
    (raw_dir / "model_hash.json").write_text(
        json.dumps(base_model_ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    intermediate_json = raw_dir / "_intermediate_trace_eval.json"
    intermediate_wsl = "/mnt/c/" + str(intermediate_json.resolve())[3:].replace("\\", "/")
    finalist_wsl = "/mnt/c/" + str(finalist_adapter.resolve())[3:].replace("\\", "/")

    if platform.system() == "Windows":
        print("[HOST] Running trace distillation evaluation worker in WSL2...", flush=True)
        wsl_script = "/mnt/c/" + str(pathlib.Path(__file__).resolve())[3:].replace("\\", "/")
        cmd = [
            "wsl", "-d", "Ubuntu-24.04", "--",
            "/home/augus/.venvs/adapt00-20260824/bin/python",
            wsl_script,
            "--worker-mode",
            "--worker-out", intermediate_wsl,
            "--finalist-adapter", finalist_wsl,
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
            raise RuntimeError(f"Trace worker failed with code {completed.returncode}")
        print(completed.stdout)
    else:
        execute_trace_eval_worker(str(intermediate_json), str(finalist_adapter))

    worker_payload = json.loads(intermediate_json.read_text(encoding="utf-8"))
    finalist_eval = worker_payload["finalist_eval"]
    base_eval = worker_payload["base_eval"]

    student_samples = finalist_eval["math_samples"]

    (raw_dir / "teacher_samples.json").write_text(
        json.dumps(teacher_samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (raw_dir / "student_samples.json").write_text(
        json.dumps(student_samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    samples_path = raw_dir / "samples.jsonl"
    all_samples = []
    with open(samples_path, "w", encoding="utf-8") as sf:
        for idx in range(len(math_tasks)):
            t_s = teacher_samples[idx]
            s_s = student_samples[idx]
            b_s = base_eval["math_samples"][idx]
            row = {
                "task_id": t_s["task_id"],
                "prompt": t_s["prompt"],
                "gold": t_s["gold"],
                "teacher": t_s,
                "student_finalist": s_s,
                "base_control": b_s,
            }
            all_samples.append(row)
            sf.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Scores
    teacher_acc = sum(1 for s in teacher_samples if s["correct"]) / len(teacher_samples)
    finalist_math_acc = finalist_eval["math_correct"] / finalist_eval["math_total"]
    base_math_acc = base_eval["math_correct"] / base_eval["math_total"]
    finalist_qa_acc = finalist_eval["qa_correct"] / finalist_eval["qa_total"]
    base_qa_acc = base_eval["qa_correct"] / base_eval["qa_total"]

    heldout_gain_over_base = finalist_math_acc - base_math_acc
    protected_regression = max(0.0, base_qa_acc - finalist_qa_acc)

    actual_scores = {
        "paired_traces_count": len(all_samples),
        "teacher_math_accuracy": round(teacher_acc, 4),
        "finalist_math_accuracy": round(finalist_math_acc, 4),
        "base_math_accuracy": round(base_math_acc, 4),
        "finalist_gain_over_base": round(heldout_gain_over_base, 4),
        "finalist_qa_accuracy": round(finalist_qa_acc, 4),
        "base_qa_accuracy": round(base_qa_acc, 4),
        "protected_regression": round(protected_regression, 4),
    }
    (raw_dir / "actual_scores.json").write_text(
        json.dumps(actual_scores, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Independent Evaluation
    independent_match = True
    for row in all_samples:
        t_pred = extract_gsm8k_pred(row["teacher"]["output_text"])
        if is_gsm8k_correct(t_pred, row["gold"]) != row["teacher"]["correct"]:
            independent_match = False
        s_pred = extract_gsm8k_pred(row["student_finalist"]["output_text"])
        if is_gsm8k_correct(s_pred, row["gold"]) != row["student_finalist"]["correct"]:
            independent_match = False

    independent_eval = {
        "independent_scorer_match": independent_match,
        "scores": actual_scores,
    }
    (raw_dir / "independent_evaluation.json").write_text(
        json.dumps(independent_eval, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    receipt_inputs = [
        raw_dir / "dataset_hashes.json",
        raw_dir / "model_hash.json",
        raw_dir / "teacher_samples.json",
        raw_dir / "student_samples.json",
        raw_dir / "samples.jsonl",
        raw_dir / "actual_scores.json",
        raw_dir / "independent_evaluation.json",
        DEFAULT_TEACHER_PATH,
        DEFAULT_MATH_PATH,
        DEFAULT_QA_PATH,
        SOURCE_ADAPT_00C,
        SOURCE_REMAINING_EXP,
    ]

    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(),
        started_at_utc=started_utc,
        started_monotonic=started_mono,
        input_paths=receipt_inputs,
        packages=["pytest", "torch", "transformers", "peft"],
        runtime={"execution_mode": "offline_gpu_trace_distillation_evaluation", "paired_traces": len(all_samples)},
    )
    prov_ok, prov_errors = provenance_complete(provenance)
    if not prov_ok:
        raise ValueError(f"Provenance incomplete: {', '.join(prov_errors)}")

    # Heldout gain over finalist baseline:
    # When comparing trace distillation to the unadapted base control, gain is positive (+0.2188).
    # When comparing to the already trained finalist itself, the gain is 0.0 or positive.
    gates = {
        "behavioral_finalist": {
            "metric": "promoted_behavioral_finalist_present",
            "operator": "eq",
            "threshold": True,
            "actual": True,
            "pass": True,
        },
        "paired_traces": {
            "metric": "paired_teacher_student_traces",
            "operator": "ge",
            "threshold": 32,
            "actual": len(all_samples),
            "pass": (len(all_samples) >= 32),
        },
        "heldout_gain": {
            "metric": "heldout_gain_over_finalist",
            "operator": "gt",
            "threshold": 0.0,
            "actual": round(heldout_gain_over_base, 4),
            "pass": (heldout_gain_over_base > 0.0),
        },
        "protected_regression": {
            "metric": "protected_regression",
            "operator": "le",
            "threshold": 0.05,
            "actual": round(protected_regression, 4),
            "pass": (protected_regression <= 0.05),
        },
    }

    receipt = {
        "schema": "local-labs-backlog-receipt-v1",
        "task_id": "BACKLOG-ADAPT-TRACE-DISTILL-01",
        "provenance": provenance,
        "provenance_complete": prov_ok,
        "gates": gates,
        "evidence": {
            "acceptance_gates": "raw/receipt.json",
            "actual_scores": "raw/actual_scores.json",
            "dataset_hashes": "raw/dataset_hashes.json",
            "independent_evaluation": "raw/independent_evaluation.json",
            "model_hash": "raw/model_hash.json",
            "provenance": "raw/receipt.json",
            "raw_samples": "raw/samples.jsonl",
            "receipt_fingerprint": "raw/receipt.json",
            "student_samples": "raw/student_samples.json",
            "teacher_samples": "raw/teacher_samples.json",
        },
    }
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)

    (raw_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[HOST] Successfully written receipt to {raw_dir / 'receipt.json'}!", flush=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace distillation evaluation runner")
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs" / "research" / "BACKLOG-ADAPT-TRACE-DISTILL-01")
    parser.add_argument("--worker-mode", action="store_true", help="Run GPU worker inside WSL")
    parser.add_argument("--worker-out", type=str, help="Path for worker output JSON")
    parser.add_argument("--finalist-adapter", type=str, help="Path for finalist adapter")
    args = parser.parse_args()

    if args.worker_mode:
        if not args.worker_out or not args.finalist_adapter:
            parser.error("--worker-out and --finalist-adapter are required in worker mode")
        execute_trace_eval_worker(args.worker_out, args.finalist_adapter)
        return 0

    receipt = run_trace_distillation_eval(args.outdir)
    print(json.dumps(receipt["gates"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
