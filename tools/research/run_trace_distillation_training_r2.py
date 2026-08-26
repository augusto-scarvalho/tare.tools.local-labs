#!/usr/bin/env python3
"""Matched answer-only versus full-trace SFT for BACKLOG-ADAPT-TRACE-DISTILL-02."""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import subprocess
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file
from tools.research.run_adapter_requalification import (
    DEFAULT_MATH_PATH,
    DEFAULT_QA_PATH,
    FROZEN_GSM8K_IDS,
    FROZEN_QA_IDS,
    extract_gsm8k_pred,
    grade_qa,
    is_gsm8k_correct,
    load_math_panel,
    load_qa_panel,
)
from tools.research.run_adapter_requalification_r2 import (
    http_get_json,
    query_gpu,
    query_service,
    systemctl,
    verify_base_model,
    wait_for_health,
    windows_path_to_wsl,
)

TASK_ID = "BACKLOG-ADAPT-TRACE-DISTILL-02"
BASE_MODEL_WSL = "/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe"
WSL_PYTHON = "/home/augus/.venvs/adapt00-20260824/bin/python"
TEACHER_PATH = ROOT / "runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json"
AUDIT_PATH = ROOT / "docs/AUDIT_2026-08-25_CODEX_INDEPENDENT_AGY_EXECUTION.md"
ADMISSION_PATH = ROOT / "config/research_backlog_admissions/BACKLOG-ADAPT-TRACE-DISTILL-02.json"
PREDECESSOR_TEACHER = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-01/raw/teacher_samples.json"
PREDECESSOR_TRACE = ROOT / "runs/research/BACKLOG-ADAPT-TRAIN-01/raw/training_trace.json"
SEEDS = [20260824, 20260825, 20260826]
ARM_ORDERS = {
    20260824: ["answer_only", "full_trace"],
    20260825: ["full_trace", "answer_only"],
    20260826: ["answer_only", "full_trace"],
}
EXPECTED_HOST_HASHES = {
    ADMISSION_PATH: "4b88fa1a699c886e6ba3f2a3654e4cb6f79aafa12e006ae7455cf20e423e2690",
    AUDIT_PATH: "e4364456156a3c2f015306d986192792fb1aa9ae9333b63a2237ec46e3ffc11f",
    TEACHER_PATH: "dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e",
    DEFAULT_MATH_PATH: "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    DEFAULT_QA_PATH: "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
    PREDECESSOR_TEACHER: "9b9f86bdcfae10ccdd28a0f8a48ccf95da57b8b04cbf06c2f94cb9d6c14e8d08",
    PREDECESSOR_TRACE: "a1c21848acf5d6cf90610806db8f67a3d61acc979be52f36aff998ad37826a31",
}
PROMPT_TEMPLATE = (
    "Solve the problem. Show your reasoning, then on the final line write only:\n"
    "#### <answer>\nwhere <answer> is the final number.\n\n{prompt}"
)


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_host_inputs() -> dict:
    ledger = {}
    for path, expected in EXPECTED_HOST_HASHES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen input mismatch: {path}: {actual} != {expected}")
        ledger[str(path.relative_to(ROOT).as_posix())] = {"bytes": path.stat().st_size, "sha256": actual}
    return ledger


def load_prompt_map() -> dict[str, dict]:
    prompts = {}
    with DEFAULT_MATH_PATH.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                prompts[row["task_id"]] = row
    return prompts


def build_training_manifest() -> dict:
    prompts = load_prompt_map()
    teacher = json.loads(TEACHER_PATH.read_text(encoding="utf-8"))
    eligible = []
    for row in teacher:
        task_id = row.get("task_id")
        completion = (row.get("completion") or "").strip()
        if task_id in FROZEN_GSM8K_IDS:
            continue
        if row.get("ok") and task_id in prompts and completion:
            gold = str(prompts[task_id]["answer"]).split("####")[-1].strip().replace(",", "")
            eligible.append({
                "task_id": task_id,
                "prompt": PROMPT_TEMPLATE.format(prompt=prompts[task_id]["prompt"]),
                "gold": gold,
                "answer_only": f"#### {gold}",
                "full_trace": completion,
            })
    eligible.sort(key=lambda row: row["task_id"])
    if len(eligible) != 168 or len({row["task_id"] for row in eligible}) != 168:
        raise ValueError(f"expected 168 unique eligible examples, got {len(eligible)}")
    if any(row["task_id"] in FROZEN_GSM8K_IDS for row in eligible):
        raise ValueError("held-out math task leaked into eligible training pool")

    selections = {}
    for seed in SEEDS:
        selected = list(eligible)
        random.Random(seed).shuffle(selected)
        selected = selected[:128]
        selections[str(seed)] = selected
    return {
        "schema": "local-labs-trace-training-pairs-v1",
        "eligible_count": len(eligible),
        "heldout_ids": list(FROZEN_GSM8K_IDS),
        "seeds": selections,
    }


def count_peft_modules(model: Any) -> int:
    return sum(1 for module in model.modules() if module.__class__.__module__.startswith("peft."))


def worker(output_path: str, checkpoint_path: str, manifest_path: str, arm: str, seed: int) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_WSL, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_WSL, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    preexisting = count_peft_modules(model)
    if preexisting != 0:
        raise RuntimeError(f"fresh base contained {preexisting} PEFT modules")
    model.config.use_cache = False
    config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM",
    )
    trained = get_peft_model(model, config)
    trained.train()
    optimizer = torch.optim.AdamW(trained.parameters(), lr=1e-4, weight_decay=0.01)
    manifest = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
    rows = manifest["seeds"][str(seed)]
    if len(rows) != 128:
        raise ValueError("worker did not receive 128 training pairs")
    target_key = arm
    trace = []
    started = time.monotonic()
    for step, row in enumerate(rows, 1):
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"][-256:]
        completion_ids = tokenizer(row[target_key], add_special_tokens=False)["input_ids"]
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        completion_ids = completion_ids[-max(1, 512 - len(prompt_ids) - len(eos)):]
        input_ids = prompt_ids + completion_ids + eos
        completion_n = len(completion_ids) + len(eos)
        labels = [-100] * (len(input_ids) - completion_n) + input_ids[-completion_n:]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda:0")
        label_tensor = torch.tensor([labels], dtype=torch.long, device="cuda:0")
        optimizer.zero_grad()
        result = trained(input_ids=input_tensor, attention_mask=torch.ones_like(input_tensor), labels=label_tensor)
        loss = result.loss
        value = float(loss.item())
        if torch.isnan(loss) or torch.isinf(loss) or value > 100:
            raise RuntimeError(f"loss diverged at step {step}: {value}")
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(trained.parameters(), 1.0))
        optimizer.step()
        trace.append({
            "step": step,
            "task_id": row["task_id"],
            "loss": round(value, 6),
            "grad_norm": round(gradient, 6),
            "elapsed_s": round(time.monotonic() - started, 4),
        })
        if step % 32 == 0:
            print(f"[WORKER] {arm} seed={seed} step={step}/128 loss={value:.4f}", flush=True)
    checkpoint = pathlib.Path(checkpoint_path)
    checkpoint.mkdir(parents=True, exist_ok=True)
    trained.save_pretrained(checkpoint)
    trained.eval()

    def generate(prompt: str, maximum: int) -> tuple[str, int, bool, float]:
        tokens = tokenizer(prompt, return_tensors="pt").to(trained.device)
        prompt_n = tokens["input_ids"].shape[1]
        before = time.monotonic()
        with torch.no_grad():
            generated = trained.generate(
                **tokens, max_new_tokens=maximum, do_sample=False, temperature=None, top_p=None,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        output_ids = generated[0][prompt_n:]
        return (
            tokenizer.decode(output_ids, skip_special_tokens=True).strip(),
            len(output_ids),
            bool(generated[0][-1].item() == tokenizer.eos_token_id),
            round(time.monotonic() - before, 4),
        )

    math_samples = []
    for task in load_math_panel(DEFAULT_MATH_PATH, FROZEN_GSM8K_IDS):
        text, token_n, eos, elapsed = generate(task["prompt"], 192)
        extracted = extract_gsm8k_pred(text)
        math_samples.append({
            "panel": "math", "task_id": task["task_id"], "prompt": task["prompt"], "gold": task["gold"],
            "output_text": text, "extracted": extracted, "correct": is_gsm8k_correct(extracted, task["gold"]),
            "new_tokens": token_n, "natural_eos": eos, "elapsed_s": elapsed,
        })
    qa_samples = []
    for task in load_qa_panel(DEFAULT_QA_PATH, FROZEN_QA_IDS):
        text, token_n, eos, elapsed = generate(task["prompt"], 128)
        correct, detail = grade_qa(task, text)
        qa_samples.append({
            "panel": "qa", "task_id": task["id"], "prompt": task["prompt"], "output_text": text,
            "correct": correct, "grade_detail": detail, "new_tokens": token_n,
            "natural_eos": eos, "elapsed_s": elapsed,
        })
    payload = {
        "arm": arm,
        "seed": seed,
        "pid": __import__("os").getpid(),
        "base_preexisting_peft_module_count": preexisting,
        "post_injection_peft_module_count": count_peft_modules(trained),
        "training_pair_count": len(rows),
        "training_task_ids": [row["task_id"] for row in rows],
        "training_target_sha256": canonical_json_sha256([row[target_key] for row in rows]),
        "training_trace": trace,
        "checkpoint": str(checkpoint),
        "math_samples": math_samples,
        "qa_samples": qa_samples,
        "math_correct": sum(sample["correct"] for sample in math_samples),
        "math_total": len(math_samples),
        "qa_correct": sum(sample["correct"] for sample in qa_samples),
        "qa_total": len(qa_samples),
    }
    pathlib.Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def worker_command(script: pathlib.Path, manifest: pathlib.Path, output: pathlib.Path, checkpoint: pathlib.Path, arm: str, seed: int) -> list[str]:
    return [
        "wsl", "-d", "Ubuntu-24.04", "--", WSL_PYTHON, windows_path_to_wsl(script),
        "--worker-mode", "--worker-out", windows_path_to_wsl(output),
        "--checkpoint-out", windows_path_to_wsl(checkpoint), "--manifest", windows_path_to_wsl(manifest),
        "--arm", arm, "--seed", str(seed),
    ]


def normalize_exec_start(value: str) -> str:
    return value.split(" ; ignore_errors=", 1)[0]


def score_outputs(payloads: list[dict]) -> tuple[dict, bool]:
    qa_map = {task["id"]: task for task in load_qa_panel(DEFAULT_QA_PATH, FROZEN_QA_IDS)}
    independent = True
    seed_scores = []
    for seed in SEEDS:
        arms = {payload["arm"]: payload for payload in payloads if payload["seed"] == seed}
        if set(arms) != {"answer_only", "full_trace"}:
            raise ValueError(f"incomplete seed pair: {seed}")
        recomputed = {}
        for arm, payload in arms.items():
            math_correct = 0
            for sample in payload["math_samples"]:
                prediction = extract_gsm8k_pred(sample["output_text"])
                correct = is_gsm8k_correct(prediction, sample["gold"])
                math_correct += int(correct)
                independent = independent and correct == sample["correct"]
            qa_correct = 0
            for sample in payload["qa_samples"]:
                correct, _ = grade_qa(qa_map[sample["task_id"]], sample["output_text"])
                qa_correct += int(correct)
                independent = independent and correct == sample["correct"]
            recomputed[arm] = {"math_correct": math_correct, "qa_correct": qa_correct}
        answer_math = recomputed["answer_only"]["math_correct"] / 32
        trace_math = recomputed["full_trace"]["math_correct"] / 32
        answer_qa = recomputed["answer_only"]["qa_correct"] / 16
        trace_qa = recomputed["full_trace"]["qa_correct"] / 16
        seed_scores.append({
            "seed": seed,
            "answer_math_correct": recomputed["answer_only"]["math_correct"],
            "trace_math_correct": recomputed["full_trace"]["math_correct"],
            "math_gain": round(trace_math - answer_math, 6),
            "answer_qa_correct": recomputed["answer_only"]["qa_correct"],
            "trace_qa_correct": recomputed["full_trace"]["qa_correct"],
            "qa_regression": round(max(0.0, answer_qa - trace_qa), 6),
        })
    scores = {
        "seeds": seed_scores,
        "mean_trace_math_gain_over_answer_only": round(statistics.mean(row["math_gain"] for row in seed_scores), 6),
        "seeds_with_nonnegative_trace_math_gain": sum(row["math_gain"] >= 0 for row in seed_scores),
        "seeds_with_positive_trace_math_gain": sum(row["math_gain"] > 0 for row in seed_scores),
        "mean_protected_qa_regression_vs_answer_only": round(statistics.mean(row["qa_regression"] for row in seed_scores), 6),
    }
    return scores, independent


def run_experiment(outdir: pathlib.Path) -> dict:
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError(f"raw output directory is not empty: {raw}")
    workers = raw / "workers"
    checkpoints = raw / "checkpoints"
    workers.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    host_ledger = verify_host_inputs()
    base_ledger = verify_base_model()
    manifest = build_training_manifest()
    manifest_path = raw / "training_pairs.json"
    write_json(manifest_path, manifest)
    write_json(raw / "dataset_hashes.json", {
        "inputs": host_ledger,
        "eligible_training_examples": manifest["eligible_count"],
        "heldout_math_count": len(FROZEN_GSM8K_IDS),
        "heldout_qa_count": len(FROZEN_QA_IDS),
    })
    write_json(raw / "model_hash.json", base_ledger)

    initial_service = query_service()
    initial_gpu = query_gpu()
    initial_embedding = http_get_json("http://127.0.0.1:8081/health")
    maintenance: dict[str, Any] = {
        "initial_service": initial_service, "initial_gpu": initial_gpu,
        "initial_embedding": initial_embedding, "service_stopped_for_vram": False,
    }
    service_stopped = False
    payloads = []
    script = pathlib.Path(__file__).resolve()
    try:
        if initial_gpu["memory_free_mib"] < 6000 and initial_service["active_state"] == "active":
            systemctl("stop")
            service_stopped = True
            maintenance["service_stopped_for_vram"] = True
            maintenance["service_after_stop"] = query_service()
            maintenance["embedding_after_stop"] = http_get_json("http://127.0.0.1:8081/health")
            if maintenance["embedding_after_stop"].get("status") != "ok":
                raise RuntimeError("embedding service became unhealthy")
        for seed in SEEDS:
            for arm in ARM_ORDERS[seed]:
                label = f"seed_{seed}_{arm}"
                output = workers / f"{label}.json"
                checkpoint = checkpoints / label
                command = worker_command(script, manifest_path, output, checkpoint, arm, seed)
                completed = subprocess.run(command, capture_output=True, text=True, timeout=1800, check=False)
                (workers / f"{label}.stdout.log").write_text(completed.stdout, encoding="utf-8")
                (workers / f"{label}.stderr.log").write_text(completed.stderr, encoding="utf-8")
                if completed.returncode != 0:
                    raise RuntimeError(f"worker {label} failed: {completed.stderr[-3000:]}")
                payload = json.loads(output.read_text(encoding="utf-8"))
                payload["host_command"] = command
                payloads.append(payload)
                print(f"[HOST] {label} complete: math={payload['math_correct']}/32 qa={payload['qa_correct']}/16", flush=True)
    finally:
        if service_stopped:
            systemctl("start")
            maintenance["inference_health_final"] = wait_for_health("http://127.0.0.1:8080/health", timeout_seconds=120)
        maintenance["final_service"] = query_service()
        maintenance["final_embedding"] = wait_for_health("http://127.0.0.1:8081/health", timeout_seconds=30)
        maintenance["final_gpu"] = query_gpu()
        maintenance["service_and_embedding_restored"] = (
            maintenance["final_service"]["active_state"] == initial_service["active_state"]
            and normalize_exec_start(maintenance["final_service"]["exec_start"]) == normalize_exec_start(initial_service["exec_start"])
            and maintenance["final_service"]["n_restarts"] == initial_service["n_restarts"]
            and maintenance["final_embedding"].get("status") == "ok"
            and (not service_stopped or maintenance.get("inference_health_final", {}).get("status") == "ok")
        )
        write_json(raw / "service_maintenance.json", maintenance)

    if len(payloads) != 6:
        raise ValueError(f"expected six workers, got {len(payloads)}")
    clean = [{
        "arm": payload["arm"], "seed": payload["seed"], "pid": payload["pid"],
        "base_preexisting_peft_module_count": payload["base_preexisting_peft_module_count"],
        "post_injection_peft_module_count": payload["post_injection_peft_module_count"],
        "training_pair_count": payload["training_pair_count"],
        "training_target_sha256": payload["training_target_sha256"],
    } for payload in payloads]
    write_json(raw / "clean_base_receipts.json", clean)
    materiality_rows = []
    for seed in SEEDS:
        arms = {payload["arm"]: payload for payload in payloads if payload["seed"] == seed}
        manifest_rows = manifest["seeds"][str(seed)]
        materiality_rows.append({
            "seed": seed,
            "same_task_order": arms["answer_only"]["training_task_ids"] == arms["full_trace"]["training_task_ids"],
            "target_texts_distinct": sum(row["answer_only"] != row["full_trace"] for row in manifest_rows),
            "answer_target_sha256": arms["answer_only"]["training_target_sha256"],
            "trace_target_sha256": arms["full_trace"]["training_target_sha256"],
        })
    treatment_verified = all(
        row["same_task_order"] and row["target_texts_distinct"] == 128
        and row["answer_target_sha256"] != row["trace_target_sha256"]
        for row in materiality_rows
    )
    write_json(raw / "treatment_materiality.json", {"verified": treatment_verified, "seeds": materiality_rows})

    checkpoint_ledger = {}
    for payload in payloads:
        label = f"seed_{payload['seed']}_{payload['arm']}"
        checkpoint = checkpoints / label
        checkpoint_ledger[label] = {
            "config_sha256": sha256_file(checkpoint / "adapter_config.json"),
            "weights_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
        }
    write_json(raw / "checkpoint_hashes.json", checkpoint_ledger)
    write_json(raw / "training_trace.json", [{
        "arm": payload["arm"], "seed": payload["seed"], "trace": payload["training_trace"]
    } for payload in payloads])

    sample_path = raw / "samples.jsonl"
    with sample_path.open("w", encoding="utf-8") as stream:
        for payload in payloads:
            for sample in [*payload["math_samples"], *payload["qa_samples"]]:
                stream.write(json.dumps({"arm": payload["arm"], "seed": payload["seed"], **sample}, ensure_ascii=False) + "\n")
    write_json(raw / "student_samples.json", [{
        "arm": payload["arm"], "seed": payload["seed"],
        "math_samples": payload["math_samples"], "qa_samples": payload["qa_samples"],
    } for payload in payloads])
    write_json(raw / "teacher_samples.json", {
        "source": str(TEACHER_PATH.relative_to(ROOT).as_posix()),
        "selected": {seed: [{"task_id": row["task_id"], "full_trace": row["full_trace"]} for row in rows]
                     for seed, rows in manifest["seeds"].items()},
    })
    scores, independent = score_outputs(payloads)
    write_json(raw / "actual_scores.json", scores)
    write_json(raw / "independent_evaluation.json", {"independent_scorer_match": independent, "scores": scores})

    inputs = [
        raw / "actual_scores.json", raw / "checkpoint_hashes.json", raw / "clean_base_receipts.json",
        raw / "dataset_hashes.json", raw / "independent_evaluation.json", raw / "model_hash.json",
        raw / "samples.jsonl", raw / "service_maintenance.json", raw / "student_samples.json",
        raw / "teacher_samples.json", raw / "training_pairs.json", raw / "training_trace.json",
        raw / "treatment_materiality.json", *EXPECTED_HOST_HASHES.keys(),
    ]
    provenance = build_provenance(
        script_path=script, started_at_utc=started_utc, started_monotonic=started_mono,
        input_paths=inputs, packages=["pytest"],
        runtime={"execution_mode": "matched_trace_distillation_training", "workers": 6, "seeds": SEEDS},
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise ValueError(f"incomplete provenance: {errors}")
    clean_count = sum(row["base_preexisting_peft_module_count"] == 0 for row in clean)
    gates = {
        "treatment_materiality": {"metric": "matched_distinct_training_targets_verified", "operator": "eq", "threshold": True, "actual": treatment_verified, "pass": treatment_verified is True},
        "clean_base": {"metric": "fresh_base_workers", "operator": "eq", "threshold": 6, "actual": clean_count, "pass": clean_count == 6},
        "paired_training": {"metric": "matched_pairs_per_arm_per_seed", "operator": "eq", "threshold": 128, "actual": min(row["training_pair_count"] for row in clean), "pass": all(row["training_pair_count"] == 128 for row in clean)},
        "seed_coverage": {"metric": "completed_paired_seeds", "operator": "eq", "threshold": 3, "actual": len(scores["seeds"]), "pass": len(scores["seeds"]) == 3},
        "heldout_gain": {"metric": "mean_trace_math_gain_over_answer_only", "operator": "gt", "threshold": 0.0, "actual": scores["mean_trace_math_gain_over_answer_only"], "pass": scores["mean_trace_math_gain_over_answer_only"] > 0},
        "directional_repeatability": {"metric": "seeds_with_nonnegative_trace_math_gain", "operator": "ge", "threshold": 2, "actual": scores["seeds_with_nonnegative_trace_math_gain"], "pass": scores["seeds_with_nonnegative_trace_math_gain"] >= 2},
        "protected_regression": {"metric": "mean_protected_qa_regression_vs_answer_only", "operator": "le", "threshold": 0.05, "actual": scores["mean_protected_qa_regression_vs_answer_only"], "pass": scores["mean_protected_qa_regression_vs_answer_only"] <= 0.05},
        "service_recovery": {"metric": "service_and_embedding_restored", "operator": "eq", "threshold": True, "actual": maintenance["service_and_embedding_restored"], "pass": maintenance["service_and_embedding_restored"] is True},
    }
    receipt = {
        "schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID,
        "provenance": provenance, "provenance_complete": complete, "gates": gates,
        "evidence": {
            "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
            "checkpoint_hashes": "raw/checkpoint_hashes.json", "clean_base_receipts": "raw/clean_base_receipts.json",
            "dataset_hashes": "raw/dataset_hashes.json", "independent_evaluation": "raw/independent_evaluation.json",
            "model_hash": "raw/model_hash.json", "provenance": "raw/receipt.json",
            "raw_samples": "raw/samples.jsonl", "receipt_fingerprint": "raw/receipt.json",
            "service_maintenance": "raw/service_maintenance.json", "student_samples": "raw/student_samples.json",
            "teacher_samples": "raw/teacher_samples.json", "training_pairs": "raw/training_pairs.json",
            "training_trace": "raw/training_trace.json", "treatment_materiality": "raw/treatment_materiality.json",
        },
    }
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-out")
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--manifest")
    parser.add_argument("--arm", choices=["answer_only", "full_trace"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.worker_mode:
        required = [args.worker_out, args.checkpoint_out, args.manifest, args.arm, args.seed]
        if any(value is None for value in required):
            parser.error("worker mode requires output, checkpoint, manifest, arm and seed")
        worker(args.worker_out, args.checkpoint_out, args.manifest, args.arm, args.seed)
        return 0
    receipt = run_experiment(args.outdir.resolve())
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
