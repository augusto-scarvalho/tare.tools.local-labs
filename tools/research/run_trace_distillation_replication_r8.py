#!/usr/bin/env python3
"""Replicate the R7 trace-distillation gain on a second frozen math panel."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pathlib
import random
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
from tools.research import run_trace_distillation_confirmation_r5 as r5
from tools.research import run_trace_distillation_confirmation_r6 as r6
from tools.research import run_trace_distillation_training_r2 as r2

TASK_ID = "BACKLOG-ADAPT-TRACE-DISTILL-08"
SEEDS = list(range(20260830, 20260837))
ARMS = ("answer_only", "full_trace")
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2026082708
SOURCE = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-07"
CHECKPOINTS = SOURCE / "raw/checkpoints"
CHECKPOINT_LEDGER = SOURCE / "raw/checkpoint_hashes.json"
SOURCE_SCORES = SOURCE / "raw/actual_scores.json"
SOURCE_STUDENTS = SOURCE / "raw/student_samples.json"
SOURCE_TRAINING = SOURCE / "raw/training_pairs.json"
SOURCE_RECEIPT = SOURCE / "raw/receipt.json"
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-ADAPT-TRACE-DISTILL-08.json"
PREREGISTRATION = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-08/PRE_REGISTRATION.md"
FIRST_PANEL_HASH = "78a3b7ef26cdf932b79eb6f64dfb576d66770d30a5ce0fd3251536cb6e76901f"
SECOND_PANEL_HASH = "4c88bd4c27eb8fea9240e11503ef781e744313c5de7b1f0391bb680f7e3379bd"
EXPECTED_HASHES = {
    ADMISSION: "d0e8e967ca3885ce39349201edd1f162795058af2e19e630b2780bc36c607059",
    PREREGISTRATION: "616dd2e9db64dca15e8578843588cc85c33ad197b85c871fe6c0fd7ae2a7cb63",
    CHECKPOINT_LEDGER: "57364aaba37c39771aaf216950bfff6df1282735b641b0c682ded89ffa8aaf4c",
    SOURCE_SCORES: "0171dcfcd70334a780a16337469f200656ec3b1d7c567889393c419cad9bae1e",
    SOURCE_STUDENTS: "5283e8e1a66227d71d7a0c5847bd2c147397f580cfaa9c22520edfc65128e19b",
    SOURCE_TRAINING: "5c3f0d5fd80d97351839bca1e38685e5e21b3357dfa56077f44f02b857bfe4cc",
    SOURCE_RECEIPT: "782d9e58a97c5ac55dd6ebc2d62c67f9e003af415fb62e14ac8124718ea93b3a",
    r2.DEFAULT_MATH_PATH: "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    r2.TEACHER_PATH: "dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e",
    r2.DEFAULT_QA_PATH: "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
}


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def panel_ids() -> tuple[list[str], list[str]]:
    teacher = json.loads(r2.TEACHER_PATH.read_text(encoding="utf-8"))
    teacher_ids = {row["task_id"] for row in teacher}
    available = [
        f"gsm8k/{index}" for index in range(1319)
        if f"gsm8k/{index}" not in teacher_ids
    ]
    first, second = available[:256], available[256:512]
    training = json.loads(SOURCE_TRAINING.read_text(encoding="utf-8"))
    training_ids = {row["task_id"] for row in training["pool"]}
    if (
        len(first) != 256
        or len(second) != 256
        or canonical_json_sha256(first) != FIRST_PANEL_HASH
        or canonical_json_sha256(second) != SECOND_PANEL_HASH
        or set(first).intersection(second)
        or teacher_ids.intersection(second)
        or training_ids.intersection(second)
    ):
        raise ValueError("second panel differs from preregistration or is not isolated")
    return first, second


def verify_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    static: dict[str, Any] = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"source mismatch: {path}: {actual} != {expected}")
        static[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": actual}
    expected_checkpoints = json.loads(CHECKPOINT_LEDGER.read_text(encoding="utf-8"))
    expected_labels = {f"seed_{seed}_{arm}" for seed in SEEDS for arm in ARMS}
    if set(expected_checkpoints) != expected_labels:
        raise ValueError("R7 checkpoint ledger does not contain exactly fourteen frozen checkpoints")
    observed: dict[str, Any] = {}
    for label, expected in expected_checkpoints.items():
        checkpoint = CHECKPOINTS / label
        row = {
            "config_sha256": sha256_file(checkpoint / "adapter_config.json"),
            "weights_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
        }
        if row["config_sha256"] != expected["config_sha256"] or row["weights_sha256"] != expected["weights_sha256"]:
            raise ValueError(f"checkpoint mismatch: {label}")
        observed[label] = row
    return static, observed


def long_systemctl(action: str) -> None:
    completed = subprocess.run(
        ["wsl", "-d", "Ubuntu-24.04", "-u", "root", "--", "systemctl", action, "llm-inference.service"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"systemctl {action} failed ({completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )


def worker(output_path: str, checkpoint_path: str, panel_path: str, arm: str, seed: int) -> None:
    import platform
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        r2.BASE_MODEL_WSL, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        r2.BASE_MODEL_WSL,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
    )
    if r2.count_peft_modules(base) != 0:
        raise RuntimeError("fresh base already contains PEFT modules")
    model = PeftModel.from_pretrained(base, checkpoint_path).eval()
    ids = json.loads(pathlib.Path(panel_path).read_text(encoding="utf-8"))["math_ids"]
    tasks = r2.load_math_panel(r2.DEFAULT_MATH_PATH, ids)
    samples: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    for index, task in enumerate(tasks, 1):
        tokens = tokenizer(task["prompt"], return_tensors="pt").to("cuda")
        prompt_n = tokens["input_ids"].shape[1]
        started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                **tokens,
                max_new_tokens=192,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        output_ids = generated[0][prompt_n:]
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        extracted = r2.extract_gsm8k_pred(text)
        samples.append({
            "panel": "math",
            "task_id": task["task_id"],
            "prompt": task["prompt"],
            "gold": task["gold"],
            "output_text": text,
            "extracted": extracted,
            "correct": r2.is_gsm8k_correct(extracted, task["gold"]),
            "new_tokens": len(output_ids),
            "natural_eos": bool(
                tokenizer.eos_token_id is not None
                and generated[0][-1].item() == tokenizer.eos_token_id
            ),
            "elapsed_s": round(time.monotonic() - started, 4),
        })
        if index % 32 == 0:
            print(f"[WORKER] {arm} seed={seed} math={index}/{len(tasks)}", flush=True)
    if len(samples) != 256:
        raise ValueError(f"worker produced {len(samples)} rows, expected 256")
    payload = {
        "arm": arm,
        "seed": seed,
        "checkpoint": checkpoint_path,
        "math_samples": samples,
        "math_correct": sum(sample["correct"] for sample in samples),
        "math_total": len(samples),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "cuda": torch.version.cuda,
        },
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_vram_gib": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 6),
    }
    write_json(pathlib.Path(output_path), payload)


def wsl_command(output: pathlib.Path, checkpoint: pathlib.Path, panel: pathlib.Path, arm: str, seed: int) -> list[str]:
    return [
        "wsl", "-d", "Ubuntu-24.04", "--", r2.WSL_PYTHON,
        r2.windows_path_to_wsl(pathlib.Path(__file__).resolve()),
        "--worker-mode",
        "--worker-out", r2.windows_path_to_wsl(output),
        "--checkpoint", r2.windows_path_to_wsl(checkpoint),
        "--panel", r2.windows_path_to_wsl(panel),
        "--arm", arm,
        "--seed", str(seed),
    ]


def hierarchical_bootstrap(differences: list[list[int]], replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    if len(differences) != 7 or any(len(row) != 256 for row in differences):
        raise ValueError("bootstrap dimensions do not match preregistration")
    rng = random.Random(BOOTSTRAP_SEED)
    estimates: list[float] = []
    for _ in range(replicates):
        total = 0
        for _seed_draw in range(7):
            row = differences[rng.randrange(7)]
            total += sum(row[rng.randrange(256)] for _ in range(256))
        estimates.append(total / (7 * 256))
    estimates.sort()
    return {
        "replicates": replicates,
        "seed": BOOTSTRAP_SEED,
        "lower_95": round(estimates[int(0.025 * replicates)], 8),
        "upper_95": round(estimates[min(replicates - 1, int(0.975 * replicates))], 8),
    }


def exact_sign_flip_pvalue(deltas: list[float]) -> float:
    observed = statistics.mean(deltas)
    magnitudes = [abs(value) for value in deltas]
    outcomes = [
        statistics.mean(sign * magnitude for sign, magnitude in zip(signs, magnitudes))
        for signs in itertools.product((-1, 1), repeat=len(magnitudes))
    ]
    return sum(value >= observed - 1e-15 for value in outcomes) / len(outcomes)


def binomial_upper_tail(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    return sum(math.comb(trials, k) for k in range(successes, trials + 1)) / (2 ** trials)


def score_workers(payloads: list[dict[str, Any]], ids: list[str]) -> tuple[dict[str, Any], bool]:
    math = {task["task_id"]: task for task in r2.load_math_panel(r2.DEFAULT_MATH_PATH, ids)}
    rows: list[dict[str, Any]] = []
    differences: list[list[int]] = []
    independent = True
    trace_only = answer_only = 0
    for seed in SEEDS:
        arms = {payload["arm"]: payload for payload in payloads if payload["seed"] == seed}
        if set(arms) != set(ARMS):
            raise ValueError(f"incomplete pair for seed {seed}")
        values: dict[str, dict[str, int]] = {}
        for arm in ARMS:
            if [sample["task_id"] for sample in arms[arm]["math_samples"]] != ids:
                raise ValueError(f"panel order mismatch for seed {seed} arm {arm}")
            values[arm] = {}
            for sample in arms[arm]["math_samples"]:
                extracted = r2.extract_gsm8k_pred(sample["output_text"])
                correct = int(r2.is_gsm8k_correct(extracted, math[sample["task_id"]]["gold"]))
                independent &= bool(correct) == bool(sample["correct"])
                values[arm][sample["task_id"]] = correct
        paired: list[int] = []
        for task_id in ids:
            answer = values["answer_only"][task_id]
            trace = values["full_trace"][task_id]
            paired.append(trace - answer)
            trace_only += int(trace == 1 and answer == 0)
            answer_only += int(answer == 1 and trace == 0)
        differences.append(paired)
        answer_correct = sum(values["answer_only"].values())
        trace_correct = sum(values["full_trace"].values())
        rows.append({
            "seed": seed,
            "answer_math_correct": answer_correct,
            "trace_math_correct": trace_correct,
            "math_gain": round((trace_correct - answer_correct) / 256, 8),
        })
    deltas = [row["math_gain"] for row in rows]
    bootstrap = hierarchical_bootstrap(differences)
    discordant = trace_only + answer_only
    return {
        "seeds": rows,
        "mean_trace_math_gain_over_answer_only": round(statistics.mean(deltas), 8),
        "hierarchical_bootstrap": bootstrap,
        "seeds_with_positive_trace_math_gain": sum(value > 0 for value in deltas),
        "exact_one_sided_seed_sign_flip_p": round(exact_sign_flip_pvalue(deltas), 8),
        "pooled_discordant_pairs": {
            "trace_only_correct": trace_only,
            "answer_only_correct": answer_only,
            "one_sided_exact_mcnemar_p": round(binomial_upper_tail(trace_only, discordant), 12),
        },
    }, independent


def imported_qa_regression() -> tuple[float, bool, list[dict[str, Any]]]:
    qa_ids = r6.actual_qa_ids()
    qa = {task["id"]: task for task in r2.load_qa_panel(r2.DEFAULT_QA_PATH, qa_ids)}
    students = json.loads(SOURCE_STUDENTS.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    independent = True
    for seed in SEEDS:
        arms = {row["arm"]: row for row in students if row["seed"] == seed}
        if set(arms) != set(ARMS):
            raise ValueError(f"R7 QA pair missing for seed {seed}")
        scores: dict[str, int] = {}
        for arm in ARMS:
            samples = arms[arm]["qa_samples"]
            if [sample["task_id"] for sample in samples] != qa_ids:
                raise ValueError(f"R7 QA order mismatch for seed {seed} arm {arm}")
            score = 0
            for sample in samples:
                correct, _ = r2.grade_qa(qa[sample["task_id"]], sample["output_text"])
                score += int(correct)
                independent &= bool(correct) == bool(sample["correct"])
            scores[arm] = score
        rows.append({
            "seed": seed,
            "answer_qa_correct": scores["answer_only"],
            "trace_qa_correct": scores["full_trace"],
            "qa_regression": round(max(0.0, (scores["answer_only"] - scores["full_trace"]) / 48), 8),
        })
    return round(statistics.mean(row["qa_regression"] for row in rows), 8), independent, rows


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    workers, finalized = raw / "workers", raw / "finalized"
    workers.mkdir(parents=True)
    finalized.mkdir(parents=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()
    static, checkpoints = verify_sources()
    first_ids, second_ids = panel_ids()
    panel = raw / "panel.json"
    write_json(panel, {"math_ids": second_ids, "math_id_sha256": canonical_json_sha256(second_ids)})
    write_json(raw / "checkpoint_hashes.json", checkpoints)
    write_json(raw / "dataset_hashes.json", {
        "static": static,
        "r7_math_ids_sha256": canonical_json_sha256(first_ids),
        "second_math_ids": second_ids,
        "second_math_ids_sha256": canonical_json_sha256(second_ids),
        "panels_disjoint": set(first_ids).isdisjoint(second_ids),
    })
    write_json(raw / "model_hash.json", r2.verify_base_model())

    initial_service = r2.query_service()
    initial_gpu = r2.query_gpu()
    initial_embedding = r2.http_get_json("http://127.0.0.1:8081/health")
    maintenance: dict[str, Any] = {
        "initial_service": initial_service,
        "initial_gpu": initial_gpu,
        "initial_embedding": initial_embedding,
        "service_stopped_for_vram": False,
    }
    service_stopped = False
    payloads: list[dict[str, Any]] = []
    worker_error: Exception | None = None
    try:
        if initial_gpu["memory_free_mib"] < 12_000 and initial_service["active_state"] == "active":
            long_systemctl("stop")
            service_stopped = True
            maintenance["service_stopped_for_vram"] = True
        maintenance["service_after_stop"] = r2.query_service()
        maintenance["embedding_after_stop"] = r2.http_get_json("http://127.0.0.1:8081/health")
        maintenance["gpu_after_stop"] = r2.query_gpu()
        if maintenance["embedding_after_stop"].get("status") != "ok":
            raise RuntimeError("embedding service became unhealthy")
        if maintenance["gpu_after_stop"]["memory_free_mib"] < 12_000:
            raise RuntimeError("insufficient free VRAM after bounded maintenance")
        for seed_index, seed in enumerate(SEEDS):
            order = ARMS if seed_index % 2 == 0 else tuple(reversed(ARMS))
            for arm in order:
                label = f"seed_{seed}_{arm}"
                output = workers / f"{label}.json"
                command = wsl_command(output, CHECKPOINTS / label, panel, arm, seed)
                completed = subprocess.run(
                    command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=7200, check=False,
                )
                (workers / f"{label}.stdout.log").write_text(completed.stdout, encoding="utf-8")
                (workers / f"{label}.stderr.log").write_text(completed.stderr, encoding="utf-8")
                if completed.returncode:
                    raise RuntimeError(f"worker {label} failed ({completed.returncode}): {completed.stderr[-4000:]}")
                payload = json.loads(output.read_text(encoding="utf-8"))
                if payload["math_total"] != 256:
                    raise ValueError(f"worker {label} is incomplete")
                payloads.append(payload)
                write_json(finalized / f"{label}.json", {
                    "label": label,
                    "worker_sha256": sha256_file(output),
                    "math_correct": payload["math_correct"],
                })
                print(f"[HOST] {label} complete: math={payload['math_correct']}/256", flush=True)
    except Exception as error:
        worker_error = error
    finally:
        if service_stopped:
            long_systemctl("start")
            maintenance["inference_health_final"] = r2.wait_for_health(
                "http://127.0.0.1:8080/health", timeout_seconds=180
            )
        maintenance["final_service"] = r2.query_service()
        maintenance["final_embedding"] = r2.wait_for_health(
            "http://127.0.0.1:8081/health", timeout_seconds=30
        )
        maintenance["final_gpu"] = r2.query_gpu()
        maintenance["service_and_embedding_restored"] = (
            maintenance["final_service"]["active_state"] == initial_service["active_state"]
            and r2.normalize_exec_start(maintenance["final_service"]["exec_start"])
            == r2.normalize_exec_start(initial_service["exec_start"])
            and maintenance["final_service"]["n_restarts"] == initial_service["n_restarts"]
            and maintenance["final_embedding"].get("status") == "ok"
            and (not service_stopped or maintenance.get("inference_health_final", {}).get("status") == "ok")
        )
        write_json(raw / "service_maintenance.json", maintenance)
    if worker_error is not None:
        raise worker_error
    if len(payloads) != 14:
        raise ValueError(f"expected fourteen workers, got {len(payloads)}")

    scores, math_match = score_workers(payloads, second_ids)
    qa_regression, qa_match, qa_rows = imported_qa_regression()
    independent_match = math_match and qa_match
    metrics = {
        **scores,
        "r7_source_and_checkpoint_hashes_verified": True,
        "second_panel_disjoint_from_teacher_training_and_r7_panel": True,
        "immutable_checkpoints_evaluated": len(payloads),
        "fresh_second_panel_generations": sum(payload["math_total"] for payload in payloads),
        "imported_r7_mean_protected_qa_regression": qa_regression,
        "independent_rescore_match": independent_match,
        "service_and_embedding_restored": maintenance["service_and_embedding_restored"],
    }
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "independent_evaluation.json", {
        "independent_rescore_match": independent_match,
        "math_scores": scores,
        "imported_qa": qa_rows,
    })
    write_json(raw / "continuation_ledger.json", {
        "source_task_id": "BACKLOG-ADAPT-TRACE-DISTILL-07",
        "source_receipt_sha256": sha256_file(SOURCE_RECEIPT),
        "checkpoint_ledger_sha256": sha256_file(CHECKPOINT_LEDGER),
        "checkpoint_count": len(checkpoints),
        "imported_qa_source_sha256": sha256_file(SOURCE_STUDENTS),
        "fresh_math_per_checkpoint": 256,
    })
    write_json(raw / "source_execution_receipt.json", {
        "source_task_id": "BACKLOG-ADAPT-TRACE-DISTILL-07",
        "receipt_sha256": sha256_file(SOURCE_RECEIPT),
        "receipt_fingerprint": json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))["receipt_fingerprint"],
    })
    write_json(raw / "student_samples.json", [
        {"arm": payload["arm"], "seed": payload["seed"], "math_samples": payload["math_samples"]}
        for payload in payloads
    ])
    write_json(raw / "teacher_samples.json", {
        "teacher_source": r2.TEACHER_PATH.relative_to(ROOT).as_posix(),
        "teacher_sha256": sha256_file(r2.TEACHER_PATH),
        "teacher_rows_used_for_training": 168,
        "teacher_rows_used_for_second_panel": 0,
    })
    write_json(raw / "wsl_environment.json", {
        "versions": payloads[0]["versions"],
        "gpu": payloads[0]["gpu"],
        "worker_count": len(payloads),
        "peak_allocated_vram_gib_max": max(payload["peak_allocated_vram_gib"] for payload in payloads),
    })
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for payload in payloads:
            for sample in payload["math_samples"]:
                stream.write(json.dumps({
                    "arm": payload["arm"], "seed": payload["seed"], **sample
                }, ensure_ascii=False) + "\n")

    gates = {
        "source_integrity": {"metric": "r7_source_and_checkpoint_hashes_verified", "operator": "eq", "threshold": True, "actual": True, "pass": True},
        "panel_isolation": {"metric": "second_panel_disjoint_from_teacher_training_and_r7_panel", "operator": "eq", "threshold": True, "actual": True, "pass": True},
        "checkpoint_coverage": {"metric": "immutable_checkpoints_evaluated", "operator": "eq", "threshold": 14, "actual": len(payloads), "pass": len(payloads) == 14},
        "evaluation_coverage": {"metric": "fresh_second_panel_generations", "operator": "eq", "threshold": 3584, "actual": metrics["fresh_second_panel_generations"], "pass": metrics["fresh_second_panel_generations"] == 3584},
        "replicated_gain": {"metric": "hierarchical_bootstrap_95ci_lower_trace_math_gain", "operator": "gt", "threshold": 0.0, "actual": scores["hierarchical_bootstrap"]["lower_95"], "pass": scores["hierarchical_bootstrap"]["lower_95"] > 0.0},
        "directional_repeatability": {"metric": "seeds_with_positive_trace_math_gain", "operator": "ge", "threshold": 5, "actual": scores["seeds_with_positive_trace_math_gain"], "pass": scores["seeds_with_positive_trace_math_gain"] >= 5},
        "protected_retention": {"metric": "imported_r7_mean_protected_qa_regression", "operator": "le", "threshold": 0.05, "actual": qa_regression, "pass": qa_regression <= 0.05},
        "independent_score": {"metric": "independent_rescore_match", "operator": "eq", "threshold": True, "actual": independent_match, "pass": independent_match is True},
        "service_recovery": {"metric": "service_and_embedding_restored", "operator": "eq", "threshold": True, "actual": maintenance["service_and_embedding_restored"], "pass": maintenance["service_and_embedding_restored"] is True},
    }
    evidence = {
        "acceptance_gates": "raw/receipt.json",
        "actual_scores": "raw/actual_scores.json",
        "checkpoint_hashes": "raw/checkpoint_hashes.json",
        "continuation_ledger": "raw/continuation_ledger.json",
        "dataset_hashes": "raw/dataset_hashes.json",
        "independent_evaluation": "raw/independent_evaluation.json",
        "model_hash": "raw/model_hash.json",
        "provenance": "raw/receipt.json",
        "raw_samples": "raw/samples.jsonl",
        "receipt_fingerprint": "raw/receipt.json",
        "service_maintenance": "raw/service_maintenance.json",
        "source_execution_receipt": "raw/source_execution_receipt.json",
        "student_samples": "raw/student_samples.json",
        "teacher_samples": "raw/teacher_samples.json",
        "wsl_environment": "raw/wsl_environment.json",
    }
    evidence_files = sorted({
        raw / value.removeprefix("raw/")
        for value in evidence.values()
        if value != "raw/receipt.json"
    })
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(),
        started_at_utc=started,
        started_monotonic=mono,
        input_paths=[*EXPECTED_HASHES.keys(), panel, *workers.glob("*.json"), *evidence_files],
        packages=["pytest"],
        runtime={
            "execution_mode": "immutable_checkpoint_second_panel_replication",
            "host_pid": os.getpid(),
            "checkpoint_count": 14,
            "fresh_generation_count": 3584,
            "timing_is_evidence": False,
        },
    )
    complete, errors = provenance_complete(provenance)
    if not complete or not independent_match:
        raise ValueError(f"evidence validation failed: provenance={errors}, scorer={independent_match}")
    receipt = {
        "schema": "local-labs-backlog-receipt-v1",
        "task_id": TASK_ID,
        "provenance": provenance,
        "provenance_complete": True,
        "gates": gates,
        "evidence": evidence,
    }
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = (
        "TRACE_DISTILLATION_SECOND_PANEL_REPLICATED_R8"
        if not failed else "TRACE_DISTILLATION_SECOND_PANEL_NOT_REPLICATED_R8"
    )
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"All 14 immutable R7 checkpoints completed 256 fresh math tasks each. "
        f"Mean trace gain `{scores['mean_trace_math_gain_over_answer_only']:.6f}`; "
        f"hierarchical bootstrap 95% interval "
        f"`[{scores['hierarchical_bootstrap']['lower_95']:.6f}, "
        f"{scores['hierarchical_bootstrap']['upper_95']:.6f}]`; positive seeds "
        f"`{scores['seeds_with_positive_trace_math_gain']}/7`; imported R7 QA regression "
        f"`{qa_regression:.6f}`. Failed gates: `{', '.join(failed) if failed else 'none'}`.\n",
        encoding="utf-8",
    )
    write_json(finalized / "complete.json", {
        "task_id": TASK_ID,
        "receipt_fingerprint": receipt["receipt_fingerprint"],
        "failed_gates": failed,
    })
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-out")
    parser.add_argument("--checkpoint")
    parser.add_argument("--panel")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.worker_mode:
        if not all((args.worker_out, args.checkpoint, args.panel, args.arm, args.seed)):
            parser.error("worker mode requires worker-out, checkpoint, panel, arm and seed")
        worker(args.worker_out, args.checkpoint, args.panel, args.arm, args.seed)
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
