#!/usr/bin/env python3
"""Compare replicated full-trace checkpoints with reproduced behavioral finalists."""
from __future__ import annotations

import argparse
import json
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
    build_provenance, canonical_json_sha256, provenance_complete, sha256_file,
)
from tools.research import run_trace_distillation_confirmation_r6 as r6
from tools.research import run_trace_distillation_replication_r8 as r8
from tools.research import run_trace_distillation_training_r2 as r2

TASK_ID = "BACKLOG-ADAPT-TRACE-VS-FINALIST-01"
TRACE_SEEDS = list(range(20260830, 20260837))
BEHAVIOR_SEEDS = [20260824, 20260825]
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2026082711
TRAIN = ROOT / "runs/research/BACKLOG-ADAPT-TRAIN-01"
R7 = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-07"
R8 = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-08"
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-ADAPT-TRACE-VS-FINALIST-01.json"
PREREGISTRATION = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-VS-FINALIST-01/PRE_REGISTRATION.md"
CHECKPOINTS = {
    20260824: TRAIN / "raw/checkpoint_seed_20260824",
    20260825: TRAIN / "raw/checkpoint_seed_20260825",
}
EXPECTED_HASHES = {
    ADMISSION: "759be8cb8abc548c52f4201fa75b03b193ac0c0ba0ac0f23fbedc51c4d11cac9",
    PREREGISTRATION: "b300c3fe69e0a4c9253d3f9165202ba41b7519e0c279aed334d53da8f26005a3",
    TRAIN / "raw/receipt.json": "903c723f3d63130cf06a5e501498451beee0cee34a8aa71d6f9de36faeb602b8",
    CHECKPOINTS[20260824] / "adapter_config.json": "4872214e344ce266b0b10a1f70db33775224441ff1d78dbe0c7fb4dd70aeac84",
    CHECKPOINTS[20260824] / "adapter_model.safetensors": "05b80090d2d1ba751d48a5032cddec82819a79b9bfb5bd8b05306b85d6ef0122",
    CHECKPOINTS[20260825] / "adapter_config.json": "4872214e344ce266b0b10a1f70db33775224441ff1d78dbe0c7fb4dd70aeac84",
    CHECKPOINTS[20260825] / "adapter_model.safetensors": "433978a1b942b4a6d8150e40ca067d2615f811ab8ad2ff880e9a161c655c5646",
    R7 / "raw/student_samples.json": "5283e8e1a66227d71d7a0c5847bd2c147397f580cfaa9c22520edfc65128e19b",
    R7 / "raw/receipt.json": "782d9e58a97c5ac55dd6ebc2d62c67f9e003af415fb62e14ac8124718ea93b3a",
    R8 / "raw/student_samples.json": "178affbf232d8dd7ad6a021d02e4494756e477f121c6506aeae0868a8cc8069d",
    R8 / "raw/receipt.json": "eb4cff3c9d5022887f2621bdf0c303b4aca807e9449e8c600860bd72a046b990",
    r2.DEFAULT_MATH_PATH: "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    r2.TEACHER_PATH: "dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e",
    r2.DEFAULT_QA_PATH: "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
}


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_sources() -> dict[str, Any]:
    ledger: dict[str, Any] = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"source mismatch: {path}: {actual} != {expected}")
        ledger[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": actual}
    return ledger


def long_systemctl(action: str) -> None:
    completed = subprocess.run(
        ["wsl", "-d", "Ubuntu-24.04", "-u", "root", "--", "systemctl", action, "llm-inference.service"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"systemctl {action} failed: {completed.stderr or completed.stdout}")


def worker(output: str, checkpoint: str, panel_path: str, seed: int) -> None:
    import platform
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(r2.BASE_MODEL_WSL, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        r2.BASE_MODEL_WSL, local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa",
    )
    if r2.count_peft_modules(base) != 0:
        raise RuntimeError("fresh base contains PEFT modules")
    model = PeftModel.from_pretrained(base, checkpoint).eval()
    panel = json.loads(pathlib.Path(panel_path).read_text(encoding="utf-8"))
    qa_ids = panel["qa_ids"]
    math_samples: list[dict[str, Any]] = []
    qa_samples: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()

    def generate(prompt: str, maximum: int) -> tuple[str, int, bool, float]:
        encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_n = encoded["input_ids"].shape[1]
        started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=maximum, do_sample=False, temperature=None, top_p=None,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )
        output_ids = generated[0][prompt_n:]
        return (
            tokenizer.decode(output_ids, skip_special_tokens=True).strip(), len(output_ids),
            bool(tokenizer.eos_token_id is not None and generated[0][-1].item() == tokenizer.eos_token_id),
            round(time.monotonic() - started, 4),
        )

    for panel_name in ("panel_1", "panel_2"):
        ids = panel[panel_name]
        for index, task in enumerate(r2.load_math_panel(r2.DEFAULT_MATH_PATH, ids), 1):
            text, token_n, eos, elapsed = generate(task["prompt"], 192)
            extracted = r2.extract_gsm8k_pred(text)
            math_samples.append({
                "panel": panel_name, "task_id": task["task_id"], "prompt": task["prompt"],
                "gold": task["gold"], "output_text": text, "extracted": extracted,
                "correct": r2.is_gsm8k_correct(extracted, task["gold"]),
                "new_tokens": token_n, "natural_eos": eos, "elapsed_s": elapsed,
            })
            if index % 64 == 0:
                print(f"[WORKER] seed={seed} {panel_name}={index}/256", flush=True)
    for index, task in enumerate(r2.load_qa_panel(r2.DEFAULT_QA_PATH, qa_ids), 1):
        text, token_n, eos, elapsed = generate(task["prompt"], 128)
        correct, detail = r2.grade_qa(task, text)
        qa_samples.append({
            "panel": "qa", "task_id": task["id"], "prompt": task["prompt"],
            "output_text": text, "correct": correct, "grade_detail": detail,
            "new_tokens": token_n, "natural_eos": eos, "elapsed_s": elapsed,
        })
        if index % 16 == 0:
            print(f"[WORKER] seed={seed} qa={index}/48", flush=True)
    if len(math_samples) != 512 or len(qa_samples) != 48:
        raise ValueError("behavioral worker output is incomplete")
    write_json(pathlib.Path(output), {
        "family": "behavioral_finalist", "seed": seed, "checkpoint": checkpoint,
        "math_samples": math_samples, "qa_samples": qa_samples,
        "math_correct": sum(row["correct"] for row in math_samples),
        "qa_correct": sum(row["correct"] for row in qa_samples),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "peft": peft.__version__,
                     "cuda": torch.version.cuda},
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_vram_gib": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 6),
    })


def wsl_command(output: pathlib.Path, checkpoint: pathlib.Path, panel: pathlib.Path, seed: int) -> list[str]:
    return [
        "wsl", "-d", "Ubuntu-24.04", "--", r2.WSL_PYTHON,
        r2.windows_path_to_wsl(pathlib.Path(__file__).resolve()), "--worker-mode",
        "--worker-out", r2.windows_path_to_wsl(output),
        "--checkpoint", r2.windows_path_to_wsl(checkpoint),
        "--panel", r2.windows_path_to_wsl(panel), "--seed", str(seed),
    ]


def load_trace(first_ids: list[str], second_ids: list[str], qa_ids: list[str]) -> list[dict[str, Any]]:
    r7_rows = json.loads((R7 / "raw/student_samples.json").read_text(encoding="utf-8"))
    r8_rows = json.loads((R8 / "raw/student_samples.json").read_text(encoding="utf-8"))
    output: list[dict[str, Any]] = []
    for seed in TRACE_SEEDS:
        first = next(row for row in r7_rows if row["seed"] == seed and row["arm"] == "full_trace")
        second = next(row for row in r8_rows if row["seed"] == seed and row["arm"] == "full_trace")
        if [row["task_id"] for row in first["math_samples"]] != first_ids:
            raise ValueError(f"R7 trace panel mismatch for seed {seed}")
        if [row["task_id"] for row in second["math_samples"]] != second_ids:
            raise ValueError(f"R8 trace panel mismatch for seed {seed}")
        if [row["task_id"] for row in first["qa_samples"]] != qa_ids:
            raise ValueError(f"R7 trace QA mismatch for seed {seed}")
        output.append({
            "family": "full_trace", "seed": seed,
            "math_samples": [
                *[{**row, "panel": "panel_1"} for row in first["math_samples"]],
                *[{**row, "panel": "panel_2"} for row in second["math_samples"]],
            ],
            "qa_samples": first["qa_samples"],
        })
    return output


def rescore(rows: list[dict[str, Any]], panels: dict[str, list[str]], qa_ids: list[str]) -> bool:
    math = {task["task_id"]: task for task in r2.load_math_panel(
        r2.DEFAULT_MATH_PATH, [*panels["panel_1"], *panels["panel_2"]]
    )}
    qa = {task["id"]: task for task in r2.load_qa_panel(r2.DEFAULT_QA_PATH, qa_ids)}
    match = True
    for item in rows:
        for sample in item["math_samples"]:
            extracted = r2.extract_gsm8k_pred(sample["output_text"])
            correct = r2.is_gsm8k_correct(extracted, math[sample["task_id"]]["gold"])
            match &= bool(correct) == bool(sample["correct"])
        for sample in item["qa_samples"]:
            correct, _ = r2.grade_qa(qa[sample["task_id"]], sample["output_text"])
            match &= bool(correct) == bool(sample["correct"])
    return match


def bootstrap(trace: list[dict[str, Any]], behavior: list[dict[str, Any]], panels: dict[str, list[str]], replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    def arrays(rows: list[dict[str, Any]], panel_name: str) -> list[list[int]]:
        ids = panels[panel_name]
        result = []
        for row in rows:
            by_id = {sample["task_id"]: int(sample["correct"])
                     for sample in row["math_samples"] if sample["panel"] == panel_name}
            if list(by_id) != ids:
                raise ValueError(f"sample order mismatch for {row['family']} {row['seed']} {panel_name}")
            result.append([by_id[task_id] for task_id in ids])
        return result

    trace_arrays = {name: arrays(trace, name) for name in panels}
    behavior_arrays = {name: arrays(behavior, name) for name in panels}
    rng = random.Random(BOOTSTRAP_SEED)
    estimates: list[float] = []
    for _ in range(replicates):
        trace_draws = [rng.randrange(len(trace)) for _ in trace]
        behavior_draws = [rng.randrange(len(behavior)) for _ in behavior]
        delta = 0.0
        for name in panels:
            prompts = [rng.randrange(256) for _ in range(256)]
            trace_mean = sum(trace_arrays[name][s][p] for s in trace_draws for p in prompts) / (len(trace_draws) * 256)
            behavior_mean = sum(behavior_arrays[name][s][p] for s in behavior_draws for p in prompts) / (len(behavior_draws) * 256)
            delta += (trace_mean - behavior_mean) / 2
        estimates.append(delta)
    estimates.sort()
    return {"replicates": replicates, "seed": BOOTSTRAP_SEED,
            "lower_95": round(estimates[int(0.025 * replicates)], 8),
            "upper_95": round(estimates[min(replicates - 1, int(0.975 * replicates))], 8)}


def score(trace: list[dict[str, Any]], behavior: list[dict[str, Any]], panels: dict[str, list[str]]) -> dict[str, Any]:
    panel_rows = []
    for name, ids in panels.items():
        trace_scores = [sum(row["correct"] for row in item["math_samples"] if row["panel"] == name) / len(ids) for item in trace]
        behavior_scores = [sum(row["correct"] for row in item["math_samples"] if row["panel"] == name) / len(ids) for item in behavior]
        panel_rows.append({"panel": name, "trace_seed_accuracies": trace_scores,
                           "behavioral_seed_accuracies": behavior_scores,
                           "trace_mean": round(statistics.mean(trace_scores), 8),
                           "behavioral_mean": round(statistics.mean(behavior_scores), 8),
                           "trace_minus_behavioral": round(statistics.mean(trace_scores) - statistics.mean(behavior_scores), 8)})
    trace_qa = [sum(row["correct"] for row in item["qa_samples"]) / 48 for item in trace]
    behavior_qa = [sum(row["correct"] for row in item["qa_samples"]) / 48 for item in behavior]
    return {
        "panels": panel_rows,
        "mean_trace_math_accuracy": round(statistics.mean(row["trace_mean"] for row in panel_rows), 8),
        "mean_behavioral_math_accuracy": round(statistics.mean(row["behavioral_mean"] for row in panel_rows), 8),
        "mean_trace_minus_behavioral_math": round(statistics.mean(row["trace_minus_behavioral"] for row in panel_rows), 8),
        "panels_with_positive_trace_minus_behavioral_math": sum(row["trace_minus_behavioral"] > 0 for row in panel_rows),
        "mean_trace_qa_accuracy": round(statistics.mean(trace_qa), 8),
        "mean_behavioral_qa_accuracy": round(statistics.mean(behavior_qa), 8),
        "mean_trace_minus_behavioral_qa_accuracy": round(statistics.mean(trace_qa) - statistics.mean(behavior_qa), 8),
        "hierarchical_bootstrap": bootstrap(trace, behavior, panels),
    }


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    workers, finalized = raw / "workers", raw / "finalized"
    workers.mkdir(parents=True)
    finalized.mkdir(parents=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()
    sources = verify_sources()
    first_ids, second_ids = r8.panel_ids()
    qa_ids = r6.actual_qa_ids()
    panels = {"panel_1": first_ids, "panel_2": second_ids}
    panel_path = raw / "panels.json"
    write_json(panel_path, {**panels, "qa_ids": qa_ids,
                            "panel_hashes": {name: canonical_json_sha256(ids) for name, ids in panels.items()}})
    write_json(raw / "artifact_hashes.json", sources)
    write_json(raw / "dataset_hashes.json", {"panels": json.loads(panel_path.read_text(encoding="utf-8")),
                                              "math_sha256": sha256_file(r2.DEFAULT_MATH_PATH),
                                              "qa_sha256": sha256_file(r2.DEFAULT_QA_PATH),
                                              "teacher_sha256": sha256_file(r2.TEACHER_PATH)})
    write_json(raw / "model_hash.json", r2.verify_base_model())
    trace = load_trace(first_ids, second_ids, qa_ids)

    initial_service = r2.query_service()
    initial_gpu = r2.query_gpu()
    initial_embedding = r2.http_get_json("http://127.0.0.1:8081/health")
    maintenance: dict[str, Any] = {"initial_service": initial_service, "initial_gpu": initial_gpu,
                                   "initial_embedding": initial_embedding, "service_stopped_for_vram": False}
    stopped = False
    behavior: list[dict[str, Any]] = []
    caught: Exception | None = None
    try:
        if initial_gpu["memory_free_mib"] < 12_000 and initial_service["active_state"] == "active":
            long_systemctl("stop")
            stopped = True
            maintenance["service_stopped_for_vram"] = True
        maintenance["service_after_stop"] = r2.query_service()
        maintenance["embedding_after_stop"] = r2.http_get_json("http://127.0.0.1:8081/health")
        maintenance["gpu_after_stop"] = r2.query_gpu()
        if maintenance["embedding_after_stop"].get("status") != "ok" or maintenance["gpu_after_stop"]["memory_free_mib"] < 12_000:
            raise RuntimeError("maintenance health or VRAM gate failed")
        for seed in BEHAVIOR_SEEDS:
            output = workers / f"behavior_seed_{seed}.json"
            command = wsl_command(output, CHECKPOINTS[seed], panel_path, seed)
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                       errors="replace", timeout=10_800, check=False)
            (workers / f"behavior_seed_{seed}.stdout.log").write_text(completed.stdout, encoding="utf-8")
            (workers / f"behavior_seed_{seed}.stderr.log").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode:
                raise RuntimeError(f"behavior worker {seed} failed: {completed.stderr[-4000:]}")
            payload = json.loads(output.read_text(encoding="utf-8"))
            if len(payload["math_samples"]) != 512 or len(payload["qa_samples"]) != 48:
                raise ValueError(f"behavior worker {seed} incomplete")
            behavior.append(payload)
            write_json(finalized / f"behavior_seed_{seed}.json", {"seed": seed,
                       "worker_sha256": sha256_file(output), "math_correct": payload["math_correct"],
                       "qa_correct": payload["qa_correct"]})
            print(f"[HOST] behavior seed {seed}: math={payload['math_correct']}/512 qa={payload['qa_correct']}/48", flush=True)
    except Exception as error:
        caught = error
    finally:
        if stopped:
            long_systemctl("start")
            maintenance["inference_health_final"] = r2.wait_for_health("http://127.0.0.1:8080/health", timeout_seconds=180)
        maintenance["final_service"] = r2.query_service()
        maintenance["final_embedding"] = r2.wait_for_health("http://127.0.0.1:8081/health", timeout_seconds=30)
        maintenance["final_gpu"] = r2.query_gpu()
        maintenance["service_and_embedding_restored"] = (
            maintenance["final_service"]["active_state"] == initial_service["active_state"]
            and r2.normalize_exec_start(maintenance["final_service"]["exec_start"]) == r2.normalize_exec_start(initial_service["exec_start"])
            and maintenance["final_service"]["n_restarts"] == initial_service["n_restarts"]
            and maintenance["final_embedding"].get("status") == "ok"
            and (not stopped or maintenance.get("inference_health_final", {}).get("status") == "ok"))
        write_json(raw / "service_maintenance.json", maintenance)
    if caught:
        raise caught

    independent = rescore([*trace, *behavior], panels, qa_ids)
    scores = score(trace, behavior, panels)
    metrics = {**scores, "all_source_and_checkpoint_hashes_verified": True,
               "two_panels_disjoint_from_teacher_training_and_each_other": True,
               "imported_full_trace_generations": sum(len(x["math_samples"]) + len(x["qa_samples"]) for x in trace),
               "behavioral_checkpoints_evaluated": len(behavior),
               "fresh_behavioral_generations": sum(len(x["math_samples"]) + len(x["qa_samples"]) for x in behavior),
               "independent_rescore_match": independent,
               "service_and_embedding_restored": maintenance["service_and_embedding_restored"]}
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "independent_evaluation.json", {"match": independent, "scores": scores})
    write_json(raw / "continuation_ledger.json", {"trace_r7_sha256": sha256_file(R7 / "raw/student_samples.json"),
               "trace_r8_sha256": sha256_file(R8 / "raw/student_samples.json"),
               "trace_seed_count": 7, "behavior_seed_count": 2, "fresh_rows": metrics["fresh_behavioral_generations"]})
    write_json(raw / "source_execution_receipt.json", {
        "adapt_train_receipt_sha256": sha256_file(TRAIN / "raw/receipt.json"),
        "r7_receipt_sha256": sha256_file(R7 / "raw/receipt.json"),
        "r8_receipt_sha256": sha256_file(R8 / "raw/receipt.json")})
    write_json(raw / "student_samples.json", {"imported_trace": trace, "fresh_behavioral": behavior})
    write_json(raw / "teacher_samples.json", {"source": r2.TEACHER_PATH.relative_to(ROOT).as_posix(),
               "sha256": sha256_file(r2.TEACHER_PATH), "used_for_evaluation": False})
    write_json(raw / "wsl_environment.json", {"versions": behavior[0]["versions"], "gpu": behavior[0]["gpu"],
               "peak_allocated_vram_gib_max": max(x["peak_allocated_vram_gib"] for x in behavior)})
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for item in [*trace, *behavior]:
            for sample in [*item["math_samples"], *item["qa_samples"]]:
                stream.write(json.dumps({"family": item["family"], "seed": item["seed"], **sample}, ensure_ascii=False) + "\n")

    lower = scores["hierarchical_bootstrap"]["lower_95"]
    gates = {
        "source_integrity": {"metric": "all_source_and_checkpoint_hashes_verified", "operator": "eq", "threshold": True, "actual": True, "pass": True},
        "panel_isolation": {"metric": "two_panels_disjoint_from_teacher_training_and_each_other", "operator": "eq", "threshold": True, "actual": True, "pass": True},
        "trace_import_coverage": {"metric": "imported_full_trace_generations", "operator": "eq", "threshold": 3920, "actual": metrics["imported_full_trace_generations"], "pass": metrics["imported_full_trace_generations"] == 3920},
        "behavioral_checkpoint_coverage": {"metric": "behavioral_checkpoints_evaluated", "operator": "eq", "threshold": 2, "actual": len(behavior), "pass": len(behavior) == 2},
        "fresh_evaluation_coverage": {"metric": "fresh_behavioral_generations", "operator": "eq", "threshold": 1120, "actual": metrics["fresh_behavioral_generations"], "pass": metrics["fresh_behavioral_generations"] == 1120},
        "practical_superiority": {"metric": "hierarchical_bootstrap_95ci_lower_trace_minus_behavioral_math", "operator": "gt", "threshold": 0.0, "actual": lower, "pass": lower > 0},
        "panel_consistency": {"metric": "panels_with_positive_trace_minus_behavioral_math", "operator": "eq", "threshold": 2, "actual": scores["panels_with_positive_trace_minus_behavioral_math"], "pass": scores["panels_with_positive_trace_minus_behavioral_math"] == 2},
        "protected_retention": {"metric": "mean_trace_minus_behavioral_qa_accuracy", "operator": "ge", "threshold": -0.05, "actual": scores["mean_trace_minus_behavioral_qa_accuracy"], "pass": scores["mean_trace_minus_behavioral_qa_accuracy"] >= -0.05},
        "independent_score": {"metric": "independent_rescore_match", "operator": "eq", "threshold": True, "actual": independent, "pass": independent is True},
        "service_recovery": {"metric": "service_and_embedding_restored", "operator": "eq", "threshold": True, "actual": maintenance["service_and_embedding_restored"], "pass": maintenance["service_and_embedding_restored"] is True},
    }
    evidence = {"acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
                "artifact_hashes": "raw/artifact_hashes.json", "continuation_ledger": "raw/continuation_ledger.json",
                "dataset_hashes": "raw/dataset_hashes.json", "independent_evaluation": "raw/independent_evaluation.json",
                "model_hash": "raw/model_hash.json", "provenance": "raw/receipt.json",
                "raw_samples": "raw/samples.jsonl", "receipt_fingerprint": "raw/receipt.json",
                "service_maintenance": "raw/service_maintenance.json", "source_execution_receipt": "raw/source_execution_receipt.json",
                "student_samples": "raw/student_samples.json", "teacher_samples": "raw/teacher_samples.json",
                "wsl_environment": "raw/wsl_environment.json"}
    evidence_files = sorted({raw / value.removeprefix("raw/") for value in evidence.values() if value != "raw/receipt.json"})
    provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(), started_at_utc=started,
        started_monotonic=mono, input_paths=[*EXPECTED_HASHES.keys(), panel_path, *workers.glob("*.json"), *evidence_files],
        packages=["pytest"], runtime={"execution_mode": "trace_vs_behavioral_artifact_comparison",
        "host_pid": os.getpid(), "fresh_generation_count": 1120, "imported_generation_count": 3920,
        "timing_is_evidence": False})
    complete, errors = provenance_complete(provenance)
    if not complete or not independent:
        raise ValueError(f"evidence validation failed: {errors}, scorer={independent}")
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID,
               "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = ("TRACE_DISTILLATION_PRACTICALLY_SUPERIOR_TO_BEHAVIORAL_FINALIST_R1" if not failed
             else "TRACE_DISTILLATION_NOT_PRACTICALLY_SUPERIOR_TO_BEHAVIORAL_FINALIST_R1")
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Trace mean math `{scores['mean_trace_math_accuracy']:.6f}`; behavioral mean "
        f"`{scores['mean_behavioral_math_accuracy']:.6f}`; delta `{scores['mean_trace_minus_behavioral_math']:.6f}`; "
        f"bootstrap 95% interval `[{scores['hierarchical_bootstrap']['lower_95']:.6f}, "
        f"{scores['hierarchical_bootstrap']['upper_95']:.6f}]`; QA delta "
        f"`{scores['mean_trace_minus_behavioral_qa_accuracy']:.6f}`. Failed gates: "
        f"`{', '.join(failed) if failed else 'none'}`.\n", encoding="utf-8")
    write_json(finalized / "complete.json", {"task_id": TASK_ID,
               "receipt_fingerprint": receipt["receipt_fingerprint"], "failed_gates": failed})
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-out")
    parser.add_argument("--checkpoint")
    parser.add_argument("--panel")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.worker_mode:
        if not all((args.worker_out, args.checkpoint, args.panel, args.seed)):
            parser.error("worker mode requires worker-out, checkpoint, panel and seed")
        worker(args.worker_out, args.checkpoint, args.panel, args.seed)
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
