#!/usr/bin/env python3
"""Run the final third panel and frozen three-panel LoKr synthesis."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import random
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
from tools.research import run_adapt01_broad_artifact_eval_r2 as r2
from tools.research import run_adapt01_broad_artifact_eval_r3 as r3
from tools.research import run_adapt01_broad_artifact_eval_r4 as r4
from tools.research import run_trace_distillation_training_r2 as training

TASK_ID = "BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-05"
R3_WORKER = ROOT / "runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-03/raw/merged_worker.json"
R3_RECEIPT = ROOT / "runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-03/raw/receipt.json"
R4_WORKER = ROOT / "runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-04/raw/combined_worker.json"
R4_RECEIPT = ROOT / "runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-04/raw/receipt.json"
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-05.json"
PREREGISTRATION = ROOT / "runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-05/PRE_REGISTRATION.md"
THIRD_PANEL_HASH = "73024245450c6158c150d654243e30ef26e562027dcc6514abd096862d0a69fe"
EXPECTED_HASHES = {
    ADMISSION: "f0a4cd0de4a1192637eb288f84ed896f0cc13f21d403f41d64bd1ea1cbf6aeee",
    R3_WORKER: "7beecfc4a2970ce39f6e5a8343d4d6b23fd5e78fa8328b03195f3e61acdb6b2f",
    R3_RECEIPT: "3ca20a2dbad797cc6ce1629d501802c6fa06ee8d67c8ab948f2959c62b298f40",
    R4_WORKER: "9556cffa5c23554a0e47184b8ef4af0a59114c575ef053954ccdbf674529e5b7",
    R4_RECEIPT: "185935ee19679509b780150d683707ef294818dd6711b7076b09ac5cb835f8e8",
    ROOT / "runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-04/PRE_REGISTRATION.md": "73cfe7178bee1770a55b41982836ac836d5e0f4b011e95e151cdce0d09db7d5c",
    ROOT / "tools/research/run_adapt01_broad_artifact_eval_r4.py": "ac1f7453cb873fba1ba04d544f87776e975da6a41c27956a3e310d76c6e1c539",
    r2.ADAPTER / "adapter_model.safetensors": "7f6d082243f6b406259791dc15a65e4b092b48597fad9b68018d507872ad8fa7",
    r2.ADAPTER / "adapter_config.json": "08cf4d254e2a6c9aba9d34ba6a0c76926b478d7cd0ad771062acefb71a31d934",
    training.DEFAULT_MATH_PATH: "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    training.TEACHER_PATH: "dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e",
    training.DEFAULT_QA_PATH: "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
}


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def third_panel_ids() -> list[str]:
    teacher = json.loads(training.TEACHER_PATH.read_text(encoding="utf-8"))
    teacher_ids = {row["task_id"] for row in teacher}
    available = [
        f"gsm8k/{index}"
        for index in range(1319)
        if f"gsm8k/{index}" not in teacher_ids
    ]
    first, second, third = available[:256], available[256:512], available[512:768]
    if (
        len(third) != 256
        or set(first).intersection(second)
        or set(first).intersection(third)
        or set(second).intersection(third)
        or canonical_json_sha256(first) != r2.HELDOUT_HASH
        or canonical_json_sha256(second) != r4.SECOND_PANEL_HASH
        or canonical_json_sha256(third) != THIRD_PANEL_HASH
    ):
        raise ValueError("third held-out panel differs from preregistration")
    return third


def verify_sources() -> dict[str, dict[str, Any]]:
    ledger = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"source mismatch: {path}: {actual} != {expected}")
        ledger[path.relative_to(ROOT).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    return ledger


def stratified_bootstrap(
    panel_differences: list[list[int]], replicates: int = 20_000
) -> dict[str, Any]:
    if len(panel_differences) != 3 or any(len(panel) != 256 for panel in panel_differences):
        raise ValueError("stratified bootstrap requires three 256-task panels")
    rng = random.Random(2026082705)
    estimates = []
    for _ in range(replicates):
        panel_means = [
            sum(panel[rng.randrange(256)] for _ in range(256)) / 256
            for panel in panel_differences
        ]
        estimates.append(sum(panel_means) / 3)
    estimates.sort()
    return {
        "replicates": replicates,
        "seed": 2026082705,
        "lower_95": round(estimates[int(0.025 * replicates)], 8),
        "upper_95": round(estimates[min(replicates - 1, int(0.975 * replicates))], 8),
    }


def long_systemctl(action: str) -> None:
    completed = subprocess.run(
        [
            "wsl", "-d", "Ubuntu-24.04", "-u", "root", "--",
            "systemctl", action, "llm-inference.service",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"systemctl {action} failed ({completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )


def wsl_command(*arguments: str) -> list[str]:
    return [
        "wsl", "-d", "Ubuntu-24.04", "--", training.WSL_PYTHON,
        training.windows_path_to_wsl(pathlib.Path(__file__).resolve()),
        *arguments,
    ]


def score_math_panel(payload: dict[str, Any], ids: list[str]) -> tuple[list[int], dict[str, int], bool]:
    arms = {arm["arm"]: arm for arm in payload["arms"]}
    math_map = {
        task["task_id"]: task
        for task in training.load_math_panel(training.DEFAULT_MATH_PATH, ids)
    }
    differences = []
    correct_counts = {"base": 0, "lokr_3ep_lr1e4": 0}
    match = True
    for task_id in ids:
        values = {}
        for arm_name in ("base", "lokr_3ep_lr1e4"):
            sample = next(row for row in arms[arm_name]["math_samples"] if row["task_id"] == task_id)
            extracted = training.extract_gsm8k_pred(sample["output_text"])
            correct = int(training.is_gsm8k_correct(extracted, math_map[task_id]["gold"]))
            match &= bool(correct) == bool(sample["correct"])
            correct_counts[arm_name] += correct
            values[arm_name] = correct
        differences.append(values["lokr_3ep_lr1e4"] - values["base"])
    return differences, correct_counts, match


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    finalized = raw / "finalized"
    finalized.mkdir(parents=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()
    sources = verify_sources()
    panel_ids = [r2.heldout_ids(), r4.second_panel_ids(), third_panel_ids()]
    if any(set(panel_ids[index]).intersection(panel_ids[other]) for index in range(3) for other in range(index)):
        raise ValueError("frozen panels overlap")
    qa_ids = r3.actual_qa_ids()
    r3_payload = json.loads(R3_WORKER.read_text(encoding="utf-8"))
    r4_payload = json.loads(R4_WORKER.read_text(encoding="utf-8"))
    panel = raw / "panel.json"
    write_json(panel, {
        "math_ids": panel_ids[2],
        "qa_ids": [],
        "math_id_sha256": canonical_json_sha256(panel_ids[2]),
    })
    write_json(raw / "artifact_hashes.json", sources)
    write_json(raw / "dataset_hashes.json", {
        "math": sources[training.DEFAULT_MATH_PATH.relative_to(ROOT).as_posix()],
        "teacher": sources[training.TEACHER_PATH.relative_to(ROOT).as_posix()],
        "qa": sources[training.DEFAULT_QA_PATH.relative_to(ROOT).as_posix()],
        "panel_hashes": [canonical_json_sha256(ids) for ids in panel_ids],
        "qa_id_sha256": canonical_json_sha256(qa_ids),
    })
    write_json(raw / "model_hash.json", training.verify_base_model())

    initial_service = training.query_service()
    initial_gpu = training.query_gpu()
    initial_embedding = training.http_get_json("http://127.0.0.1:8081/health")
    maintenance: dict[str, Any] = {
        "initial_service": initial_service,
        "initial_gpu": initial_gpu,
        "initial_embedding": initial_embedding,
        "service_stopped_for_vram": False,
    }
    service_stopped = False
    worker_path = raw / "fresh_third_panel_worker.json"
    worker_error: Exception | None = None
    try:
        if initial_gpu["memory_free_mib"] < 12_000 and initial_service["active_state"] == "active":
            long_systemctl("stop")
            service_stopped = True
            maintenance["service_stopped_for_vram"] = True
        maintenance["service_after_stop"] = training.query_service()
        maintenance["embedding_after_stop"] = training.http_get_json("http://127.0.0.1:8081/health")
        maintenance["gpu_after_stop"] = training.query_gpu()
        if maintenance["embedding_after_stop"].get("status") != "ok":
            raise RuntimeError("embedding service became unhealthy")
        if maintenance["gpu_after_stop"]["memory_free_mib"] < 12_000:
            raise RuntimeError("insufficient free VRAM after bounded maintenance")
        command = wsl_command(
            "--worker-mode", "--worker-out", training.windows_path_to_wsl(worker_path),
            "--panel", training.windows_path_to_wsl(panel),
            "--adapter", training.windows_path_to_wsl(r2.ADAPTER),
        )
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=7200,
            check=False,
        )
        (raw / "worker.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (raw / "worker.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"third-panel worker failed ({completed.returncode}): {completed.stderr[-4000:]}")
    except Exception as error:
        worker_error = error
    finally:
        if service_stopped:
            long_systemctl("start")
            maintenance["inference_health_final"] = training.wait_for_health(
                "http://127.0.0.1:8080/health", timeout_seconds=180
            )
        maintenance["final_service"] = training.query_service()
        maintenance["final_embedding"] = training.wait_for_health(
            "http://127.0.0.1:8081/health", timeout_seconds=30
        )
        maintenance["final_gpu"] = training.query_gpu()
        maintenance["service_and_embedding_restored"] = (
            maintenance["final_service"]["active_state"] == initial_service["active_state"]
            and training.normalize_exec_start(maintenance["final_service"]["exec_start"])
            == training.normalize_exec_start(initial_service["exec_start"])
            and maintenance["final_service"]["n_restarts"] == initial_service["n_restarts"]
            and maintenance["final_embedding"].get("status") == "ok"
            and (
                not service_stopped
                or maintenance.get("inference_health_final", {}).get("status") == "ok"
            )
        )
        write_json(raw / "service_maintenance.json", maintenance)
    if worker_error is not None:
        raise worker_error
    if not worker_path.is_file():
        raise RuntimeError("third-panel worker returned without output")

    r5_payload = json.loads(worker_path.read_text(encoding="utf-8"))
    r5_arms = {arm["arm"]: arm for arm in r5_payload["arms"]}
    r3_arms = {arm["arm"]: arm for arm in r3_payload["arms"]}
    for arm_name in ("base", "lokr_3ep_lr1e4"):
        if len(r5_arms[arm_name]["math_samples"]) != 256 or r5_arms[arm_name]["qa_samples"]:
            raise ValueError(f"fresh third-panel dimensions invalid for {arm_name}")
        if [sample["task_id"] for sample in r3_arms[arm_name]["qa_samples"]] != qa_ids:
            raise ValueError(f"R3 QA source order invalid for {arm_name}")

    payloads = [r3_payload, r4_payload, r5_payload]
    scored = [score_math_panel(payload, ids) for payload, ids in zip(payloads, panel_ids)]
    panel_differences = [item[0] for item in scored]
    panel_counts = [item[1] for item in scored]
    independent_match = all(item[2] for item in scored)
    qa_map = {
        task["id"]: task
        for task in training.load_qa_panel(training.DEFAULT_QA_PATH, qa_ids)
    }
    for arm_name in ("base", "lokr_3ep_lr1e4"):
        for sample in r3_arms[arm_name]["qa_samples"]:
            correct, _ = training.grade_qa(qa_map[sample["task_id"]], sample["output_text"])
            independent_match &= bool(correct) == bool(sample["correct"])
    bootstrap = stratified_bootstrap(panel_differences)
    panel_gains = [
        (counts["lokr_3ep_lr1e4"] - counts["base"]) / 256
        for counts in panel_counts
    ]
    qa_correct = {
        arm_name: sum(sample["correct"] for sample in r3_arms[arm_name]["qa_samples"])
        for arm_name in ("base", "lokr_3ep_lr1e4")
    }
    qa_regression = max(0.0, qa_correct["base"] / 48 - qa_correct["lokr_3ep_lr1e4"] / 48)
    metrics = {
        "r3_r4_source_hashes_verified": True,
        "artifact_hashes_verified": True,
        "three_panels_pairwise_and_teacher_disjoint": True,
        "fresh_third_panel_paired_math_generations": sum(len(arm["math_samples"]) for arm in r5_arms.values()),
        "panel_correct_counts": panel_counts,
        "panel_math_gains": [round(value, 8) for value in panel_gains],
        "three_panel_math_gain": round(sum(panel_gains) / 3, 8),
        "stratified_bootstrap": bootstrap,
        "third_panel_math_gain": round(panel_gains[2], 8),
        "panels_with_positive_math_gain": sum(value > 0 for value in panel_gains),
        "base_qa_correct": qa_correct["base"],
        "adapter_qa_correct": qa_correct["lokr_3ep_lr1e4"],
        "imported_r3_protected_qa_regression": round(qa_regression, 8),
        "independent_rescore_match": independent_match,
        "service_and_embedding_restored": maintenance["service_and_embedding_restored"],
    }
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "independent_evaluation.json", {"match": independent_match, "metrics": metrics})
    write_json(raw / "clean_base_receipts.json", {
        "base_preexisting_peft_module_count": r5_payload["base_preexisting_peft_module_count"]
    })
    write_json(raw / "continuation_ledger.json", {
        "r3_worker_sha256": sha256_file(R3_WORKER),
        "r3_receipt_sha256": sha256_file(R3_RECEIPT),
        "r4_worker_sha256": sha256_file(R4_WORKER),
        "r4_receipt_sha256": sha256_file(R4_RECEIPT),
        "imported_math_panels": 2,
        "fresh_math_panels": 1,
        "tasks_per_panel": 256,
    })
    write_json(raw / "isolation_smoke.json", {
        "teacher_disjoint": True,
        "pairwise_disjoint": True,
        "panel_hashes": [canonical_json_sha256(ids) for ids in panel_ids],
        "fixed_final_panel": True,
    })
    write_json(raw / "paired_baseline.json", {
        "panel_correct_counts": panel_counts,
        "panel_math_gains": metrics["panel_math_gains"],
        "qa": qa_correct,
    })
    write_json(raw / "scorer_hashes.json", {
        "runner": sha256_file(pathlib.Path(__file__).resolve()),
        "r2_runner": sha256_file(ROOT / "tools/research/run_adapt01_broad_artifact_eval_r2.py"),
        "math": sha256_file(ROOT / "tools/analysis/a2_stats.py"),
        "qa": sha256_file(ROOT / "tools/benchmarks/normal_qa_ab.py"),
    })
    write_json(raw / "source_execution_receipt.json", {
        "source_task_ids": [
            "BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-03",
            "BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-04",
        ],
        "source_hashes": {
            "r3_worker": sha256_file(R3_WORKER),
            "r3_receipt": sha256_file(R3_RECEIPT),
            "r4_worker": sha256_file(R4_WORKER),
            "r4_receipt": sha256_file(R4_RECEIPT),
        },
    })
    write_json(raw / "wsl_environment.json", r5_payload["versions"] | {
        "gpu": r5_payload["gpu"],
        "peak_allocated_vram_gib": r5_payload["peak_allocated_vram_gib"],
    })
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for panel_index, payload in enumerate(payloads, 1):
            for arm in payload["arms"]:
                for sample in arm["math_samples"]:
                    stream.write(json.dumps({"source_panel": panel_index, "arm": arm["arm"], **sample}, ensure_ascii=False) + "\n")
        for arm in r3_payload["arms"]:
            for sample in arm["qa_samples"]:
                stream.write(json.dumps({"source_panel": "r3_qa", "arm": arm["arm"], **sample}, ensure_ascii=False) + "\n")

    gates = {
        "source_integrity": {"metric": "r3_r4_source_hashes_verified", "operator": "eq", "threshold": True, "actual": True, "pass": True},
        "artifact_identity": {"metric": "artifact_hashes_verified", "operator": "eq", "threshold": True, "actual": True, "pass": True},
        "panel_isolation": {"metric": "three_panels_pairwise_and_teacher_disjoint", "operator": "eq", "threshold": True, "actual": True, "pass": True},
        "evaluation_coverage": {"metric": "fresh_third_panel_paired_math_generations", "operator": "eq", "threshold": 512, "actual": metrics["fresh_third_panel_paired_math_generations"], "pass": metrics["fresh_third_panel_paired_math_generations"] == 512},
        "pooled_gain": {"metric": "stratified_bootstrap_95ci_lower_three_panel_math_gain", "operator": "gt", "threshold": 0.0, "actual": bootstrap["lower_95"], "pass": bootstrap["lower_95"] > 0.0},
        "third_panel_direction": {"metric": "third_panel_math_gain", "operator": "gt", "threshold": 0.0, "actual": metrics["third_panel_math_gain"], "pass": metrics["third_panel_math_gain"] > 0.0},
        "panel_repeatability": {"metric": "panels_with_positive_math_gain", "operator": "ge", "threshold": 2, "actual": metrics["panels_with_positive_math_gain"], "pass": metrics["panels_with_positive_math_gain"] >= 2},
        "protected_retention": {"metric": "imported_r3_protected_qa_regression", "operator": "le", "threshold": 0.05, "actual": metrics["imported_r3_protected_qa_regression"], "pass": metrics["imported_r3_protected_qa_regression"] <= 0.05},
        "independent_score": {"metric": "independent_rescore_match", "operator": "eq", "threshold": True, "actual": independent_match, "pass": independent_match is True},
        "service_recovery": {"metric": "service_and_embedding_restored", "operator": "eq", "threshold": True, "actual": maintenance["service_and_embedding_restored"], "pass": maintenance["service_and_embedding_restored"] is True},
    }
    evidence = {
        "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
        "artifact_hashes": "raw/artifact_hashes.json", "clean_base_receipts": "raw/clean_base_receipts.json",
        "continuation_ledger": "raw/continuation_ledger.json", "dataset_hashes": "raw/dataset_hashes.json",
        "independent_evaluation": "raw/independent_evaluation.json", "isolation_smoke": "raw/isolation_smoke.json",
        "model_hash": "raw/model_hash.json", "paired_baseline": "raw/paired_baseline.json",
        "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl",
        "receipt_fingerprint": "raw/receipt.json", "scorer_hashes": "raw/scorer_hashes.json",
        "service_maintenance": "raw/service_maintenance.json", "source_execution_receipt": "raw/source_execution_receipt.json",
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
        input_paths=[*EXPECTED_HASHES.keys(), PREREGISTRATION, worker_path, panel, *evidence_files],
        packages=["pytest"],
        runtime={
            "execution_mode": "fixed_final_third_panel_and_stratified_synthesis",
            "host_pid": os.getpid(),
            "timing_is_evidence": False,
            "no_additional_adaptive_panels": True,
        },
    )
    complete, errors = provenance_complete(provenance)
    if not complete or not independent_match:
        raise ValueError(f"evidence validation failed: provenance={errors}, scorer_match={independent_match}")
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
        "ADAPT01_384_ARTIFACT_THREE_PANEL_GAIN_R5"
        if not failed
        else "ADAPT01_384_ARTIFACT_THREE_PANEL_GAIN_NOT_CONFIRMED_R5"
    )
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Panel gains `{metrics['panel_math_gains']}`; equal-panel pooled gain "
        f"`{metrics['three_panel_math_gain']:.6f}` with stratified-bootstrap 95% "
        f"interval `[{bootstrap['lower_95']:.6f}, {bootstrap['upper_95']:.6f}]`. "
        f"Third-panel base `{panel_counts[2]['base']}/256`; adapter "
        f"`{panel_counts[2]['lokr_3ep_lr1e4']}/256`. Imported R3 QA base "
        f"`{qa_correct['base']}/48`; adapter `{qa_correct['lokr_3ep_lr1e4']}/48`. "
        f"Failed gates: `{', '.join(failed) if failed else 'none'}`. This family "
        f"stops after R5 regardless of outcome.\n",
        encoding="utf-8",
    )
    write_json(finalized / "complete.json", {
        "task_id": TASK_ID,
        "receipt_fingerprint": receipt["receipt_fingerprint"],
        "failed_gates": failed,
        "family_stop": True,
    })
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-out")
    parser.add_argument("--panel")
    parser.add_argument("--adapter")
    args = parser.parse_args()
    if args.worker_mode:
        if not all((args.worker_out, args.panel, args.adapter)):
            parser.error("worker mode requires output, panel and adapter")
        r2.worker(args.worker_out, args.panel, args.adapter)
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
