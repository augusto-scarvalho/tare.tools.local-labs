#!/usr/bin/env python3
"""Replicate the R3 LoKr gain on a second teacher-disjoint math panel."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
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
from tools.research import run_trace_distillation_training_r2 as training

TASK_ID = "BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-04"
SOURCE = ROOT / "runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-03"
SOURCE_WORKER = SOURCE / "raw/merged_worker.json"
SOURCE_RECEIPT = SOURCE / "raw/receipt.json"
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-04.json"
PREREGISTRATION = ROOT / "runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-04/PRE_REGISTRATION.md"
SECOND_PANEL_HASH = "4c88bd4c27eb8fea9240e11503ef781e744313c5de7b1f0391bb680f7e3379bd"
EXPECTED_HASHES = {
    ADMISSION: "2630de4c9f952b6fcfbdd54ea8c314d31465eb80bf81a99e0cc1491c57eecdcf",
    SOURCE_WORKER: "7beecfc4a2970ce39f6e5a8343d4d6b23fd5e78fa8328b03195f3e61acdb6b2f",
    SOURCE_RECEIPT: "3ca20a2dbad797cc6ce1629d501802c6fa06ee8d67c8ab948f2959c62b298f40",
    SOURCE / "PRE_REGISTRATION.md": "a4fd1091d47bcaf2da78cf0e719a2cab64ba935703f6bdbd2a02504c80642c4b",
    ROOT / "tools/research/run_adapt01_broad_artifact_eval_r3.py": "4aee5af93a5585977ce0dfe5c7b020218b4bc34130a9d1419607663f3b85794f",
    r2.ADAPTER / "adapter_model.safetensors": "7f6d082243f6b406259791dc15a65e4b092b48597fad9b68018d507872ad8fa7",
    r2.ADAPTER / "adapter_config.json": "08cf4d254e2a6c9aba9d34ba6a0c76926b478d7cd0ad771062acefb71a31d934",
    training.DEFAULT_MATH_PATH: "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    training.TEACHER_PATH: "dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e",
    training.DEFAULT_QA_PATH: "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
}


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def second_panel_ids() -> list[str]:
    teacher = json.loads(training.TEACHER_PATH.read_text(encoding="utf-8"))
    teacher_ids = {row["task_id"] for row in teacher}
    available = [
        f"gsm8k/{index}"
        for index in range(1319)
        if f"gsm8k/{index}" not in teacher_ids
    ]
    first = available[:256]
    second = available[256:512]
    if (
        len(second) != 256
        or set(first).intersection(second)
        or canonical_json_sha256(first) != r2.HELDOUT_HASH
        or canonical_json_sha256(second) != SECOND_PANEL_HASH
    ):
        raise ValueError("second held-out panel differs from preregistration")
    return second


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


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    finalized = raw / "finalized"
    finalized.mkdir(parents=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()
    sources = verify_sources()
    math_ids = second_panel_ids()
    qa_ids = r3.actual_qa_ids()
    source = json.loads(SOURCE_WORKER.read_text(encoding="utf-8"))
    source_arms = {arm["arm"]: arm for arm in source["arms"]}
    for arm_name in ("base", "lokr_3ep_lr1e4"):
        if len(source_arms[arm_name]["qa_samples"]) != 48:
            raise ValueError(f"R3 QA source incomplete for {arm_name}")
        if [sample["task_id"] for sample in source_arms[arm_name]["qa_samples"]] != qa_ids:
            raise ValueError(f"R3 QA order differs for {arm_name}")
    panel = raw / "panel.json"
    write_json(panel, {
        "math_ids": math_ids,
        "qa_ids": [],
        "math_id_sha256": canonical_json_sha256(math_ids),
    })
    write_json(raw / "artifact_hashes.json", sources)
    write_json(raw / "dataset_hashes.json", {
        "math": sources[training.DEFAULT_MATH_PATH.relative_to(ROOT).as_posix()],
        "teacher": sources[training.TEACHER_PATH.relative_to(ROOT).as_posix()],
        "qa": sources[training.DEFAULT_QA_PATH.relative_to(ROOT).as_posix()],
        "math_ids": math_ids,
        "math_id_sha256": canonical_json_sha256(math_ids),
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
    worker_path = raw / "fresh_math_worker.json"
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
            "--worker-mode",
            "--worker-out", training.windows_path_to_wsl(worker_path),
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
            raise RuntimeError(f"math worker failed ({completed.returncode}): {completed.stderr[-4000:]}")
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
        raise RuntimeError("math worker returned without output")

    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    combined = copy.deepcopy(worker)
    arms = {arm["arm"]: arm for arm in combined["arms"]}
    for arm_name in ("base", "lokr_3ep_lr1e4"):
        if len(arms[arm_name]["math_samples"]) != 256 or arms[arm_name]["qa_samples"]:
            raise ValueError(f"fresh worker dimensions invalid for {arm_name}")
        arms[arm_name]["qa_samples"] = copy.deepcopy(source_arms[arm_name]["qa_samples"])
        arms[arm_name]["qa_correct"] = sum(
            sample["correct"] for sample in arms[arm_name]["qa_samples"]
        )
    write_json(raw / "combined_worker.json", combined)
    math_map = {
        task["task_id"]: task
        for task in training.load_math_panel(training.DEFAULT_MATH_PATH, math_ids)
    }
    qa_map = {
        task["id"]: task
        for task in training.load_qa_panel(training.DEFAULT_QA_PATH, qa_ids)
    }
    independent_match = True
    differences = []
    for task_id in math_ids:
        values = {}
        for arm_name in ("base", "lokr_3ep_lr1e4"):
            sample = next(row for row in arms[arm_name]["math_samples"] if row["task_id"] == task_id)
            extracted = training.extract_gsm8k_pred(sample["output_text"])
            correct = int(training.is_gsm8k_correct(extracted, math_map[task_id]["gold"]))
            independent_match &= bool(correct) == bool(sample["correct"])
            values[arm_name] = correct
        differences.append(values["lokr_3ep_lr1e4"] - values["base"])
    for arm_name in ("base", "lokr_3ep_lr1e4"):
        for sample in arms[arm_name]["qa_samples"]:
            correct, _ = training.grade_qa(qa_map[sample["task_id"]], sample["output_text"])
            independent_match &= bool(correct) == bool(sample["correct"])
    bootstrap = r2.paired_bootstrap(differences)
    base_math = arms["base"]["math_correct"] / 256
    adapter_math = arms["lokr_3ep_lr1e4"]["math_correct"] / 256
    base_qa = arms["base"]["qa_correct"] / 48
    adapter_qa = arms["lokr_3ep_lr1e4"]["qa_correct"] / 48
    source_receipt = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    r3_gain = source_receipt["gates"]["broad_gain"]["actual"] > 0
    metrics = {
        "r3_source_hashes_verified": True,
        "artifact_hashes_verified": True,
        "second_panel_disjoint_from_teacher_and_r3": True,
        "fresh_paired_math_generations": sum(len(arm["math_samples"]) for arm in arms.values()),
        "base_math_correct": arms["base"]["math_correct"],
        "adapter_math_correct": arms["lokr_3ep_lr1e4"]["math_correct"],
        "math_gain": round(adapter_math - base_math, 8),
        "paired_bootstrap": bootstrap,
        "r3_and_r4_math_gain_positive": bool(r3_gain and adapter_math - base_math > 0),
        "base_qa_correct": arms["base"]["qa_correct"],
        "adapter_qa_correct": arms["lokr_3ep_lr1e4"]["qa_correct"],
        "imported_r3_protected_qa_regression": round(max(0.0, base_qa - adapter_qa), 8),
        "independent_rescore_match": independent_match,
        "service_and_embedding_restored": maintenance["service_and_embedding_restored"],
    }
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "independent_evaluation.json", {"match": independent_match, "metrics": metrics})
    write_json(raw / "clean_base_receipts.json", {
        "base_preexisting_peft_module_count": worker["base_preexisting_peft_module_count"]
    })
    write_json(raw / "continuation_ledger.json", {
        "source_worker_sha256": sha256_file(SOURCE_WORKER),
        "source_receipt_sha256": sha256_file(SOURCE_RECEIPT),
        "imported_qa_per_arm": 48,
        "fresh_math_per_arm": 256,
    })
    write_json(raw / "isolation_smoke.json", {
        "teacher_disjoint": True,
        "r3_panel_disjoint": True,
        "second_panel_sha256": canonical_json_sha256(math_ids),
        "r3_panel_sha256": r2.HELDOUT_HASH,
    })
    write_json(raw / "paired_baseline.json", {
        "base": {"math_correct": metrics["base_math_correct"], "qa_correct": metrics["base_qa_correct"]},
        "adapter": {"math_correct": metrics["adapter_math_correct"], "qa_correct": metrics["adapter_qa_correct"]},
    })
    write_json(raw / "scorer_hashes.json", {
        "runner": sha256_file(pathlib.Path(__file__).resolve()),
        "r2_runner": sha256_file(ROOT / "tools/research/run_adapt01_broad_artifact_eval_r2.py"),
        "math": sha256_file(ROOT / "tools/analysis/a2_stats.py"),
        "qa": sha256_file(ROOT / "tools/benchmarks/normal_qa_ab.py"),
    })
    write_json(raw / "source_execution_receipt.json", {
        "source_task_id": "BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-03",
        "source_worker_sha256": sha256_file(SOURCE_WORKER),
        "source_receipt_sha256": sha256_file(SOURCE_RECEIPT),
    })
    write_json(raw / "wsl_environment.json", worker["versions"] | {
        "gpu": worker["gpu"],
        "peak_allocated_vram_gib": worker["peak_allocated_vram_gib"],
    })
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for arm in combined["arms"]:
            for sample in [*arm["math_samples"], *arm["qa_samples"]]:
                stream.write(json.dumps({"arm": arm["arm"], **sample}, ensure_ascii=False) + "\n")

    gates = {
        "source_integrity": {"metric": "r3_source_hashes_verified", "operator": "eq", "threshold": True, "actual": True, "pass": True},
        "artifact_identity": {"metric": "artifact_hashes_verified", "operator": "eq", "threshold": True, "actual": True, "pass": True},
        "panel_isolation": {"metric": "second_panel_disjoint_from_teacher_and_r3", "operator": "eq", "threshold": True, "actual": True, "pass": True},
        "evaluation_coverage": {"metric": "fresh_paired_math_generations", "operator": "eq", "threshold": 512, "actual": metrics["fresh_paired_math_generations"], "pass": metrics["fresh_paired_math_generations"] == 512},
        "replicated_gain": {"metric": "paired_bootstrap_95ci_lower_math_gain", "operator": "gt", "threshold": 0.0, "actual": bootstrap["lower_95"], "pass": bootstrap["lower_95"] > 0.0},
        "directional_consistency": {"metric": "r3_and_r4_math_gain_positive", "operator": "eq", "threshold": True, "actual": metrics["r3_and_r4_math_gain_positive"], "pass": metrics["r3_and_r4_math_gain_positive"] is True},
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
            "execution_mode": "fresh_second_panel_replication",
            "host_pid": os.getpid(),
            "timing_is_evidence": False,
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
        "ADAPT01_384_ARTIFACT_SECOND_PANEL_GAIN_R4"
        if not failed
        else "ADAPT01_384_ARTIFACT_SECOND_PANEL_GAIN_NOT_CONFIRMED_R4"
    )
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Second-panel base math `{metrics['base_math_correct']}/256`; adapter math "
        f"`{metrics['adapter_math_correct']}/256`; gain `{metrics['math_gain']:.6f}` with "
        f"paired bootstrap 95% interval `[{bootstrap['lower_95']:.6f}, "
        f"{bootstrap['upper_95']:.6f}]`. Imported R3 QA base "
        f"`{metrics['base_qa_correct']}/48`; adapter `{metrics['adapter_qa_correct']}/48`. "
        f"Failed gates: `{', '.join(failed) if failed else 'none'}`.\n",
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
