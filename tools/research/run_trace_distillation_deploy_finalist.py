#!/usr/bin/env python3
"""Select a trace finalist from prior panels and validate it on a third panel."""
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
from tools.research import run_trace_distillation_replication_r8 as r8

TASK_ID = "BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01"
SELECTED_SEED = 20260832
THIRD_PANEL_HASH = "73024245450c6158c150d654243e30ef26e562027dcc6514abd096862d0a69fe"
BOOTSTRAP_SEED = 2026082711
BOOTSTRAP_REPLICATES = 20_000
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01.json"
PREREGISTRATION = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/PRE_REGISTRATION.md"
R8_RUN = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-08"
EXPECTED_HASHES = {
    ADMISSION: "220160c63e606ce905f80c0c07b41d163c1b73da1e3b57805fb209b2c1357beb",
    PREREGISTRATION: "6b4a4d7eabc1a2f6f9e90e7f56b697b2b153eb4d01fa1d4c50ee0f0da33aef0d",
    r8.CHECKPOINT_LEDGER: "57364aaba37c39771aaf216950bfff6df1282735b641b0c682ded89ffa8aaf4c",
    r8.SOURCE_SCORES: "0171dcfcd70334a780a16337469f200656ec3b1d7c567889393c419cad9bae1e",
    r8.SOURCE_STUDENTS: "5283e8e1a66227d71d7a0c5847bd2c147397f580cfaa9c22520edfc65128e19b",
    r8.SOURCE_TRAINING: "5c3f0d5fd80d97351839bca1e38685e5e21b3357dfa56077f44f02b857bfe4cc",
    r8.SOURCE_RECEIPT: "782d9e58a97c5ac55dd6ebc2d62c67f9e003af415fb62e14ac8124718ea93b3a",
    R8_RUN / "raw/actual_scores.json": "15d359d9701a10ed449b8b325c1be93bb002a1ba685ec1bb97c21f3f30efed45",
    R8_RUN / "raw/dataset_hashes.json": "934dfa0b0e45ce73deab24a2e1ee5684a7a223843290296755e8eb83f5e19171",
    R8_RUN / "raw/receipt.json": "eb4cff3c9d5022887f2621bdf0c303b4aca807e9449e8c600860bd72a046b990",
    ROOT / "tools/research/run_trace_distillation_replication_r8.py": "0ad1f687c8ed1b9f0d923a61fa853a47ece9b35dbaaaa0b72b483e9793cbcbec",
    ROOT / "workloads/gsm8k.jsonl": "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
}


def verify_sources() -> tuple[dict[str, Any], dict[str, Any], list[pathlib.Path]]:
    static: dict[str, Any] = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen source mismatch: {path}: {actual} != {expected}")
        static[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": actual}
    _, checkpoints = r8.verify_sources()
    return static, checkpoints, list(EXPECTED_HASHES)


def select_seed() -> tuple[int, list[dict[str, Any]]]:
    r7 = json.loads(r8.SOURCE_SCORES.read_text(encoding="utf-8"))
    r8_scores = json.loads((R8_RUN / "raw/actual_scores.json").read_text(encoding="utf-8"))
    rows = []
    for prior in r7["seeds"]:
        if prior["qa_regression"] > 0:
            continue
        replication = next(row for row in r8_scores["seeds"] if row["seed"] == prior["seed"])
        rows.append({"seed": prior["seed"], "r7_trace_correct": prior["trace_math_correct"],
                     "r8_trace_correct": replication["trace_math_correct"],
                     "combined_trace_correct": prior["trace_math_correct"] + replication["trace_math_correct"],
                     "r7_qa_regression": prior["qa_regression"]})
    selected = max(rows, key=lambda row: (row["combined_trace_correct"], -row["seed"]))["seed"]
    if selected != SELECTED_SEED:
        raise ValueError(f"frozen selection rule changed: {selected}")
    return selected, rows


def third_panel_ids() -> tuple[list[str], list[str], list[str]]:
    first, second = r8.panel_ids()
    teacher = json.loads(r8.r2.TEACHER_PATH.read_text(encoding="utf-8"))
    teacher_ids = {row["task_id"] for row in teacher}
    available = [f"gsm8k/{index}" for index in range(1319)
                 if f"gsm8k/{index}" not in teacher_ids]
    third = available[512:768]
    training = json.loads(r8.SOURCE_TRAINING.read_text(encoding="utf-8"))
    training_ids = {row["task_id"] for row in training["pool"]}
    if (len(third) != 256 or canonical_json_sha256(third) != THIRD_PANEL_HASH
            or set(third) & set(first) or set(third) & set(second)
            or set(third) & teacher_ids or set(third) & training_ids):
        raise ValueError("third panel is not frozen and fully isolated")
    return first, second, third


def paired_score(payloads: list[dict[str, Any]], ids: list[str]) -> tuple[dict[str, Any], bool]:
    tasks = {task["task_id"]: task for task in r8.r2.load_math_panel(r8.r2.DEFAULT_MATH_PATH, ids)}
    arms = {payload["arm"]: payload for payload in payloads}
    if set(arms) != set(r8.ARMS):
        raise ValueError("selected checkpoint pair is incomplete")
    values: dict[str, dict[str, int]] = {}
    independent = True
    for arm in r8.ARMS:
        samples = arms[arm]["math_samples"]
        if [sample["task_id"] for sample in samples] != ids:
            raise ValueError(f"panel order mismatch for {arm}")
        values[arm] = {}
        for sample in samples:
            prediction = r8.r2.extract_gsm8k_pred(sample["output_text"])
            correct = int(r8.r2.is_gsm8k_correct(prediction, tasks[sample["task_id"]]["gold"]))
            independent &= bool(correct) == bool(sample["correct"])
            values[arm][sample["task_id"]] = correct
    differences = [values["full_trace"][task] - values["answer_only"][task] for task in ids]
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = [sum(differences[rng.randrange(256)] for _ in range(256)) / 256
                 for _ in range(BOOTSTRAP_REPLICATES)]
    estimates.sort()
    answer_correct = sum(values["answer_only"].values())
    trace_correct = sum(values["full_trace"].values())
    return {
        "answer_math_correct": answer_correct,
        "trace_math_correct": trace_correct,
        "answer_accuracy": answer_correct / 256,
        "trace_accuracy": trace_correct / 256,
        "trace_minus_answer": (trace_correct - answer_correct) / 256,
        "paired_bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                             "lower_95": estimates[int(0.025 * BOOTSTRAP_REPLICATES)],
                             "upper_95": estimates[int(0.975 * BOOTSTRAP_REPLICATES)]},
        "trace_only_correct": sum(value == 1 for value in differences),
        "answer_only_correct": sum(value == -1 for value in differences),
    }, independent


def selected_qa() -> tuple[float, bool, dict[str, Any]]:
    _, independent, rows = r8.imported_qa_regression()
    selected = next(row for row in rows if row["seed"] == SELECTED_SEED)
    return selected["qa_regression"], independent, selected


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    workers, finalized = raw / "workers", raw / "finalized"
    workers.mkdir(parents=True)
    finalized.mkdir(parents=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()
    static, checkpoints, frozen_paths = verify_sources()
    selected, selection_rows = select_seed()
    first, second, third = third_panel_ids()
    panel = raw / "panel.json"
    r8.write_json(panel, {"math_ids": third, "math_id_sha256": canonical_json_sha256(third)})
    selected_checkpoints = {f"seed_{selected}_{arm}": checkpoints[f"seed_{selected}_{arm}"] for arm in r8.ARMS}
    r8.write_json(raw / "checkpoint_hashes.json", selected_checkpoints)
    r8.write_json(raw / "dataset_hashes.json", {"static": static,
                  "first_panel_sha256": canonical_json_sha256(first),
                  "second_panel_sha256": canonical_json_sha256(second),
                  "third_panel_ids": third, "third_panel_sha256": canonical_json_sha256(third),
                  "all_panels_disjoint": not (set(first) & set(second) or set(first) & set(third) or set(second) & set(third))})
    r8.write_json(raw / "model_hash.json", r8.r2.verify_base_model())

    initial_service = r8.r2.query_service()
    initial_gpu = r8.r2.query_gpu()
    initial_embedding = r8.r2.http_get_json("http://127.0.0.1:8081/health")
    maintenance: dict[str, Any] = {"initial_service": initial_service, "initial_gpu": initial_gpu,
        "initial_embedding": initial_embedding, "service_stopped_for_vram": False}
    service_stopped = False
    payloads: list[dict[str, Any]] = []
    error: Exception | None = None
    try:
        if initial_gpu["memory_free_mib"] < 12_000 and initial_service["active_state"] == "active":
            r8.long_systemctl("stop")
            service_stopped = True
            maintenance["service_stopped_for_vram"] = True
        maintenance["service_after_stop"] = r8.r2.query_service()
        maintenance["embedding_after_stop"] = r8.r2.http_get_json("http://127.0.0.1:8081/health")
        maintenance["gpu_after_stop"] = r8.r2.query_gpu()
        if maintenance["embedding_after_stop"].get("status") != "ok" or maintenance["gpu_after_stop"]["memory_free_mib"] < 12_000:
            raise RuntimeError("safe worker preconditions not met")
        for arm in r8.ARMS:
            label = f"seed_{selected}_{arm}"
            output = workers / f"{label}.json"
            command = r8.wsl_command(output, r8.CHECKPOINTS / label, panel, arm, selected)
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                       errors="replace", timeout=7200, check=False)
            (workers / f"{label}.stdout.log").write_text(completed.stdout, encoding="utf-8")
            (workers / f"{label}.stderr.log").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode:
                raise RuntimeError(f"worker {label} failed: {completed.stderr[-4000:]}")
            payload = json.loads(output.read_text(encoding="utf-8"))
            if payload["math_total"] != 256:
                raise ValueError(f"worker {label} incomplete")
            payloads.append(payload)
            r8.write_json(finalized / f"{label}.json", {"label": label,
                          "worker_sha256": sha256_file(output), "math_correct": payload["math_correct"]})
    except Exception as caught:
        error = caught
    finally:
        if service_stopped:
            r8.long_systemctl("start")
            maintenance["inference_health_final"] = r8.r2.wait_for_health("http://127.0.0.1:8080/health", timeout_seconds=180)
        maintenance["final_service"] = r8.r2.query_service()
        maintenance["final_embedding"] = r8.r2.wait_for_health("http://127.0.0.1:8081/health", timeout_seconds=30)
        maintenance["final_gpu"] = r8.r2.query_gpu()
        maintenance["service_and_embedding_restored"] = (
            maintenance["final_service"]["active_state"] == initial_service["active_state"]
            and r8.r2.normalize_exec_start(maintenance["final_service"]["exec_start"])
            == r8.r2.normalize_exec_start(initial_service["exec_start"])
            and maintenance["final_service"]["n_restarts"] == initial_service["n_restarts"]
            and maintenance["final_embedding"].get("status") == "ok"
            and (not service_stopped or maintenance.get("inference_health_final", {}).get("status") == "ok"))
        r8.write_json(raw / "service_maintenance.json", maintenance)
    if error:
        raise error

    scores, math_match = paired_score(payloads, third)
    qa_regression, qa_match, qa_row = selected_qa()
    independent_match = math_match and qa_match
    metrics = {**scores, "r7_r8_sources_and_checkpoints_verified": True,
               "selected_seed": selected, "selection_candidates": selection_rows,
               "third_panel_disjoint_from_training_and_prior_panels": True,
               "immutable_checkpoints_evaluated": len(payloads),
               "fresh_third_panel_generations": sum(payload["math_total"] for payload in payloads),
               "imported_selected_seed_qa_regression": qa_regression,
               "independent_rescore_match": independent_match,
               "service_and_embedding_restored": maintenance["service_and_embedding_restored"]}
    r8.write_json(raw / "actual_scores.json", metrics)
    r8.write_json(raw / "independent_evaluation.json", {"math": scores, "qa": qa_row,
                  "independent_rescore_match": independent_match})
    r8.write_json(raw / "paired_baseline.json", {"seed": selected,
                  "baseline": "answer_only", "treatment": "full_trace",
                  "comparison": scores})
    r8.write_json(raw / "source_execution_receipt.json", {
        "source_task_id": "BACKLOG-ADAPT-TRACE-DISTILL-08",
        "receipt_sha256": sha256_file(R8_RUN / "raw/receipt.json"),
        "receipt_fingerprint": json.loads((R8_RUN / "raw/receipt.json").read_text(encoding="utf-8"))["receipt_fingerprint"]})
    r8.write_json(raw / "student_samples.json", [{"arm": payload["arm"], "seed": selected,
                  "math_samples": payload["math_samples"]} for payload in payloads])
    r8.write_json(raw / "teacher_samples.json", {"teacher_source": r8.r2.TEACHER_PATH.relative_to(ROOT).as_posix(),
                  "teacher_sha256": sha256_file(r8.r2.TEACHER_PATH), "teacher_rows_used_for_third_panel": 0})
    r8.write_json(raw / "wsl_environment.json", {"versions": payloads[0]["versions"],
                  "gpu": payloads[0]["gpu"], "worker_count": 2,
                  "peak_allocated_vram_gib_max": max(payload["peak_allocated_vram_gib"] for payload in payloads)})
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for payload in payloads:
            for sample in payload["math_samples"]:
                stream.write(json.dumps({"arm": payload["arm"], "seed": selected, **sample}, ensure_ascii=False) + "\n")

    definitions = {
        "source_integrity": ("r7_r8_sources_and_checkpoints_verified", "eq", True),
        "selection_reproducibility": ("selected_seed", "eq", SELECTED_SEED),
        "panel_isolation": ("third_panel_disjoint_from_training_and_prior_panels", "eq", True),
        "checkpoint_coverage": ("immutable_checkpoints_evaluated", "eq", 2),
        "evaluation_coverage": ("fresh_third_panel_generations", "eq", 512),
        "finalist_gain": ("paired_bootstrap_95ci_lower_trace_minus_answer", "gt", 0.0),
        "finalist_absolute": ("trace_third_panel_accuracy", "ge", 0.40),
        "protected_retention": ("imported_selected_seed_qa_regression", "le", 0.05),
        "independent_score": ("independent_rescore_match", "eq", True),
        "service_recovery": ("service_and_embedding_restored", "eq", True),
    }
    actuals = {**metrics,
               "paired_bootstrap_95ci_lower_trace_minus_answer": scores["paired_bootstrap"]["lower_95"],
               "trace_third_panel_accuracy": scores["trace_accuracy"]}
    gates: dict[str, Any] = {}
    for gate_id, (metric, operator, threshold) in definitions.items():
        actual = actuals[metric]
        passed = actual == threshold if operator == "eq" else actual > threshold if operator == "gt" else actual >= threshold if operator == "ge" else actual <= threshold
        gates[gate_id] = {"metric": metric, "operator": operator, "threshold": threshold, "actual": actual, "pass": passed}
    evidence = {"acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
                "checkpoint_hashes": "raw/checkpoint_hashes.json", "dataset_hashes": "raw/dataset_hashes.json",
                "independent_evaluation": "raw/independent_evaluation.json", "model_hash": "raw/model_hash.json",
                "paired_baseline": "raw/paired_baseline.json", "provenance": "raw/receipt.json",
                "raw_samples": "raw/samples.jsonl", "receipt_fingerprint": "raw/receipt.json",
                "service_maintenance": "raw/service_maintenance.json", "source_execution_receipt": "raw/source_execution_receipt.json",
                "student_samples": "raw/student_samples.json", "teacher_samples": "raw/teacher_samples.json",
                "wsl_environment": "raw/wsl_environment.json"}
    evidence_files = sorted({raw / value.removeprefix("raw/") for value in evidence.values() if value != "raw/receipt.json"})
    provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(), started_at_utc=started,
        started_monotonic=mono, input_paths=[*frozen_paths, panel, *workers.glob("*.json"), *evidence_files],
        packages=["pytest"], runtime={"execution_mode": "selected_trace_finalist_third_panel",
        "host_pid": os.getpid(), "selected_seed": selected, "checkpoint_count": 2,
        "fresh_generation_count": 512, "timing_is_evidence": False})
    complete, errors = provenance_complete(provenance)
    if not complete or not independent_match:
        raise ValueError(f"evidence validation failed: {errors}, independent={independent_match}")
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID,
               "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    r8.write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = ("TRACE_DISTILLATION_DEPLOYMENT_FINALIST_CONFIRMED_R1" if not failed
             else "TRACE_DISTILLATION_DEPLOYMENT_FINALIST_NOT_CONFIRMED_R1")
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Selected seed `{selected}` before the third panel. Answer-only/trace scored "
        f"`{scores['answer_math_correct']}/256` and `{scores['trace_math_correct']}/256`; "
        f"gain `{scores['trace_minus_answer']:.6f}` with paired-bootstrap 95% interval "
        f"`[{scores['paired_bootstrap']['lower_95']:.6f}, {scores['paired_bootstrap']['upper_95']:.6f}]`. "
        f"Imported QA regression `{qa_regression:.6f}`. Failed gates: "
        f"`{', '.join(failed) if failed else 'none'}`. The finalist-selection family stops here.\n",
        encoding="utf-8", newline="\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        assert select_seed()[0] == SELECTED_SEED
        assert canonical_json_sha256(third_panel_ids()[2]) == THIRD_PANEL_HASH
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
