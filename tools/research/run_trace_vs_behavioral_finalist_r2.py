#!/usr/bin/env python3
"""Compare the selected trace candidate with both behavioral finalists on panel 3."""
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
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)
from tools.research import run_trace_distillation_deploy_finalist as deploy
from tools.research import run_trace_distillation_replication_r8 as r8
from tools.research import run_trace_vs_behavioral_finalist as prior

TASK_ID = "BACKLOG-ADAPT-TRACE-VS-FINALIST-02"
SELECTED_TRACE_SEED = 20260832
BEHAVIOR_SEEDS = [20260824, 20260825]
BOOTSTRAP_SEED = 2026082817
BOOTSTRAP_REPLICATES = 20_000
THIRD_PANEL_HASH = "73024245450c6158c150d654243e30ef26e562027dcc6514abd096862d0a69fe"
TRACE_RUN = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01"
PRIOR_RUN = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-VS-FINALIST-01"
TRAIN_RUN = ROOT / "runs/research/BACKLOG-ADAPT-TRAIN-01"
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-ADAPT-TRACE-VS-FINALIST-02.json"
PREREGISTRATION = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-VS-FINALIST-02/PRE_REGISTRATION.md"

EXPECTED_HASHES = {
    ADMISSION: "360c5e626bb363860d0dd675611c90519c10345b864d5051b52225f29122c8d1",
    PREREGISTRATION: "95cde6cada7a73e3b3f7b1601e4dfb699e2ce421a52712c636c6679b3f4aa2aa",
    TRACE_RUN / "raw/receipt.json": "b4fc924a1542e4913c3c1d70fdf77f8bb9be0e2662b8757d0d06f82b60d3f521",
    TRACE_RUN / "raw/student_samples.json": "288270e4faa780bbd905b593193bf9c9edc595d84bf41cc2ef3fd72ba53663c9",
    TRACE_RUN / "raw/dataset_hashes.json": "f3bd82ee0aef9b7eb7669d7ed6bc5549412c8a5ebdec33bb4496bb869c95661c",
    PRIOR_RUN / "raw/receipt.json": "54ada49af2a437513f47b766dc2f6fd9a71b93e88d06e48ae896b6cca88a1487",
    PRIOR_RUN / "raw/student_samples.json": "b94d98cd6f5f356e7011ca9ca11186d15cb40923059ef3ceaf4ba93492d3abd5",
    TRAIN_RUN / "raw/receipt.json": "903c723f3d63130cf06a5e501498451beee0cee34a8aa71d6f9de36faeb602b8",
    ROOT / "tools/research/run_trace_vs_behavioral_finalist.py": "a1cdb8766699108effcd17e53b681667bbe7757ca0e3371f712c9d3f8d7b6ff6",
    ROOT / "tools/research/run_trace_distillation_replication_r8.py": "0ad1f687c8ed1b9f0d923a61fa853a47ece9b35dbaaaa0b72b483e9793cbcbec",
    prior.CHECKPOINTS[20260824] / "adapter_config.json": "4872214e344ce266b0b10a1f70db33775224441ff1d78dbe0c7fb4dd70aeac84",
    prior.CHECKPOINTS[20260824] / "adapter_model.safetensors": "05b80090d2d1ba751d48a5032cddec82819a79b9bfb5bd8b05306b85d6ef0122",
    prior.CHECKPOINTS[20260825] / "adapter_config.json": "4872214e344ce266b0b10a1f70db33775224441ff1d78dbe0c7fb4dd70aeac84",
    prior.CHECKPOINTS[20260825] / "adapter_model.safetensors": "433978a1b942b4a6d8150e40ca067d2615f811ab8ad2ff880e9a161c655c5646",
    r8.r2.DEFAULT_MATH_PATH: "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
}


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_sources() -> dict[str, Any]:
    ledger: dict[str, Any] = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen source mismatch: {path}: {actual} != {expected}")
        ledger[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": actual}
    return ledger


def load_frozen_inputs() -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    _, _, third_ids = deploy.third_panel_ids()
    if canonical_json_sha256(third_ids) != THIRD_PANEL_HASH:
        raise ValueError("third panel hash changed")

    trace_rows = json.loads((TRACE_RUN / "raw/student_samples.json").read_text(encoding="utf-8"))
    trace = next(
        row for row in trace_rows
        if row["seed"] == SELECTED_TRACE_SEED and row["arm"] == "full_trace"
    )
    if [sample["task_id"] for sample in trace["math_samples"]] != third_ids:
        raise ValueError("selected trace rows do not match the frozen third panel")
    if len(trace["math_samples"]) != 256:
        raise ValueError("selected trace import is incomplete")

    prior_rows = json.loads((PRIOR_RUN / "raw/student_samples.json").read_text(encoding="utf-8"))
    trace_qa = next(row for row in prior_rows["imported_trace"] if row["seed"] == SELECTED_TRACE_SEED)
    behavior_qa = [
        next(row for row in prior_rows["fresh_behavioral"] if row["seed"] == seed)
        for seed in BEHAVIOR_SEEDS
    ]
    qa = {
        "trace_seed": SELECTED_TRACE_SEED,
        "trace_correct": sum(bool(sample["correct"]) for sample in trace_qa["qa_samples"]),
        "trace_total": len(trace_qa["qa_samples"]),
        "behavioral": [
            {"seed": row["seed"], "correct": sum(bool(sample["correct"]) for sample in row["qa_samples"]),
             "total": len(row["qa_samples"])}
            for row in behavior_qa
        ],
    }
    if (qa["trace_correct"], qa["trace_total"]) != (11, 48):
        raise ValueError("frozen trace QA score changed")
    if [(row["correct"], row["total"]) for row in qa["behavioral"]] != [(12, 48), (10, 48)]:
        raise ValueError("frozen behavioral QA scores changed")
    return third_ids, trace, qa


def independent_math_values(item: dict[str, Any], ids: list[str]) -> tuple[list[int], bool]:
    tasks = {task["task_id"]: task for task in r8.r2.load_math_panel(r8.r2.DEFAULT_MATH_PATH, ids)}
    samples = item["math_samples"]
    if [sample["task_id"] for sample in samples] != ids:
        raise ValueError(f"math panel order mismatch for seed {item['seed']}")
    values: list[int] = []
    match = True
    for sample in samples:
        prediction = r8.r2.extract_gsm8k_pred(sample["output_text"])
        correct = int(r8.r2.is_gsm8k_correct(prediction, tasks[sample["task_id"]]["gold"]))
        match &= bool(correct) == bool(sample["correct"])
        values.append(correct)
    return values, match


def hierarchical_bootstrap(
    trace_values: list[int], behavior_values: list[list[int]], replicates: int = BOOTSTRAP_REPLICATES
) -> dict[str, Any]:
    if len(trace_values) != 256 or len(behavior_values) != 2 or any(len(row) != 256 for row in behavior_values):
        raise ValueError("bootstrap dimensions do not match preregistration")
    rng = random.Random(BOOTSTRAP_SEED)
    estimates: list[float] = []
    for _ in range(replicates):
        prompts = [rng.randrange(256) for _ in range(256)]
        seed_draws = [rng.randrange(2) for _ in range(2)]
        trace_mean = sum(trace_values[prompt] for prompt in prompts) / 256
        behavior_mean = sum(
            behavior_values[seed][prompt] for seed in seed_draws for prompt in prompts
        ) / 512
        estimates.append(trace_mean - behavior_mean)
    estimates.sort()
    return {
        "replicates": replicates,
        "seed": BOOTSTRAP_SEED,
        "lower_95": round(estimates[int(0.025 * replicates)], 8),
        "upper_95": round(estimates[min(replicates - 1, int(0.975 * replicates))], 8),
    }


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    if any(path.is_file() for path in raw.rglob("*")):
        raise RuntimeError("raw directory is not empty")
    workers = raw / "workers"
    finalized = raw / "finalized"
    workers.mkdir(parents=True, exist_ok=True)
    finalized.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()

    sources = verify_sources()
    third_ids, trace, qa = load_frozen_inputs()
    panel_path = raw / "panel.json"
    write_json(panel_path, {"math_ids": third_ids, "math_id_sha256": canonical_json_sha256(third_ids)})
    write_json(raw / "artifact_hashes.json", sources)
    write_json(raw / "dataset_hashes.json", {
        "third_panel_ids": third_ids,
        "third_panel_sha256": canonical_json_sha256(third_ids),
        "math_sha256": sha256_file(r8.r2.DEFAULT_MATH_PATH),
        "third_panel_disjoint_from_training_and_prior_panels": True,
    })
    write_json(raw / "model_hash.json", r8.r2.verify_base_model())

    initial_service = r8.r2.query_service()
    initial_gpu = r8.r2.query_gpu()
    initial_embedding = r8.r2.http_get_json("http://127.0.0.1:8081/health")
    maintenance: dict[str, Any] = {
        "initial_service": initial_service,
        "initial_gpu": initial_gpu,
        "initial_embedding": initial_embedding,
        "service_stopped_for_vram": False,
    }
    stopped = False
    behavior: list[dict[str, Any]] = []
    caught: Exception | None = None
    try:
        if initial_gpu["memory_free_mib"] < 12_000 and initial_service["active_state"] == "active":
            prior.long_systemctl("stop")
            stopped = True
            maintenance["service_stopped_for_vram"] = True
        maintenance["service_after_stop"] = r8.r2.query_service()
        maintenance["embedding_after_stop"] = r8.r2.http_get_json("http://127.0.0.1:8081/health")
        maintenance["gpu_after_stop"] = r8.r2.query_gpu()
        if maintenance["embedding_after_stop"].get("status") != "ok" or maintenance["gpu_after_stop"]["memory_free_mib"] < 12_000:
            raise RuntimeError("maintenance health or VRAM gate failed")

        for seed in BEHAVIOR_SEEDS:
            output = workers / f"behavior_seed_{seed}.json"
            # The reused R8 worker validates its arm vocabulary even though this
            # experiment treats the checkpoint family as the comparison label.
            command = r8.wsl_command(output, prior.CHECKPOINTS[seed], panel_path, "answer_only", seed)
            completed = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=7200, check=False,
            )
            (workers / f"behavior_seed_{seed}.stdout.log").write_text(completed.stdout, encoding="utf-8")
            (workers / f"behavior_seed_{seed}.stderr.log").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode:
                raise RuntimeError(f"behavior worker {seed} failed: {completed.stderr[-4000:]}")
            payload = json.loads(output.read_text(encoding="utf-8"))
            if payload["math_total"] != 256:
                raise ValueError(f"behavior worker {seed} incomplete")
            payload["family"] = "behavioral_finalist"
            behavior.append(payload)
            write_json(finalized / f"behavior_seed_{seed}.json", {
                "seed": seed, "worker_sha256": sha256_file(output),
                "math_correct": payload["math_correct"], "math_total": payload["math_total"],
            })
            print(f"[HOST] behavior seed {seed}: math={payload['math_correct']}/256", flush=True)
    except Exception as error:
        caught = error
    finally:
        if stopped:
            prior.long_systemctl("start")
            maintenance["inference_health_final"] = r8.r2.wait_for_health(
                "http://127.0.0.1:8080/health", timeout_seconds=180
            )
        maintenance["final_service"] = r8.r2.query_service()
        maintenance["final_embedding"] = r8.r2.wait_for_health("http://127.0.0.1:8081/health", timeout_seconds=30)
        maintenance["final_gpu"] = r8.r2.query_gpu()
        maintenance["service_and_embedding_restored"] = (
            maintenance["final_service"]["active_state"] == initial_service["active_state"]
            and r8.r2.normalize_exec_start(maintenance["final_service"]["exec_start"])
            == r8.r2.normalize_exec_start(initial_service["exec_start"])
            and maintenance["final_service"]["n_restarts"] == initial_service["n_restarts"]
            and maintenance["final_embedding"].get("status") == "ok"
            and (not stopped or maintenance.get("inference_health_final", {}).get("status") == "ok")
        )
        write_json(raw / "service_maintenance.json", maintenance)
    if caught:
        raise caught

    trace_values, trace_match = independent_math_values(trace, third_ids)
    behavior_values: list[list[int]] = []
    behavior_match = True
    for item in behavior:
        values, match = independent_math_values(item, third_ids)
        behavior_values.append(values)
        behavior_match &= match
    independent = trace_match and behavior_match
    trace_correct = sum(trace_values)
    behavior_correct = [sum(row) for row in behavior_values]
    trace_accuracy = trace_correct / 256
    behavior_accuracies = [score / 256 for score in behavior_correct]
    behavior_mean = statistics.mean(behavior_accuracies)
    math_delta = trace_accuracy - behavior_mean
    trace_qa_accuracy = qa["trace_correct"] / qa["trace_total"]
    behavior_qa_accuracies = [row["correct"] / row["total"] for row in qa["behavioral"]]
    qa_delta = trace_qa_accuracy - statistics.mean(behavior_qa_accuracies)
    bootstrap = hierarchical_bootstrap(trace_values, behavior_values)
    scores = {
        "selected_trace_seed": SELECTED_TRACE_SEED,
        "selected_trace_math_correct": trace_correct,
        "selected_trace_math_accuracy": round(trace_accuracy, 8),
        "behavioral_seed_math_correct": dict(zip(map(str, BEHAVIOR_SEEDS), behavior_correct)),
        "behavioral_seed_math_accuracy": dict(zip(map(str, BEHAVIOR_SEEDS), behavior_accuracies)),
        "behavioral_mean_math_accuracy": round(behavior_mean, 8),
        "selected_trace_minus_behavioral_mean_math": round(math_delta, 8),
        "selected_trace_qa_accuracy": round(trace_qa_accuracy, 8),
        "behavioral_mean_qa_accuracy": round(statistics.mean(behavior_qa_accuracies), 8),
        "selected_trace_minus_behavioral_mean_qa": round(qa_delta, 8),
        "hierarchical_bootstrap": bootstrap,
    }
    metrics = {
        **scores,
        "all_source_and_checkpoint_hashes_verified": True,
        "third_panel_disjoint_from_training_and_prior_panels": True,
        "imported_selected_trace_generations": len(trace["math_samples"]),
        "behavioral_checkpoints_evaluated": len(behavior),
        "fresh_behavioral_generations": sum(item["math_total"] for item in behavior),
        "hierarchical_bootstrap_95ci_lower_trace_minus_behavioral_math": bootstrap["lower_95"],
        "independent_rescore_match": independent,
        "service_and_embedding_restored": maintenance["service_and_embedding_restored"],
    }
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "independent_evaluation.json", {"match": independent, "scores": scores})
    write_json(raw / "paired_baseline.json", {
        "panel": "third", "trace_seed": SELECTED_TRACE_SEED,
        "behavior_seeds": BEHAVIOR_SEEDS, "comparison": scores,
    })
    write_json(raw / "continuation_ledger.json", {
        "trace_generation_source": "BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01",
        "qa_source": "BACKLOG-ADAPT-TRACE-VS-FINALIST-01",
        "trace_seed_preselected": SELECTED_TRACE_SEED,
        "fresh_behavioral_rows": 512,
    })
    write_json(raw / "source_execution_receipt.json", {
        "trace_receipt_sha256": sha256_file(TRACE_RUN / "raw/receipt.json"),
        "prior_comparison_receipt_sha256": sha256_file(PRIOR_RUN / "raw/receipt.json"),
        "behavioral_training_receipt_sha256": sha256_file(TRAIN_RUN / "raw/receipt.json"),
    })
    write_json(raw / "student_samples.json", {"imported_trace": trace, "fresh_behavioral": behavior})
    write_json(raw / "teacher_samples.json", {
        "source": r8.r2.TEACHER_PATH.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(r8.r2.TEACHER_PATH), "used_for_evaluation": False,
    })
    write_json(raw / "wsl_environment.json", {
        "versions": behavior[0]["versions"], "gpu": behavior[0]["gpu"],
        "worker_count": 2,
        "peak_allocated_vram_gib_max": max(item["peak_allocated_vram_gib"] for item in behavior),
    })
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for sample in trace["math_samples"]:
            stream.write(json.dumps({"family": "full_trace", "seed": SELECTED_TRACE_SEED, **sample}, ensure_ascii=False) + "\n")
        for item in behavior:
            for sample in item["math_samples"]:
                stream.write(json.dumps({"family": "behavioral_finalist", "seed": item["seed"], **sample}, ensure_ascii=False) + "\n")

    definitions = {
        "source_integrity": ("all_source_and_checkpoint_hashes_verified", "eq", True),
        "panel_isolation": ("third_panel_disjoint_from_training_and_prior_panels", "eq", True),
        "trace_import_coverage": ("imported_selected_trace_generations", "eq", 256),
        "behavioral_checkpoint_coverage": ("behavioral_checkpoints_evaluated", "eq", 2),
        "fresh_evaluation_coverage": ("fresh_behavioral_generations", "eq", 512),
        "practical_superiority": ("hierarchical_bootstrap_95ci_lower_trace_minus_behavioral_math", "gt", 0.0),
        "point_superiority": ("selected_trace_minus_behavioral_mean_math", "gt", 0.0),
        "protected_retention": ("selected_trace_minus_behavioral_mean_qa", "ge", -0.05),
        "independent_score": ("independent_rescore_match", "eq", True),
        "service_recovery": ("service_and_embedding_restored", "eq", True),
    }
    gates: dict[str, Any] = {}
    for gate_id, (metric, operator, threshold) in definitions.items():
        actual = metrics[metric]
        passed = (
            actual == threshold if operator == "eq" else
            actual > threshold if operator == "gt" else
            actual >= threshold
        )
        gates[gate_id] = {"metric": metric, "operator": operator, "threshold": threshold,
                          "actual": actual, "pass": passed}

    evidence = {
        "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
        "artifact_hashes": "raw/artifact_hashes.json", "continuation_ledger": "raw/continuation_ledger.json",
        "dataset_hashes": "raw/dataset_hashes.json", "independent_evaluation": "raw/independent_evaluation.json",
        "model_hash": "raw/model_hash.json", "paired_baseline": "raw/paired_baseline.json",
        "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl",
        "receipt_fingerprint": "raw/receipt.json", "service_maintenance": "raw/service_maintenance.json",
        "source_execution_receipt": "raw/source_execution_receipt.json",
        "student_samples": "raw/student_samples.json", "teacher_samples": "raw/teacher_samples.json",
        "wsl_environment": "raw/wsl_environment.json",
    }
    evidence_files = sorted({raw / value.removeprefix("raw/") for value in evidence.values() if value != "raw/receipt.json"})
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(), started_at_utc=started, started_monotonic=mono,
        input_paths=[*EXPECTED_HASHES.keys(), panel_path, *workers.glob("*.json"), *evidence_files],
        packages=["pytest"], runtime={
            "execution_mode": "selected_trace_vs_behavioral_finalists_third_panel",
            "host_pid": os.getpid(), "fresh_generation_count": 512,
            "imported_generation_count": 256, "timing_is_evidence": False,
        },
    )
    complete, errors = provenance_complete(provenance)
    if not complete or not independent:
        raise ValueError(f"evidence validation failed: {errors}, scorer={independent}")
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID,
               "provenance": provenance, "provenance_complete": True,
               "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    failed = [gate_id for gate_id, gate in gates.items() if not gate["pass"]]
    claim = (
        "SELECTED_TRACE_PRACTICALLY_SUPERIOR_TO_BEHAVIORAL_FINALISTS_R2"
        if not failed else "SELECTED_TRACE_NOT_PRACTICALLY_SUPERIOR_TO_BEHAVIORAL_FINALISTS_R2"
    )
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Selected trace seed `{SELECTED_TRACE_SEED}` scored `{trace_correct}/256` "
        f"(`{trace_accuracy:.6f}`). Behavioral seeds `{BEHAVIOR_SEEDS[0]}` and "
        f"`{BEHAVIOR_SEEDS[1]}` scored `{behavior_correct[0]}/256` and "
        f"`{behavior_correct[1]}/256`; mean `{behavior_mean:.6f}`. Trace-minus-mean "
        f"delta `{math_delta:.6f}`, hierarchical-bootstrap 95% interval "
        f"`[{bootstrap['lower_95']:.6f}, {bootstrap['upper_95']:.6f}]`, and frozen QA "
        f"delta `{qa_delta:.6f}`. Failed gates: "
        f"`{', '.join(failed) if failed else 'none'}`.\n",
        encoding="utf-8", newline="\n",
    )
    write_json(finalized / "complete.json", {
        "task_id": TASK_ID, "receipt_fingerprint": receipt["receipt_fingerprint"],
        "failed_gates": failed,
    })
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        verify_sources()
        third_ids, trace, qa = load_frozen_inputs()
        assert len(third_ids) == len(trace["math_samples"]) == 256
        assert qa["trace_correct"] == 11
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
