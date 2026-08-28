#!/usr/bin/env python3
"""Continue full HumanEval+ after R1 scorer-import abort without regenerating Qwen3.8."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (
    build_provenance, canonical_json_sha256, provenance_complete, sha256_file,
)
from tools.research import run_fleet_humanevalplus as r1
from tools.research import run_trace_distillation_training_r2 as paths

TASK_ID = "BACKLOG-FLEET-HUMANEVALPLUS-02"
SOURCE = ROOT / "runs/research/BACKLOG-FLEET-HUMANEVALPLUS-01"
WATCH_FINAL = ROOT / "runs/autonomous/EXPERIMENT-WATCH-2026-08-27-FLEET-HUMANEVAL-R1/FINAL.json"
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-FLEET-HUMANEVALPLUS-02.json"
PREREGISTRATION = ROOT / "runs/research/BACKLOG-FLEET-HUMANEVALPLUS-02/PRE_REGISTRATION.md"
EXPECTED_HASHES = {
    ADMISSION: "dc6440ecc190b69e35af792b15cf87c37699a2b564f26b12d76e827158ca1c61",
    PREREGISTRATION: "54d727d209264b11853f07ed59137414f2db10673cf5ec6a154473171d05f862",
    SOURCE / "raw/samples.jsonl": "a243f7e842ad837521084ba2e881c7708491028c0d3bac778c0536cbfc402b1f",
    SOURCE / "raw/qwen38.samples.jsonl": "a325dcfa96bd92e018c8f7fbd76a836e2d3e0e46185ccc026fb5cfc9896b55d3",
    SOURCE / "raw/recovery_state.json": "7f9104d994b58c54a653fa17aabd2ae2a93b0543de89d660b02d4c05f8c98be6",
    SOURCE / "PRE_REGISTRATION.md": "bb93658ec154eaaa136074ec418ddfbc6ce657957670e45c4c497b0d90b1b0fd",
    SOURCE / "runner.stderr.log": "648b63739f42c7f42f6858c48fa8003d31071c5b5582d386d207b29ee925e0f9",
    WATCH_FINAL: "7d6437de24949e8fd40c358ad6837f284cd7d74f2aed35587926efdb91c6471a",
    ROOT / "tools/research/run_fleet_humanevalplus.py": "e495a2094097d1e616709f6064850e8b3d2fd0778b81c9f4ef05944934cb6cfe",
    r1.FLEET_REGISTRY: "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    r1.DATASET: "08d3df5c27a5f9a40176c27592b2d81e931b55d8d9edb7b1ffc28f2ccbdba735",
    r1.SCORER: "cdbefde3f0e12b0dbf8697aff154ea88ec07d7fafb2a959a3c88882f63f1aa0d",
    r1.SHARED_QA: "60af3eac1e119047e3b0d767c52ee8295ac44abbfbaa44b1c42eee45945336c6",
}


def write_json(path: pathlib.Path, value: Any) -> None:
    r1.write_json(path, value)


def verify_sources() -> tuple[dict[str, Any], list[pathlib.Path]]:
    host: dict[str, Any] = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"continuation source mismatch: {path}: {actual} != {expected}")
        host[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": actual}
    registry = json.loads(r1.FLEET_REGISTRY.read_text(encoding="utf-8"))
    artifacts: dict[str, Any] = {}
    for model in r1.MODELS:
        expected = registry["models"][model]["artifact"]
        size = r1.fleet.wsl("stat", "-c", "%s", expected["path"], timeout=120)
        digest = r1.fleet.wsl("sha256sum", expected["path"], timeout=1800)
        actual_size = int(size["stdout"]) if size["returncode"] == 0 else -1
        actual_sha = digest["stdout"].split()[0] if digest["returncode"] == 0 else ""
        if actual_size != expected["bytes"] or actual_sha != expected["sha256"]:
            raise ValueError(f"artifact mismatch for {model}")
        artifacts[model] = {"path": expected["path"], "bytes": actual_size,
                            "sha256": actual_sha, "quant": expected["quant"]}
    return {"host": host, "model_artifacts": artifacts}, list(EXPECTED_HASHES)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_import(panel: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = read_jsonl(SOURCE / "raw/samples.jsonl")
    samples = read_jsonl(SOURCE / "raw/qwen38.samples.jsonl")
    ids = [row["task_id"] for row in panel]
    if (len(records) != 164 or len(samples) != 164
            or [row["task_id"] for row in records] != ids
            or [row["task_id"] for row in samples] != ids
            or any(row["model"] != "qwen38" for row in records)
            or sum(bool(row["answered"]) for row in records) != 164):
        raise ValueError("R1 Qwen3.8 import is incomplete or out of order")
    return records, samples


def score_command(samples: pathlib.Path, output: pathlib.Path) -> list[str]:
    repo_wsl = paths.windows_path_to_wsl(ROOT)
    return ["wsl", "-d", "Ubuntu-24.04", "--", "env", f"PYTHONPATH={repo_wsl}",
            r1.EVALPLUS_PYTHON, paths.windows_path_to_wsl(r1.SCORER),
            paths.windows_path_to_wsl(samples), paths.windows_path_to_wsl(output)]


def score_model(model: str, samples: pathlib.Path, raw: pathlib.Path) -> dict[str, bool]:
    output = raw / f"{model}.scores.json"
    command = score_command(samples, output)
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=3600, check=False)
    (raw / f"{model}.score.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (raw / f"{model}.score.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"EvalPlus failed for {model}: {completed.stderr[-4000:]}")
    scores = json.loads(output.read_text(encoding="utf-8"))
    ids = [row["task_id"] for row in r1.load_panel()]
    if list(scores) != ids or len(scores) != 164 or not all(isinstance(value, bool) for value in scores.values()):
        raise ValueError(f"official score mismatch for {model}")
    return scores


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    finalized = raw / "finalized"
    finalized.mkdir(parents=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()
    frozen, input_paths = verify_sources()
    panel = r1.load_panel()
    imported_records, imported_samples = verify_import(panel)
    shutil.copy2(SOURCE / "raw/samples.jsonl", raw / "samples.jsonl")
    shutil.copy2(SOURCE / "raw/qwen38.samples.jsonl", raw / "qwen38.samples.jsonl")
    write_json(raw / "artifact_hashes.json", frozen)
    write_json(raw / "dataset_hashes.json", {"dataset_sha256": sha256_file(r1.DATASET),
               "task_ids": [row["task_id"] for row in panel],
               "task_ids_sha256": canonical_json_sha256([row["task_id"] for row in panel]),
               "imported_qwen_records_sha256": sha256_file(SOURCE / "raw/samples.jsonl"),
               "imported_qwen_samples_sha256": sha256_file(SOURCE / "raw/qwen38.samples.jsonl")})

    initial_service = r1.fleet.service_state()
    initial_gateway = r1.fleet.gateway_status()
    initial_model = initial_gateway.get("current_model")
    if initial_service.get("active_state") != "active" or initial_model not in r1.MODELS:
        raise RuntimeError("initial route is not safely restorable")
    if r1.fleet.embedding_health() != 200:
        raise RuntimeError("embedding unhealthy before continuation")
    source_recovery = json.loads((SOURCE / "raw/recovery_state.json").read_text(encoding="utf-8"))
    route_snapshots: list[dict[str, Any]] = [{"model": "qwen38", "imported": True,
        "source_recovery_sha256": sha256_file(SOURCE / "raw/recovery_state.json"),
        "status": source_recovery["initial_gateway"]}]
    records = list(imported_records)
    scores: dict[str, dict[str, bool]] = {}
    error: Exception | None = None
    restored_status: dict[str, Any] = {}
    try:
        scores["qwen38"] = score_model("qwen38", raw / "qwen38.samples.jsonl", raw)
        write_json(finalized / "qwen38.json", {"model": "qwen38", "imported": 164,
                   "plus_pass": sum(scores["qwen38"].values()),
                   "plus_pass_at_1": sum(scores["qwen38"].values()) / 164})
        print(f"[SCORE] qwen38 imported plus={sum(scores['qwen38'].values())}/164", flush=True)
        for model in ("hauhaucs", "fable-tc", "qwen36-moe"):
            status, gpu = r1.fleet.switch_model(model)
            snapshot = {"model": model, "imported": False, "status": status, "gpu": gpu,
                        "embedding_status": r1.fleet.embedding_health()}
            if snapshot["embedding_status"] != 200:
                raise RuntimeError(f"embedding unhealthy at {model}")
            route_snapshots.append(snapshot)
            model_records: list[dict[str, Any]] = []
            consecutive_errors = 0
            sample_path = raw / f"{model}.samples.jsonl"
            for index, problem in enumerate(panel, 1):
                row = r1.generate(model, problem)
                r1.append_jsonl(raw / "samples.jsonl", row)
                r1.append_jsonl(sample_path, {"task_id": row["task_id"], "solution": row["solution"]})
                records.append(row)
                model_records.append(row)
                consecutive_errors = consecutive_errors + 1 if row["http_status"] != 200 or row["error"] else 0
                if consecutive_errors >= 4:
                    raise RuntimeError(f"four consecutive failures on {model}")
                if index % 20 == 0 or index == 164:
                    print(f"[GEN] {model} {index}/164", flush=True)
            if [row["task_id"] for row in model_records] != [row["task_id"] for row in panel]:
                raise ValueError(f"fresh coverage/order mismatch for {model}")
            scores[model] = score_model(model, sample_path, raw)
            passed = sum(scores[model].values())
            write_json(finalized / f"{model}.json", {"model": model, "fresh": 164,
                       "plus_pass": passed, "plus_pass_at_1": passed / 164})
            print(f"[SCORE] {model} plus={passed}/164", flush=True)
    except Exception as caught:
        error = caught
    finally:
        try:
            restored_status, _ = r1.fleet.switch_model(str(initial_model))
        except Exception as restore_error:
            if error is None:
                error = restore_error
        final_service = r1.fleet.service_state()
        final_embedding = r1.fleet.embedding_health()
        recovery = {"initial_service": initial_service, "initial_gateway": initial_gateway,
                    "initial_model": initial_model, "restored_gateway": restored_status,
                    "final_service": final_service, "final_embedding_status": final_embedding,
                    "initial_route_and_services_restored": (
                        restored_status.get("current_model") == initial_model
                        and restored_status.get("backend_healthy") is True
                        and final_service.get("active_state") == "active"
                        and final_service.get("n_restarts") == initial_service.get("n_restarts")
                        and final_embedding == 200)}
        write_json(raw / "recovery_state.json", recovery)
    if error:
        raise error

    comparison = r1.paired_bootstrap(scores["hauhaucs"], scores["qwen38"])
    official = {model: {"n": 164, "plus_pass": sum(values.values()),
                        "plus_pass_at_1": sum(values.values()) / 164,
                        "per_task": values} for model, values in scores.items()}
    metrics = {"r1_partial_sources_and_model_artifacts_verified": True,
               "imported_qwen38_generations": len(imported_records),
               "fresh_remaining_generations": len(records) - len(imported_records),
               "verified_text_routes_completed": len(route_snapshots),
               "successful_nonempty_responses": sum(row["http_status"] == 200 and row["answered"] for row in records),
               "truncated_responses": sum(row["truncated"] for row in records),
               "models_scored_by_evalplus": len(scores),
               "hauhaucs_humaneval_plus_pass_at_1": official["hauhaucs"]["plus_pass_at_1"],
               "paired_hauhaucs_minus_qwen38": comparison,
               "models": {model: {"plus_pass": official[model]["plus_pass"],
                                   "plus_pass_at_1": official[model]["plus_pass_at_1"],
                                   "answered": sum(row["answered"] for row in records if row["model"] == model),
                                   "truncated": sum(row["truncated"] for row in records if row["model"] == model)} for model in r1.MODELS},
               "initial_route_and_services_restored": recovery["initial_route_and_services_restored"]}
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "official_scores.json", official)
    write_json(raw / "effective_route.json", {"models": r1.MODELS, "snapshots": route_snapshots})
    write_json(raw / "service_identity.json", {"initial": initial_service, "final": recovery["final_service"]})
    write_json(raw / "paired_baseline.json", {"hauhaucs": official["hauhaucs"],
                                               "qwen38": official["qwen38"], "comparison": comparison})
    write_json(raw / "hardware_metrics.json", {"route_snapshots": route_snapshots, "final_gpu": r1.fleet.gpu_state()})
    write_json(raw / "independent_evaluation.json", {"official_executor": "EvalPlus 0.3.1",
               "scores": official, "paired_bootstrap": comparison,
               "scorer_sha256": sha256_file(r1.SCORER), "extractor_sha256": sha256_file(r1.SHARED_QA),
               "scorer_bootstrap": {"PYTHONPATH": paths.windows_path_to_wsl(ROOT)}})
    write_json(raw / "source_execution_receipt.json", {"source_task_id": "BACKLOG-FLEET-HUMANEVALPLUS-01",
               "source_status": "failed_no_receipt", "samples_sha256": sha256_file(SOURCE / "raw/samples.jsonl"),
               "solution_samples_sha256": sha256_file(SOURCE / "raw/qwen38.samples.jsonl"),
               "stderr_sha256": sha256_file(SOURCE / "runner.stderr.log"),
               "watch_final_sha256": sha256_file(WATCH_FINAL)})

    gates = {
        "source_integrity": {"metric": "r1_partial_sources_and_model_artifacts_verified", "operator": "eq", "threshold": True, "actual": True, "pass": True},
        "imported_coverage": {"metric": "imported_qwen38_generations", "operator": "eq", "threshold": 164, "actual": len(imported_records), "pass": len(imported_records) == 164},
        "fresh_coverage": {"metric": "fresh_remaining_generations", "operator": "eq", "threshold": 492, "actual": len(records) - len(imported_records), "pass": len(records) - len(imported_records) == 492},
        "route_coverage": {"metric": "verified_text_routes_completed", "operator": "eq", "threshold": 4, "actual": len(route_snapshots), "pass": len(route_snapshots) == 4},
        "response_integrity": {"metric": "successful_nonempty_responses", "operator": "ge", "threshold": 650, "actual": metrics["successful_nonempty_responses"], "pass": metrics["successful_nonempty_responses"] >= 650},
        "official_score_coverage": {"metric": "models_scored_by_evalplus", "operator": "eq", "threshold": 4, "actual": len(scores), "pass": len(scores) == 4},
        "coding_alias_absolute": {"metric": "hauhaucs_humaneval_plus_pass_at_1", "operator": "ge", "threshold": 0.80, "actual": official["hauhaucs"]["plus_pass_at_1"], "pass": official["hauhaucs"]["plus_pass_at_1"] >= 0.80},
        "coding_alias_noninferiority": {"metric": "paired_bootstrap_95ci_lower_hauhaucs_minus_qwen38", "operator": "gt", "threshold": -0.05, "actual": comparison["lower_95"], "pass": comparison["lower_95"] > -0.05},
        "service_recovery": {"metric": "initial_route_and_services_restored", "operator": "eq", "threshold": True, "actual": recovery["initial_route_and_services_restored"], "pass": recovery["initial_route_and_services_restored"] is True},
    }
    evidence = {"acceptance_gates": "raw/receipt.json", "raw_samples": "raw/samples.jsonl",
                "provenance": "raw/receipt.json", "receipt_fingerprint": "raw/receipt.json",
                "effective_route": "raw/effective_route.json", "service_identity": "raw/service_identity.json",
                "paired_baseline": "raw/paired_baseline.json", "recovery_state": "raw/recovery_state.json",
                "hardware_metrics": "raw/hardware_metrics.json", "artifact_hashes": "raw/artifact_hashes.json",
                "dataset_hashes": "raw/dataset_hashes.json", "official_scores": "raw/official_scores.json",
                "independent_evaluation": "raw/independent_evaluation.json",
                "source_execution_receipt": "raw/source_execution_receipt.json"}
    evidence_files = sorted({raw / value.removeprefix("raw/") for value in evidence.values() if value != "raw/receipt.json"})
    provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(),
        started_at_utc=started, started_monotonic=mono,
        input_paths=[*input_paths, *evidence_files, *raw.glob("*.samples.jsonl"), *raw.glob("*.scores.json")],
        packages=["pytest"], runtime={"execution_mode": "humanevalplus_continuation_after_import_abort",
        "host_pid": os.getpid(), "imported_generation_count": 164, "fresh_generation_count": 492,
        "models": r1.MODELS, "evalplus_version": "0.3.1", "timing_is_evidence": False})
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise ValueError(f"incomplete provenance: {errors}")
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID,
               "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = "HAUHAUCS_FULL_HUMANEVALPLUS_CODING_ALIAS_RETAINED_R2" if not failed else "HAUHAUCS_FULL_HUMANEVALPLUS_CODING_ALIAS_NOT_RETAINED_R2"
    summary = ", ".join(f"{model}={official[model]['plus_pass']}/164" for model in r1.MODELS)
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Official HumanEval+ scores: {summary}. HauhauCS minus Qwen3.8 paired-bootstrap "
        f"95% interval `[{comparison['lower_95']:.4f}, {comparison['upper_95']:.4f}]`. "
        f"Failed gates: `{', '.join(failed) if failed else 'none'}`.\n", encoding="utf-8")
    write_json(finalized / "complete.json", {"task_id": TASK_ID,
               "receipt_fingerprint": receipt["receipt_fingerprint"], "failed_gates": failed})
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
