#!/usr/bin/env python3
"""Final disjoint MBPP+ panel and frozen two-panel coding-alias synthesis."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import shutil
import statistics
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (
    build_provenance, canonical_json_sha256, provenance_complete, sha256_file,
)
from tools.research import run_fleet_mbppplus as r1

TASK_ID = "BACKLOG-FLEET-MBPPPLUS-02"
SOURCE = ROOT / "runs/research/BACKLOG-FLEET-MBPPPLUS-01"
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-FLEET-MBPPPLUS-02.json"
PREREGISTRATION = ROOT / "runs/research/BACKLOG-FLEET-MBPPPLUS-02/PRE_REGISTRATION.md"
SECOND_PANEL_HASH = "27374bbc1040b08ee5c4b4ecd518ed03e00d826a3a5cf9a5b4c86e45ea380ef0"
BOOTSTRAP_SEED = 2026082713
BOOTSTRAP_REPLICATES = 20_000
EXPECTED_ADDITIONAL = {
    ADMISSION: "84c0fbb21eba0cc380ece1be59bc921ea333900dbba8793d4febb81870f60b63",
    PREREGISTRATION: "3a507eccc7e69926d1a8afa222680bbff13e94722ab54921edcc58f8bbbcedd5",
    SOURCE / "raw/samples.jsonl": "357a3b29558867559d87c2083bcb08ac4bf0ba26e0f5f054efc9513840e5e46e",
    SOURCE / "raw/official_scores.json": "5c44c3741a380b0869658630ca886d5e364351965a59c1b894f5cb8ca2951280",
    SOURCE / "raw/receipt.json": "cbbd3ea566581b152c71d992f8cf29bf2131eabb8d7ccd09de551ddc51205b5e",
    SOURCE / "PRE_REGISTRATION.md": "3be7bdf941f7e9f903b7b49a78f60b5c09db1306f3dd6f140f210ea443e2cab8",
    ROOT / "tools/research/run_fleet_mbppplus.py": "bab01e2ca100cba1f280947322b0c22b137cb35a2181b490a0c1e44387ef3a39",
}


def write_json(path: pathlib.Path, value: Any) -> None:
    r1.write_json(path, value)


def second_subset() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in r1.DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["task_id"].split("/")[-1]))
    order = list(range(len(rows)))
    random.Random(r1.SUBSET_SEED).shuffle(order)
    first = [rows[index] for index in sorted(order[:100])]
    second = [rows[index] for index in sorted(order[100:200])]
    first_ids = [row["task_id"] for row in first]
    second_ids = [row["task_id"] for row in second]
    if (canonical_json_sha256(first_ids) != r1.SUBSET_HASH
            or canonical_json_sha256(second_ids) != SECOND_PANEL_HASH
            or set(first_ids).intersection(second_ids)):
        raise ValueError("two-panel selection differs from preregistration")
    return second


def verify_sources() -> tuple[dict[str, Any], list[pathlib.Path]]:
    base, paths = r1.verify_inputs()
    additional: dict[str, Any] = {}
    for path, expected in EXPECTED_ADDITIONAL.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"R2 source mismatch: {path}: {actual} != {expected}")
        additional[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": actual}
    return {"r1_frozen": base, "r2_additional": additional}, [*paths, *EXPECTED_ADDITIONAL]


def failure_set(score: dict[str, Any]) -> set[str]:
    return {row["task_id"] for row in score["failures"]}


def stratified_bootstrap(panel_scores: list[tuple[list[str], dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    differences: list[list[int]] = []
    for ids, hauhaucs, qwen38 in panel_scores:
        failed_h = failure_set(hauhaucs)
        failed_q = failure_set(qwen38)
        differences.append([int(task not in failed_h) - int(task not in failed_q) for task in ids])
    if len(differences) != 2 or any(len(row) != 100 for row in differences):
        raise ValueError("bootstrap panel dimensions differ from preregistration")
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = []
    for _ in range(BOOTSTRAP_REPLICATES):
        panel_means = [sum(row[rng.randrange(100)] for _ in range(100)) / 100 for row in differences]
        estimates.append(statistics.mean(panel_means))
    estimates.sort()
    points = [statistics.mean(row) for row in differences]
    return {"panel_deltas": [round(value, 8) for value in points],
            "point": round(statistics.mean(points), 8), "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "lower_95": round(estimates[int(0.025 * BOOTSTRAP_REPLICATES)], 8),
            "upper_95": round(estimates[min(BOOTSTRAP_REPLICATES - 1, int(0.975 * BOOTSTRAP_REPLICATES))], 8)}


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    finalized = raw / "finalized"
    finalized.mkdir(parents=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()
    frozen, input_paths = verify_sources()
    problems = second_subset()
    first_ids = [row["task_id"] for row in r1.load_subset()]
    second_ids = [row["task_id"] for row in problems]
    write_json(raw / "artifact_hashes.json", frozen)
    write_json(raw / "dataset_hashes.json", {
        "dataset_sha256": sha256_file(r1.DATASET), "first_panel_ids": first_ids,
        "first_panel_sha256": canonical_json_sha256(first_ids), "second_panel_ids": second_ids,
        "second_panel_sha256": canonical_json_sha256(second_ids),
        "panels_disjoint": set(first_ids).isdisjoint(second_ids)})
    shutil.copy2(SOURCE / "raw/samples.jsonl", raw / "samples.jsonl")

    initial_service = r1.fleet.service_state()
    initial_gateway = r1.fleet.gateway_status()
    initial_model = initial_gateway.get("current_model")
    if initial_service.get("active_state") != "active" or initial_model not in r1.MODELS:
        raise RuntimeError("initial route is not safely restorable")
    if r1.fleet.embedding_health() != 200:
        raise RuntimeError("embedding unhealthy before experiment")
    route_snapshots: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    second_scores: dict[str, Any] = {}
    error: Exception | None = None
    restored_status: dict[str, Any] = {}
    try:
        for model in r1.MODELS:
            status, gpu = r1.fleet.switch_model(model)
            snapshot = {"model": model, "status": status, "gpu": gpu,
                        "embedding_status": r1.fleet.embedding_health()}
            if snapshot["embedding_status"] != 200:
                raise RuntimeError(f"embedding unhealthy at {model}")
            route_snapshots.append(snapshot)
            model_records: list[dict[str, Any]] = []
            consecutive_errors = 0
            model_samples = raw / f"{model}.samples.jsonl"
            for index, problem in enumerate(problems, 1):
                row = r1.response_row(model, problem)
                r1.append_jsonl(raw / "samples.jsonl", {"source_panel": "r2", **row})
                r1.append_jsonl(model_samples, {"task_id": row["task_id"], "solution": row["solution"]})
                records.append(row)
                model_records.append(row)
                consecutive_errors = consecutive_errors + 1 if row["http_status"] != 200 or row["error"] else 0
                if consecutive_errors >= 4:
                    raise RuntimeError(f"four consecutive failures on {model}")
                if index % 20 == 0:
                    print(f"[GEN] {model} {index}/100", flush=True)
            if [row["task_id"] for row in model_records] != second_ids:
                raise ValueError(f"second-panel coverage/order mismatch for {model}")
            second_scores[model] = r1.score_model(model, model_samples, raw)
            write_json(finalized / f"{model}.json", {"model": model, "generated": 100,
                       "plus_pass": second_scores[model]["plus_pass"],
                       "plus_pass_at_1": second_scores[model]["plus_pass_at_1"]})
            print(f"[SCORE] {model} plus={second_scores[model]['plus_pass']}/100", flush=True)
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

    first_scores = json.loads((SOURCE / "raw/official_scores.json").read_text(encoding="utf-8"))
    comparison = stratified_bootstrap([
        (first_ids, first_scores["hauhaucs"], first_scores["qwen38"]),
        (second_ids, second_scores["hauhaucs"], second_scores["qwen38"]),
    ])
    combined: dict[str, Any] = {}
    for model in r1.MODELS:
        base_pass = first_scores[model]["base_pass"] + second_scores[model]["base_pass"]
        plus_pass = first_scores[model]["plus_pass"] + second_scores[model]["plus_pass"]
        combined[model] = {"n": 200, "base_pass": base_pass, "base_pass_at_1": base_pass / 200,
                           "plus_pass": plus_pass, "plus_pass_at_1": plus_pass / 200,
                           "r1_plus_pass": first_scores[model]["plus_pass"],
                           "r2_plus_pass": second_scores[model]["plus_pass"]}
    metrics = {"r1_sources_and_model_artifacts_verified": True,
               "two_mbpp_panels_disjoint": set(first_ids).isdisjoint(second_ids),
               "verified_text_routes_completed": len(route_snapshots),
               "fresh_second_panel_generations": len(records),
               "successful_nonempty_second_panel_responses": sum(row["http_status"] == 200 and row["answered"] for row in records),
               "official_model_panel_scores": len(first_scores) + len(second_scores),
               "combined_hauhaucs_mbpp_plus_pass_at_1": combined["hauhaucs"]["plus_pass_at_1"],
               "stratified_hauhaucs_minus_qwen38": comparison,
               "combined": combined,
               "initial_route_and_services_restored": recovery["initial_route_and_services_restored"]}
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "official_scores.json", {"r1": first_scores, "r2": second_scores, "combined": combined})
    write_json(raw / "effective_route.json", {"models": r1.MODELS, "snapshots": route_snapshots})
    write_json(raw / "service_identity.json", {"initial": initial_service, "final": recovery["final_service"]})
    write_json(raw / "paired_baseline.json", {"r1": {"hauhaucs": first_scores["hauhaucs"], "qwen38": first_scores["qwen38"]},
                                               "r2": {"hauhaucs": second_scores["hauhaucs"], "qwen38": second_scores["qwen38"]},
                                               "combined": comparison})
    write_json(raw / "hardware_metrics.json", {"route_snapshots": route_snapshots, "final_gpu": r1.fleet.gpu_state()})
    write_json(raw / "independent_evaluation.json", {"official_executor": "EvalPlus 0.3.1",
               "r1_scores_sha256": sha256_file(SOURCE / "raw/official_scores.json"),
               "r2_scores": second_scores, "combined": combined, "stratified_bootstrap": comparison,
               "scorer_sha256": sha256_file(r1.SCORER)})
    write_json(raw / "source_execution_receipt.json", {"source_task_id": "BACKLOG-FLEET-MBPPPLUS-01",
               "receipt_sha256": sha256_file(SOURCE / "raw/receipt.json"),
               "receipt_fingerprint": json.loads((SOURCE / "raw/receipt.json").read_text(encoding="utf-8"))["receipt_fingerprint"]})

    lower = comparison["lower_95"]
    gates = {
        "source_integrity": {"metric": "r1_sources_and_model_artifacts_verified", "operator": "eq", "threshold": True, "actual": True, "pass": True},
        "panel_isolation": {"metric": "two_mbpp_panels_disjoint", "operator": "eq", "threshold": True, "actual": True, "pass": True},
        "route_coverage": {"metric": "verified_text_routes_completed", "operator": "eq", "threshold": 4, "actual": len(route_snapshots), "pass": len(route_snapshots) == 4},
        "fresh_generation_coverage": {"metric": "fresh_second_panel_generations", "operator": "eq", "threshold": 400, "actual": len(records), "pass": len(records) == 400},
        "combined_score_coverage": {"metric": "official_model_panel_scores", "operator": "eq", "threshold": 8, "actual": len(first_scores) + len(second_scores), "pass": len(first_scores) + len(second_scores) == 8},
        "coding_alias_absolute": {"metric": "combined_hauhaucs_mbpp_plus_pass_at_1", "operator": "ge", "threshold": 0.70, "actual": combined["hauhaucs"]["plus_pass_at_1"], "pass": combined["hauhaucs"]["plus_pass_at_1"] >= 0.70},
        "coding_alias_noninferiority": {"metric": "stratified_bootstrap_95ci_lower_hauhaucs_minus_qwen38", "operator": "gt", "threshold": -0.05, "actual": lower, "pass": lower > -0.05},
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
        input_paths=[*input_paths, *evidence_files, *raw.glob("*.samples.jsonl"), *raw.glob("*_score.json")],
        packages=["pytest"], runtime={"execution_mode": "final_disjoint_mbppplus_panel_synthesis",
        "host_pid": os.getpid(), "fresh_generation_count": 400, "imported_generation_count": 400,
        "models": r1.MODELS, "timing_is_evidence": False})
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise ValueError(f"incomplete provenance: {errors}")
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID,
               "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = "HAUHAUCS_MBPPPLUS_200_CODING_ALIAS_RETAINED_R2" if not failed else "HAUHAUCS_MBPPPLUS_200_CODING_ALIAS_NOT_RETAINED_R2"
    summary = ", ".join(f"{model}={combined[model]['plus_pass']}/200" for model in r1.MODELS)
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Combined official MBPP+ scores: {summary}. HauhauCS minus Qwen3.8 stratified-bootstrap "
        f"95% interval `[{comparison['lower_95']:.4f}, {comparison['upper_95']:.4f}]`; "
        f"panel deltas `{comparison['panel_deltas']}`. Failed gates: "
        f"`{', '.join(failed) if failed else 'none'}`. No further MBPP panel is permitted.\n",
        encoding="utf-8")
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
