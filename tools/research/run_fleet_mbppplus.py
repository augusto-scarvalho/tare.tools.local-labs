#!/usr/bin/env python3
"""Run a frozen 100-task MBPP+ screen across the qualified text fleet."""
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

from benchmark_harness_qa import extract_code
from tools.analysis.experiment_provenance import (
    build_provenance, canonical_json_sha256, provenance_complete, sha256_file,
)
from tools.research import run_fleet_regression_screen as fleet
from tools.research import run_trace_distillation_training_r2 as paths

TASK_ID = "BACKLOG-FLEET-MBPPPLUS-01"
MODELS = ("qwen38", "hauhaucs", "fable-tc", "qwen36-moe")
SUBSET_SEED = 20260726
SUBSET_N = 100
SUBSET_HASH = "288ceec36e29dea4f39822770e8a61bd93df8f57e8fc5b8cdf9f0fc58a238e91"
BOOTSTRAP_SEED = 2026082712
BOOTSTRAP_REPLICATES = 20_000
MAX_TOKENS = 768
EVALPLUS_PYTHON = "/home/augus/evalplus-venv/bin/python"
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-FLEET-MBPPPLUS-01.json"
PREREGISTRATION = ROOT / "runs/research/BACKLOG-FLEET-MBPPPLUS-01/PRE_REGISTRATION.md"
DATASET = ROOT / "workloads/mbpp_plus.jsonl"
DATASET_IDENTITY = ROOT / "workloads/mbpp_plus.identity.json"
SCORER = ROOT / "tools/analysis/score_mbpp_subset.py"
REFERENCE_RUNNER = ROOT / "tools/benchmarks/mbpp_plus_bench.py"
SHARED_QA = ROOT / "benchmark_harness_qa.py"
FLEET_REGISTRY = ROOT / "config/qualified_model_fleet.json"
PRIOR_RECEIPT = ROOT / "runs/research/BACKLOG-FLEET-REGRESSION-SCREEN-01/raw/receipt.json"
EXPECTED_HASHES = {
    ADMISSION: "b164711c6cb1851d0cd669a4007b5e9c208d0623227bf7cd5a1ead8203546f78",
    PREREGISTRATION: "3be7bdf941f7e9f903b7b49a78f60b5c09db1306f3dd6f140f210ea443e2cab8",
    FLEET_REGISTRY: "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    DATASET: "0b3f47548d2e7676fac6b9bcbbc074b3a8df49606c6949343c716e46f1df9264",
    DATASET_IDENTITY: "d35e1146d622b27c3f7e0a8d776834404ff1dc8cd2137f7ad30fae5a5c5f62d0",
    SCORER: "c0c80841b92756cda807a1d13c28dbcfe82df014ce13c32fb257fa823e7d9f6e",
    REFERENCE_RUNNER: "0c50677f26fe92e9dfc14b4db2dacd3c8b17e5824b1a11a688bc731ced34cacf",
    SHARED_QA: "60af3eac1e119047e3b0d767c52ee8295ac44abbfbaa44b1c42eee45945336c6",
    PRIOR_RECEIPT: "d303490b152babcc8f590b0b840fb14c4da02c865430f8398bdca7023a0eeb94",
}
INSTRUCTION = (
    "Solve the following Python programming task. Reply with the complete executable "
    "solution inside one ```python code block, and nothing else. Include all required "
    "imports and the requested function. Do not include tests or example usage.\n\n{prompt}"
)


def write_json(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: pathlib.Path, value: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_subset() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["task_id"].split("/")[-1]))
    order = list(range(len(rows)))
    random.Random(SUBSET_SEED).shuffle(order)
    selected = [rows[index] for index in sorted(order[:SUBSET_N])]
    ids = [row["task_id"] for row in selected]
    if len(selected) != SUBSET_N or len(set(ids)) != SUBSET_N or canonical_json_sha256(ids) != SUBSET_HASH:
        raise ValueError("MBPP+ subset differs from preregistration")
    return selected


def verify_inputs() -> tuple[dict[str, Any], list[pathlib.Path]]:
    host: dict[str, Any] = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"input mismatch: {path}: {actual} != {expected}")
        host[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": actual}
    registry = json.loads(FLEET_REGISTRY.read_text(encoding="utf-8"))
    artifacts: dict[str, Any] = {}
    for model in MODELS:
        expected = registry["models"][model]["artifact"]
        size = fleet.wsl("stat", "-c", "%s", expected["path"], timeout=120)
        digest = fleet.wsl("sha256sum", expected["path"], timeout=1800)
        actual_size = int(size["stdout"]) if size["returncode"] == 0 else -1
        actual_sha = digest["stdout"].split()[0] if digest["returncode"] == 0 else ""
        if actual_size != expected["bytes"] or actual_sha != expected["sha256"]:
            raise ValueError(f"artifact identity mismatch for {model}")
        artifacts[model] = {"path": expected["path"], "bytes": actual_size,
                            "sha256": actual_sha, "quant": expected["quant"]}
    return {"host": host, "model_artifacts": artifacts}, list(EXPECTED_HASHES)


def payload(model: str, problem: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": INSTRUCTION.format(prompt=problem["prompt"])}],
        "temperature": 0.0,
        "top_k": 1,
        "seed": 20260827,
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def response_row(model: str, problem: dict[str, Any]) -> dict[str, Any]:
    request = payload(model, problem)
    started = time.perf_counter()
    status, response = fleet.http_json(f"{fleet.BASE_URL}/v1/chat/completions", request, timeout=900)
    elapsed = round(time.perf_counter() - started, 4)
    message: dict[str, Any] = {}
    try:
        value = response["choices"][0]["message"]
        if isinstance(value, dict):
            message = value
    except (KeyError, IndexError, TypeError):
        pass
    text = str(message.get("content") or "")
    finish_reason = None
    try:
        finish_reason = response["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        pass
    timings = response.get("timings") or {}
    return {
        "model": model,
        "task_id": problem["task_id"],
        "entry_point": problem["entry_point"],
        "http_status": status,
        "error": response.get("_error"),
        "answered": bool(text.strip()),
        "fenced": "```" in text,
        "finish_reason": finish_reason,
        "predicted_n": timings.get("predicted_n"),
        "decode_tps": timings.get("predicted_per_second"),
        "wall_s": elapsed,
        "truncated": finish_reason == "length" or (timings.get("predicted_n") or 0) >= MAX_TOKENS,
        "completion": text,
        "solution": extract_code(text),
        "request_sha256": canonical_json_sha256(request),
        "response": response,
    }


def score_model(model: str, samples: pathlib.Path, raw: pathlib.Path) -> dict[str, Any]:
    command = [
        "wsl", "-d", "Ubuntu-24.04", "--", EVALPLUS_PYTHON,
        paths.windows_path_to_wsl(SCORER), paths.windows_path_to_wsl(samples),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=3600, check=False)
    (raw / f"{model}.score.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (raw / f"{model}.score.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"EvalPlus failed for {model}: {completed.stderr[-4000:]}")
    score_path = samples.with_name(samples.stem + "_score.json")
    score = json.loads(score_path.read_text(encoding="utf-8"))
    if score.get("n") != SUBSET_N:
        raise ValueError(f"EvalPlus coverage mismatch for {model}")
    return score


def paired_bootstrap(hauhaucs: dict[str, Any], qwen38: dict[str, Any]) -> dict[str, Any]:
    ids = [row["task_id"] for row in load_subset()]
    failed_h = {row["task_id"] for row in hauhaucs["failures"]}
    failed_q = {row["task_id"] for row in qwen38["failures"]}
    differences = [int(task not in failed_h) - int(task not in failed_q) for task in ids]
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = [sum(differences[rng.randrange(SUBSET_N)] for _ in range(SUBSET_N)) / SUBSET_N
                 for _ in range(BOOTSTRAP_REPLICATES)]
    estimates.sort()
    return {
        "point": round(statistics.mean(differences), 8),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "lower_95": round(estimates[int(0.025 * BOOTSTRAP_REPLICATES)], 8),
        "upper_95": round(estimates[min(BOOTSTRAP_REPLICATES - 1, int(0.975 * BOOTSTRAP_REPLICATES))], 8),
        "hauhaucs_only_pass": sum(value == 1 for value in differences),
        "qwen38_only_pass": sum(value == -1 for value in differences),
    }


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    finalized = raw / "finalized"
    finalized.mkdir(parents=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()
    frozen, input_paths = verify_inputs()
    problems = load_subset()
    write_json(raw / "artifact_hashes.json", frozen)
    write_json(raw / "dataset_hashes.json", {
        "dataset_sha256": sha256_file(DATASET),
        "identity_sha256": sha256_file(DATASET_IDENTITY),
        "subset_seed": SUBSET_SEED,
        "subset_ids": [row["task_id"] for row in problems],
        "subset_ids_sha256": canonical_json_sha256([row["task_id"] for row in problems]),
    })
    initial_service = fleet.service_state()
    initial_gateway = fleet.gateway_status()
    initial_model = initial_gateway.get("current_model")
    if initial_service.get("active_state") != "active" or initial_model not in MODELS:
        raise RuntimeError("initial gateway/service is not a restorable qualified text route")
    if fleet.embedding_health() != 200:
        raise RuntimeError("embedding unhealthy before experiment")

    route_snapshots: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    scores: dict[str, Any] = {}
    error: Exception | None = None
    restored_status: dict[str, Any] = {}
    try:
        for model in MODELS:
            status, gpu = fleet.switch_model(model)
            snapshot = {"model": model, "status": status, "gpu": gpu,
                        "embedding_status": fleet.embedding_health()}
            if snapshot["embedding_status"] != 200:
                raise RuntimeError(f"embedding unhealthy at route {model}")
            route_snapshots.append(snapshot)
            model_records: list[dict[str, Any]] = []
            consecutive_errors = 0
            model_samples = raw / f"{model}.samples.jsonl"
            for index, problem in enumerate(problems, 1):
                row = response_row(model, problem)
                append_jsonl(raw / "samples.jsonl", row)
                append_jsonl(model_samples, {"task_id": row["task_id"], "solution": row["solution"]})
                records.append(row)
                model_records.append(row)
                consecutive_errors = consecutive_errors + 1 if row["http_status"] != 200 or row["error"] else 0
                if consecutive_errors >= 4:
                    raise RuntimeError(f"four consecutive request failures on {model}")
                if index % 10 == 0:
                    print(f"[GEN] {model} {index}/100", flush=True)
            if len(model_records) != SUBSET_N or [row["task_id"] for row in model_records] != [row["task_id"] for row in problems]:
                raise ValueError(f"generation coverage/order mismatch for {model}")
            scores[model] = score_model(model, model_samples, raw)
            write_json(finalized / f"{model}.json", {
                "model": model,
                "generated": len(model_records),
                "answered": sum(row["answered"] for row in model_records),
                "plus_pass": scores[model]["plus_pass"],
                "plus_pass_at_1": scores[model]["plus_pass_at_1"],
            })
            print(f"[SCORE] {model} plus={scores[model]['plus_pass']}/100", flush=True)
    except Exception as caught:
        error = caught
    finally:
        try:
            restored_status, _ = fleet.switch_model(str(initial_model))
        except Exception as restore_error:
            if error is None:
                error = restore_error
        final_service = fleet.service_state()
        final_embedding = fleet.embedding_health()
        recovery = {
            "initial_service": initial_service,
            "initial_gateway": initial_gateway,
            "initial_model": initial_model,
            "restored_gateway": restored_status,
            "final_service": final_service,
            "final_embedding_status": final_embedding,
            "initial_route_and_services_restored": (
                restored_status.get("current_model") == initial_model
                and restored_status.get("backend_healthy") is True
                and final_service.get("active_state") == "active"
                and final_service.get("n_restarts") == initial_service.get("n_restarts")
                and final_embedding == 200
            ),
        }
        write_json(raw / "recovery_state.json", recovery)
    if error:
        raise error

    comparison = paired_bootstrap(scores["hauhaucs"], scores["qwen38"])
    metrics = {
        "qualified_model_artifacts_verified": len(frozen["model_artifacts"]),
        "verified_text_routes_completed": len(route_snapshots),
        "fresh_mbpp_generations": len(records),
        "successful_nonempty_responses": sum(row["http_status"] == 200 and row["answered"] for row in records),
        "truncated_responses": sum(row["truncated"] for row in records),
        "models_scored_by_evalplus": len(scores),
        "hauhaucs_mbpp_plus_pass_at_1": scores["hauhaucs"]["plus_pass_at_1"],
        "paired_hauhaucs_minus_qwen38": comparison,
        "initial_route_and_services_restored": recovery["initial_route_and_services_restored"],
        "models": {model: {"base_pass_at_1": scores[model]["base_pass_at_1"],
                            "plus_pass_at_1": scores[model]["plus_pass_at_1"],
                            "plus_pass": scores[model]["plus_pass"],
                            "answered": sum(row["answered"] for row in records if row["model"] == model),
                            "truncated": sum(row["truncated"] for row in records if row["model"] == model)}
                   for model in MODELS},
    }
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "official_scores.json", scores)
    write_json(raw / "effective_route.json", {"models": MODELS, "snapshots": route_snapshots})
    write_json(raw / "service_identity.json", {"initial": initial_service, "final": recovery["final_service"]})
    write_json(raw / "paired_baseline.json", {"hauhaucs": scores["hauhaucs"],
                                               "qwen38": scores["qwen38"], "comparison": comparison})
    write_json(raw / "hardware_metrics.json", {"route_snapshots": route_snapshots,
                                                 "final_gpu": fleet.gpu_state()})
    write_json(raw / "independent_evaluation.json", {"official_executor": "EvalPlus 0.3.1",
               "scores": scores, "paired_bootstrap": comparison,
               "scorer_sha256": sha256_file(SCORER), "extractor_sha256": sha256_file(SHARED_QA)})
    write_json(raw / "source_execution_receipt.json", {"source_task_id": "BACKLOG-FLEET-REGRESSION-SCREEN-01",
               "receipt_sha256": sha256_file(PRIOR_RECEIPT),
               "receipt_fingerprint": json.loads(PRIOR_RECEIPT.read_text(encoding="utf-8"))["receipt_fingerprint"]})

    gates = {
        "artifact_identity": {"metric": "qualified_model_artifacts_verified", "operator": "eq", "threshold": 4, "actual": metrics["qualified_model_artifacts_verified"], "pass": metrics["qualified_model_artifacts_verified"] == 4},
        "route_coverage": {"metric": "verified_text_routes_completed", "operator": "eq", "threshold": 4, "actual": len(route_snapshots), "pass": len(route_snapshots) == 4},
        "generation_coverage": {"metric": "fresh_mbpp_generations", "operator": "eq", "threshold": 400, "actual": len(records), "pass": len(records) == 400},
        "response_integrity": {"metric": "successful_nonempty_responses", "operator": "ge", "threshold": 396, "actual": metrics["successful_nonempty_responses"], "pass": metrics["successful_nonempty_responses"] >= 396},
        "official_score_coverage": {"metric": "models_scored_by_evalplus", "operator": "eq", "threshold": 4, "actual": len(scores), "pass": len(scores) == 4},
        "coding_alias_absolute": {"metric": "hauhaucs_mbpp_plus_pass_at_1", "operator": "ge", "threshold": 0.70, "actual": scores["hauhaucs"]["plus_pass_at_1"], "pass": scores["hauhaucs"]["plus_pass_at_1"] >= 0.70},
        "coding_alias_noninferiority": {"metric": "paired_bootstrap_95ci_lower_hauhaucs_minus_qwen38", "operator": "gt", "threshold": -0.05, "actual": comparison["lower_95"], "pass": comparison["lower_95"] > -0.05},
        "service_recovery": {"metric": "initial_route_and_services_restored", "operator": "eq", "threshold": True, "actual": recovery["initial_route_and_services_restored"], "pass": recovery["initial_route_and_services_restored"] is True},
    }
    evidence = {
        "acceptance_gates": "raw/receipt.json", "raw_samples": "raw/samples.jsonl",
        "provenance": "raw/receipt.json", "receipt_fingerprint": "raw/receipt.json",
        "effective_route": "raw/effective_route.json", "service_identity": "raw/service_identity.json",
        "paired_baseline": "raw/paired_baseline.json", "recovery_state": "raw/recovery_state.json",
        "hardware_metrics": "raw/hardware_metrics.json", "artifact_hashes": "raw/artifact_hashes.json",
        "dataset_hashes": "raw/dataset_hashes.json", "official_scores": "raw/official_scores.json",
        "independent_evaluation": "raw/independent_evaluation.json",
        "source_execution_receipt": "raw/source_execution_receipt.json",
    }
    evidence_files = sorted({raw / value.removeprefix("raw/") for value in evidence.values()
                             if value != "raw/receipt.json"})
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(), started_at_utc=started,
        started_monotonic=mono, input_paths=[*input_paths, *evidence_files,
                                             *raw.glob("*.samples.jsonl"), *raw.glob("*_score.json")],
        packages=["pytest"], runtime={"execution_mode": "qualified_fleet_mbppplus",
        "host_pid": os.getpid(), "models": MODELS, "tasks_per_model": SUBSET_N,
        "evalplus_version": "0.3.1", "timing_is_evidence": False})
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise ValueError(f"incomplete provenance: {errors}")
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID,
               "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = "HAUHAUCS_MBPPPLUS_CODING_ALIAS_RETAINED_R1" if not failed else "HAUHAUCS_MBPPPLUS_CODING_ALIAS_NOT_RETAINED_R1"
    summary = ", ".join(f"{model}={scores[model]['plus_pass']}/100" for model in MODELS)
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Official MBPP+ scores: {summary}. HauhauCS minus Qwen3.8 paired bootstrap 95% interval "
        f"`[{comparison['lower_95']:.4f}, {comparison['upper_95']:.4f}]`. Failed gates: "
        f"`{', '.join(failed) if failed else 'none'}`.\n", encoding="utf-8")
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
