#!/usr/bin/env python3
"""Correct only objectively truncated R2 HumanEval+ rows, then rescore the fleet."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (
    build_provenance, canonical_json_sha256, provenance_complete, sha256_file,
)
from tools.research import run_fleet_humanevalplus as r1
from tools.research import run_fleet_humanevalplus_r2 as r2

TASK_ID = "BACKLOG-FLEET-HUMANEVALPLUS-03"
SOURCE = ROOT / "runs/research/BACKLOG-FLEET-HUMANEVALPLUS-02"
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-FLEET-HUMANEVALPLUS-03.json"
PREREGISTRATION = ROOT / "runs/research/BACKLOG-FLEET-HUMANEVALPLUS-03/PRE_REGISTRATION.md"
MAX_TOKENS = 1536
EXPECTED_TARGETS = {
    "qwen38": ("HumanEval/129", "HumanEval/130", "HumanEval/147"),
    "hauhaucs": ("HumanEval/32", "HumanEval/116", "HumanEval/129", "HumanEval/134", "HumanEval/147"),
    "fable-tc": (),
    "qwen36-moe": ("HumanEval/116", "HumanEval/129", "HumanEval/130", "HumanEval/132", "HumanEval/134"),
}
EXPECTED_HASHES = {
    ADMISSION: "5d49ba3f3831b7923cad55f073715ab29f827f710afa9b8599dc1381b13e188b",
    PREREGISTRATION: "797e2b67ee31bd16062ea08d2a6db2439e4ac12b19c6f8c4286a21e0bb034ba6",
    SOURCE / "raw/samples.jsonl": "3f2d5d2df02e2443e05324436db32ba8b4b1f7e6c7c5ac02032fad5f58bd8da2",
    SOURCE / "raw/official_scores.json": "035107400cca9ae9393b73b82df05657aa22cc245b461c0d62bf431adbae3159",
    SOURCE / "raw/receipt.json": "b6cce633af34db44f92d76f345e19b3c3b0a5e8ccc0c9b756904f545be9f615a",
    SOURCE / "raw/dataset_hashes.json": "79540c786ea1777478fc88b46e879568c01c204d204ee2c88c36469bdc35edc6",
    SOURCE / "PRE_REGISTRATION.md": "54d727d209264b11853f07ed59137414f2db10673cf5ec6a154473171d05f862",
    ROOT / "tools/research/run_fleet_humanevalplus_r2.py": "139e5596bcd5cb8fda1cbcf0813aaee87959513232885a146dd83b1913203fe2",
    r1.FLEET_REGISTRY: "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    r1.DATASET: "08d3df5c27a5f9a40176c27592b2d81e931b55d8d9edb7b1ffc28f2ccbdba735",
    r1.SCORER: "cdbefde3f0e12b0dbf8697aff154ea88ec07d7fafb2a959a3c88882f63f1aa0d",
    r1.SHARED_QA: "60af3eac1e119047e3b0d767c52ee8295ac44abbfbaa44b1c42eee45945336c6",
}


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_sources() -> tuple[dict[str, Any], list[pathlib.Path]]:
    host: dict[str, Any] = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen source mismatch: {path}: {actual} != {expected}")
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


def select_targets(records: list[dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    selected = {
        model: tuple(sorted(row["task_id"] for row in records
                            if row["model"] == model and row.get("truncated") is True))
        for model in r1.MODELS
    }
    expected = {model: tuple(sorted(ids)) for model, ids in EXPECTED_TARGETS.items()}
    if selected != expected or sum(map(len, selected.values())) != 13:
        raise ValueError(f"objective truncation target set changed: {selected}")
    return selected


def correction_payload(model: str, problem: dict[str, Any]) -> dict[str, Any]:
    request = r1.payload(model, problem)
    if request["max_tokens"] != r1.MAX_TOKENS:
        raise ValueError("R2 payload baseline changed")
    return request | {"max_tokens": MAX_TOKENS}


def generate_correction(model: str, problem: dict[str, Any]) -> dict[str, Any]:
    request = correction_payload(model, problem)
    started = time.perf_counter()
    status, response = r1.fleet.http_json(
        f"{r1.fleet.BASE_URL}/v1/chat/completions", request, timeout=1800
    )
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
    return {"model": model, "task_id": problem["task_id"], "http_status": status,
            "error": response.get("_error"), "answered": bool(text.strip()),
            "fenced": "```" in text, "finish_reason": finish_reason,
            "predicted_n": timings.get("predicted_n"),
            "decode_tps": timings.get("predicted_per_second"), "wall_s": elapsed,
            "truncated": finish_reason == "length" or (timings.get("predicted_n") or 0) >= MAX_TOKENS,
            "completion": text, "solution": r1.extract_code(text),
            "request_sha256": canonical_json_sha256(request), "response": response}


def merge_records(source: list[dict[str, Any]], corrections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replacements = {(row["model"], row["task_id"]): row for row in corrections}
    if len(replacements) != 13:
        raise ValueError("correction coverage is not exactly thirteen unique rows")
    merged = [replacements.get((row["model"], row["task_id"]), row) for row in source]
    for old, new in zip(source, merged):
        key = (old["model"], old["task_id"])
        if key not in replacements and canonical_json_sha256(old) != canonical_json_sha256(new):
            raise ValueError(f"non-target row mutated: {key}")
    return merged


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
    problems = {row["task_id"]: row for row in panel}
    source_records = read_jsonl(SOURCE / "raw/samples.jsonl")
    if len(source_records) != 656:
        raise ValueError("R2 matrix is not 4 x 164")
    selected = select_targets(source_records)
    r1.write_json(raw / "artifact_hashes.json", frozen)
    r1.write_json(raw / "dataset_hashes.json", {
        "dataset_sha256": sha256_file(r1.DATASET),
        "task_ids": [row["task_id"] for row in panel],
        "task_ids_sha256": canonical_json_sha256([row["task_id"] for row in panel]),
        "source_samples_sha256": sha256_file(SOURCE / "raw/samples.jsonl"),
        "objective_targets": selected,
        "objective_targets_sha256": canonical_json_sha256(selected),
    })

    initial_service = r1.fleet.service_state()
    initial_gateway = r1.fleet.gateway_status()
    initial_model = initial_gateway.get("current_model")
    if initial_service.get("active_state") != "active" or initial_model not in r1.MODELS:
        raise RuntimeError("initial route is not safely restorable")
    if r1.fleet.embedding_health() != 200:
        raise RuntimeError("embedding unhealthy before correction")
    route_snapshots: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    error: Exception | None = None
    restored_status: dict[str, Any] = {}
    try:
        for model in ("qwen38", "hauhaucs", "qwen36-moe"):
            status, gpu = r1.fleet.switch_model(model)
            snapshot = {"model": model, "status": status, "gpu": gpu,
                        "embedding_status": r1.fleet.embedding_health()}
            if snapshot["embedding_status"] != 200:
                raise RuntimeError(f"embedding unhealthy at {model}")
            route_snapshots.append(snapshot)
            for task_id in selected[model]:
                row = generate_correction(model, problems[task_id])
                if row["http_status"] != 200 or row["error"] or not row["answered"]:
                    raise RuntimeError(f"correction generation failed for {model}/{task_id}")
                r1.append_jsonl(raw / "corrections.jsonl", row)
                corrections.append(row)
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
        r1.write_json(raw / "recovery_state.json", recovery)
    if error:
        raise error

    merged = merge_records(source_records, corrections)
    for row in merged:
        r1.append_jsonl(raw / "samples.jsonl", row)
    scores: dict[str, dict[str, bool]] = {}
    for model in r1.MODELS:
        model_rows = [row for row in merged if row["model"] == model]
        if len(model_rows) != 164 or [row["task_id"] for row in model_rows] != [row["task_id"] for row in panel]:
            raise ValueError(f"merged matrix invalid for {model}")
        sample_path = raw / f"{model}.samples.jsonl"
        for row in model_rows:
            r1.append_jsonl(sample_path, {"task_id": row["task_id"], "solution": row["solution"]})
        scores[model] = r2.score_model(model, sample_path, raw)
        passed = sum(scores[model].values())
        r1.write_json(finalized / f"{model}.json", {"model": model,
                      "corrected": len(selected[model]), "plus_pass": passed,
                      "plus_pass_at_1": passed / 164})

    comparison = r1.paired_bootstrap(scores["hauhaucs"], scores["qwen38"])
    official = {model: {"n": 164, "plus_pass": sum(values.values()),
                        "plus_pass_at_1": sum(values.values()) / 164,
                        "per_task": values} for model, values in scores.items()}
    remaining = sum(bool(row["truncated"]) for row in corrections)
    metrics = {
        "r2_sources_and_model_artifacts_verified": True,
        "source_rows_selected_only_by_truncation": sum(map(len, selected.values())),
        "fresh_1536_token_regenerations": len(corrections),
        "models_rescored_by_evalplus": len(scores),
        "remaining_truncated_target_rows": remaining,
        "corrected_hauhaucs_humaneval_plus_pass_at_1": official["hauhaucs"]["plus_pass_at_1"],
        "corrected_paired_hauhaucs_minus_qwen38": comparison,
        "models": {model: {"plus_pass": official[model]["plus_pass"],
                            "plus_pass_at_1": official[model]["plus_pass_at_1"],
                            "corrected_rows": len(selected[model]),
                            "truncated": sum(row["truncated"] for row in merged if row["model"] == model)}
                   for model in r1.MODELS},
        "initial_route_and_services_restored": recovery["initial_route_and_services_restored"],
    }
    r1.write_json(raw / "actual_scores.json", metrics)
    r1.write_json(raw / "official_scores.json", official)
    r1.write_json(raw / "effective_route.json", {"models": r1.MODELS, "snapshots": route_snapshots})
    r1.write_json(raw / "service_identity.json", {"initial": initial_service, "final": recovery["final_service"]})
    r1.write_json(raw / "paired_baseline.json", {"hauhaucs": official["hauhaucs"],
                  "qwen38": official["qwen38"], "comparison": comparison})
    r1.write_json(raw / "hardware_metrics.json", {"route_snapshots": route_snapshots,
                  "final_gpu": r1.fleet.gpu_state()})
    r1.write_json(raw / "independent_evaluation.json", {"official_executor": "EvalPlus 0.3.1",
                  "scores": official, "paired_bootstrap": comparison,
                  "scorer_sha256": sha256_file(r1.SCORER),
                  "extractor_sha256": sha256_file(r1.SHARED_QA),
                  "selection_predicate": "source truncated is true", "max_tokens": MAX_TOKENS})
    source_receipt = json.loads((SOURCE / "raw/receipt.json").read_text(encoding="utf-8"))
    r1.write_json(raw / "source_execution_receipt.json", {
        "source_task_id": "BACKLOG-FLEET-HUMANEVALPLUS-02",
        "receipt_sha256": sha256_file(SOURCE / "raw/receipt.json"),
        "receipt_fingerprint": source_receipt["receipt_fingerprint"],
        "samples_sha256": sha256_file(SOURCE / "raw/samples.jsonl"),
        "official_scores_sha256": sha256_file(SOURCE / "raw/official_scores.json"),
    })

    gates = {
        "source_integrity": {"metric": "r2_sources_and_model_artifacts_verified", "operator": "eq", "threshold": True, "actual": True, "pass": True},
        "objective_selection": {"metric": "source_rows_selected_only_by_truncation", "operator": "eq", "threshold": 13, "actual": 13, "pass": True},
        "fresh_correction_coverage": {"metric": "fresh_1536_token_regenerations", "operator": "eq", "threshold": 13, "actual": len(corrections), "pass": len(corrections) == 13},
        "official_score_coverage": {"metric": "models_rescored_by_evalplus", "operator": "eq", "threshold": 4, "actual": len(scores), "pass": len(scores) == 4},
        "truncation_recovery": {"metric": "remaining_truncated_target_rows", "operator": "le", "threshold": 2, "actual": remaining, "pass": remaining <= 2},
        "coding_alias_absolute": {"metric": "corrected_hauhaucs_humaneval_plus_pass_at_1", "operator": "ge", "threshold": 0.80, "actual": official["hauhaucs"]["plus_pass_at_1"], "pass": official["hauhaucs"]["plus_pass_at_1"] >= 0.80},
        "coding_alias_noninferiority": {"metric": "corrected_paired_bootstrap_95ci_lower_hauhaucs_minus_qwen38", "operator": "gt", "threshold": -0.05, "actual": comparison["lower_95"], "pass": comparison["lower_95"] > -0.05},
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
    evidence_files = sorted({raw / value.removeprefix("raw/") for value in evidence.values()
                             if value != "raw/receipt.json"})
    provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(),
        started_at_utc=started, started_monotonic=mono,
        input_paths=[*input_paths, *evidence_files, raw / "corrections.jsonl",
                     *raw.glob("*.samples.jsonl"), *raw.glob("*.scores.json")],
        packages=["pytest"], runtime={"execution_mode": "humanevalplus_objective_cap_correction",
        "host_pid": os.getpid(), "source_generation_count": 656, "fresh_generation_count": 13,
        "max_tokens": MAX_TOKENS, "models": r1.MODELS, "evalplus_version": "0.3.1",
        "timing_is_evidence": False})
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise ValueError(f"incomplete provenance: {errors}")
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID,
               "provenance": provenance, "provenance_complete": True,
               "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    r1.write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = ("HAUHAUCS_HUMANEVALPLUS_CAP_CORRECTED_ALIAS_RETAINED_R3" if not failed
             else "HAUHAUCS_HUMANEVALPLUS_CAP_CORRECTED_ALIAS_NOT_RETAINED_R3")
    summary = ", ".join(f"{model}={official[model]['plus_pass']}/164" for model in r1.MODELS)
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Official corrected HumanEval+ scores: {summary}. Regenerated exactly 13 source-truncated "
        f"rows at 1536 tokens; {remaining} remained truncated. HauhauCS minus Qwen3.8 paired-bootstrap "
        f"95% interval `[{comparison['lower_95']:.4f}, {comparison['upper_95']:.4f}]`. "
        f"Failed gates: `{', '.join(failed) if failed else 'none'}`. The cap-correction family stops here.\n",
        encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
