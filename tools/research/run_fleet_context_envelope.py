#!/usr/bin/env python3
"""Measure route-specific physical needle retrieval within each per-slot context."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time
from collections import defaultdict
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (
    build_provenance, canonical_json_sha256, provenance_complete, sha256_file,
)
from tools.research import run_fleet_regression_screen as fleet

TASK_ID = "BACKLOG-FLEET-CONTEXT-ENVELOPE-02"
MODELS = ("qwen38", "hauhaucs", "fable-tc", "qwen36-moe")
TARGETS = {
    "qwen38": (4000, 16000, 28000),
    "hauhaucs": (4000, 16000, 28000),
    "fable-tc": (2000, 6000, 7600),
    "qwen36-moe": (4000, 12000, 17000),
}
SLOT_CONTEXT = {"qwen38": 32768, "hauhaucs": 32768, "fable-tc": 8192, "qwen36-moe": 18432}
POSITIONS = ("start", "middle", "end")
REPLICATES = (0, 1)
EXPECTED_REQUESTS = 72
PRE_REG_SHA256 = "e851482e89abd1bc417db405921a1b0d4a4195f17452689eebb41b6f23e845ae"
SOURCE_HASHES = {
    "config/research_backlog_admissions/BACKLOG-FLEET-CONTEXT-ENVELOPE-02.json": "3bee678e36d8016e782a586ec3b10b5f88dafb167ee1b2d48005a3a4789543b3",
    "runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-01/PRE_REGISTRATION.md": "58496fcb6d374d2f08cc945404c1ed7cd1f68a067df70a90be3f9bc6fe3ae642",
    "runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-01/PIPELINE.json": "a8faf0c57dc74a4b46a074b73dedefbc737e0acfd365c07d54febaeec4b5caa8",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    "tools/research/run_fleet_regression_screen.py": "7cbf942375c120970ac78fb40f7f15050ebddb7ec5372cb40bd721009faf1de3",
    "runs/research/BACKLOG-GATEWAY-ROUTE-STRESS-01/raw/receipt.json": "527e308b2aa54fe96bb641f1d5380b04b42e7871245d173ee107cec0dabbfe41",
}


def verify_inputs() -> tuple[dict[str, Any], list[pathlib.Path]]:
    own: dict[str, Any] = {}
    paths: list[pathlib.Path] = []
    for relative, expected in SOURCE_HASHES.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen source mismatch: {relative}: {actual} != {expected}")
        own[relative] = {"bytes": path.stat().st_size, "sha256": actual}
        paths.append(path)
    prereg = ROOT / "runs/research" / TASK_ID / "PRE_REGISTRATION.md"
    actual = sha256_file(prereg)
    if actual != PRE_REG_SHA256:
        raise ValueError(f"preregistration mismatch: {actual}")
    own[prereg.relative_to(ROOT).as_posix()] = {"bytes": prereg.stat().st_size, "sha256": actual}
    paths.append(prereg)
    base, base_paths = fleet.verify_frozen_inputs()
    return {"experiment": own, "fleet_base": base}, [*paths, *base_paths]


def filler_line(index: int) -> str:
    return (f"Archive record {index:06d}: amber catalog ordinary transit ledger "
            "contains no requested access credential.\n")


def access_code(model: str, target: int, position: str, replicate: int) -> str:
    prefix = {"qwen38": "Q38", "hauhaucs": "HAU", "fable-tc": "FAB", "qwen36-moe": "Q36"}[model]
    return f"NX-{prefix}-{target:05d}-{position[0].upper()}{replicate}"


def make_prompt(filler_count: int, position: str, code: str) -> str:
    records = [filler_line(index) for index in range(filler_count)]
    needle = f"SECURE ACCESS RECORD: The requested access code is {code}.\n"
    insertion = 0 if position == "start" else len(records) // 2 if position == "middle" else len(records)
    records.insert(insertion, needle)
    return (
        "Read the archive below. Find the single SECURE ACCESS RECORD. "
        "Return only its access code, with no explanation.\n\n"
        + "".join(records)
        + "\nQuestion: What is the requested access code? Return only the code."
    )


def token_count(text: str) -> int:
    status, response = fleet.http_json(
        f"{fleet.BASE_URL}/tokenize",
        {"content": text, "add_special": False},
        timeout=180.0,
    )
    tokens = response.get("tokens") if status == 200 else None
    if not isinstance(tokens, list):
        raise RuntimeError(f"tokenizer failure: {status} {response}")
    return len(tokens)


def fit_filler_count(target: int) -> tuple[int, int]:
    probe_code = "NX-PROBE-00000-M0"
    low, high = 0, max(32, target // 12)
    high_tokens = token_count(make_prompt(high, "middle", probe_code))
    while high_tokens < target:
        low = high
        high *= 2
        high_tokens = token_count(make_prompt(high, "middle", probe_code))
    while high - low > 1:
        middle = (low + high) // 2
        count = token_count(make_prompt(middle, "middle", probe_code))
        if count < target:
            low = middle
        else:
            high = middle
    candidates = [(low, token_count(make_prompt(low, "middle", probe_code))),
                  (high, high_tokens if high == max(32, target // 12) else token_count(make_prompt(high, "middle", probe_code)))]
    return min(candidates, key=lambda item: abs(item[1] - target))


def build_cases(model: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for target in TARGETS[model]:
        filler_count, fitted_tokens = fit_filler_count(target)
        for position in POSITIONS:
            for replicate in REPLICATES:
                code = access_code(model, target, position, replicate)
                prompt = make_prompt(filler_count, position, code)
                exact_tokens = token_count(prompt)
                cases.append({"model": model, "target_tokens": target,
                              "position": position, "replicate": replicate,
                              "code": code, "filler_count": filler_count,
                              "fitted_probe_tokens": fitted_tokens,
                              "tokenizer_tokens": exact_tokens,
                              "prompt_chars": len(prompt),
                              "prompt_sha256": canonical_json_sha256(prompt),
                              "prompt": prompt})
    return cases


def exact_recall(content: str, code: str) -> bool:
    return fleet.normalize_text(content) == fleet.normalize_text(code)


def summarize(rows: list[dict[str, Any]], recovery: dict[str, Any], verified: int) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        by_model[model] = {
            "correct": sum(row["exact_recall"] for row in selected),
            "total": len(selected),
            "exact_recall": sum(row["exact_recall"] for row in selected) / len(selected),
            "by_target": {str(target): sum(row["exact_recall"] for row in selected if row["target_tokens"] == target) / 6
                          for target in TARGETS[model]},
        }
    by_position = {
        position: sum(row["exact_recall"] for row in rows if row["position"] == position)
                  / sum(row["position"] == position for row in rows)
        for position in POSITIONS
    }
    successful = sum(row["http_status"] == 200 and not row["error"] for row in rows)
    within = sum(row["prompt_n"] <= SLOT_CONTEXT[row["model"]] - 64 for row in rows)
    return {
        "verified_model_artifacts": verified,
        "recorded_requests": len(rows),
        "successful_response_rate": successful / len(rows),
        "requests_within_route_slot_context": within,
        "qwen38_exact_recall": by_model["qwen38"]["exact_recall"],
        "hauhaucs_exact_recall": by_model["hauhaucs"]["exact_recall"],
        "fable_tc_exact_recall": by_model["fable-tc"]["exact_recall"],
        "qwen36_moe_exact_recall": by_model["qwen36-moe"]["exact_recall"],
        "minimum_position_bucket_recall": min(by_position.values()),
        "position_buckets": by_position,
        "models": by_model,
        "initial_route_and_services_restored": recovery["initial_route_and_services_restored"],
    }


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    finalized = raw / "finalized"
    finalized.mkdir(parents=True)
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    frozen, frozen_paths = verify_inputs()
    fleet.write_json(raw / "artifact_hashes.json", frozen)
    initial_service = fleet.service_state()
    initial_gateway = fleet.gateway_status()
    initial_model = initial_gateway.get("current_model")
    if initial_service.get("active_state") != "active" or initial_model not in MODELS:
        raise RuntimeError("initial text route is not safely restorable")
    if fleet.embedding_health() != 200:
        raise RuntimeError("embedding unhealthy before context experiment")
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    error: Exception | None = None
    restored_status: dict[str, Any] = {}
    try:
        for model in MODELS:
            status, gpu = fleet.switch_model(model)
            snapshot = {"model": model, "status": status, "gpu": gpu,
                        "embedding_status": fleet.embedding_health(),
                        "slot_context": SLOT_CONTEXT[model]}
            if snapshot["embedding_status"] != 200:
                raise RuntimeError(f"embedding unhealthy at {model}")
            snapshots.append(snapshot)
            cases = build_cases(model)
            for case in cases:
                if case["tokenizer_tokens"] > SLOT_CONTEXT[model] - 64:
                    raise RuntimeError(f"case exceeds per-slot context: {model} {case['tokenizer_tokens']}")
                prompt = case.pop("prompt")
                manifests.append(case)
                payload = {"model": model,
                           "messages": [{"role": "user", "content": prompt}],
                           "temperature": 0.0, "top_k": 1, "seed": 20260827,
                           "max_tokens": 32, "stream": False, "cache_prompt": False,
                           "chat_template_kwargs": {"enable_thinking": False}}
                begin = time.perf_counter()
                http_status, response = fleet.http_json(
                    f"{fleet.BASE_URL}/v1/chat/completions", payload, timeout=1800.0)
                wall_ms = round((time.perf_counter() - begin) * 1000, 3)
                try:
                    content = str(response["choices"][0]["message"].get("content") or "")
                except (KeyError, IndexError, TypeError):
                    content = ""
                timings = response.get("timings") or {}
                prompt_n = int(timings.get("prompt_n") or 0)
                if prompt_n <= 0:
                    raise RuntimeError(f"missing physical prompt_n for {model}")
                row = {**case, "http_status": http_status, "error": response.get("_error"),
                       "wall_ms": wall_ms, "content": content,
                       "exact_recall": exact_recall(content, case["code"]),
                       "prompt_n": prompt_n, "timings": timings, "response": response}
                fleet.append_jsonl(raw / "samples.jsonl", row)
                rows.append(row)
            fleet.write_json(finalized / f"{model}.json", {
                "model": model, "recorded": len(cases),
                "correct": sum(row["exact_recall"] for row in rows if row["model"] == model)})
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
        recovery = {"initial_service": initial_service, "initial_gateway": initial_gateway,
                    "initial_model": initial_model, "restored_gateway": restored_status,
                    "final_service": final_service, "final_embedding_status": final_embedding,
                    "initial_route_and_services_restored": (
                        restored_status.get("current_model") == initial_model
                        and restored_status.get("backend_healthy") is True
                        and final_service.get("active_state") == "active"
                        and final_service.get("main_pid") == initial_service.get("main_pid")
                        and final_service.get("n_restarts") == initial_service.get("n_restarts")
                        and final_embedding == 200)}
        fleet.write_json(raw / "recovery_state.json", recovery)
    if error:
        raise error

    metrics = summarize(rows, recovery, len(frozen["fleet_base"]["wsl_artifacts"]))
    fleet.write_json(raw / "actual_scores.json", metrics)
    manifest_public = [{key: value for key, value in case.items()} for case in manifests]
    fleet.write_json(raw / "case_manifest.json", {"generator": "deterministic_archive_v1",
                     "cases": manifest_public, "slot_context": SLOT_CONTEXT,
                     "targets": TARGETS})
    fleet.write_json(raw / "dataset_hashes.json", {
        "case_manifest_sha256": canonical_json_sha256(manifest_public),
        "prompt_hashes": [case["prompt_sha256"] for case in manifest_public]})
    fleet.write_json(raw / "effective_route.json", {"snapshots": snapshots,
                     "slot_context": SLOT_CONTEXT})
    fleet.write_json(raw / "service_identity.json", {"initial": initial_service,
                     "initial_gateway": initial_gateway, "final": recovery["final_service"],
                     "restored_gateway": restored_status})
    fleet.write_json(raw / "hardware_metrics.json", {
        "snapshots": snapshots,
        "by_model": {model: {"median_wall_ms": statistics.median(row["wall_ms"] for row in rows if row["model"] == model),
                              "median_prompt_n": statistics.median(row["prompt_n"] for row in rows if row["model"] == model)}
                     for model in MODELS}, "timing_is_evidence": False})
    fleet.write_json(raw / "paired_baseline.json", {"positions": metrics["position_buckets"],
                     "models": metrics["models"], "comparison_scope": "descriptive only"})
    fleet.write_json(raw / "independent_evaluation.json", {"scoring": "normalized exact code only",
                     "metrics": metrics, "independent_review_pending": True})
    source_path = ROOT / "runs/research/BACKLOG-GATEWAY-ROUTE-STRESS-01/raw/receipt.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    fleet.write_json(raw / "source_execution_receipt.json", {
        "source_task_id": "BACKLOG-GATEWAY-ROUTE-STRESS-01",
        "receipt_sha256": sha256_file(source_path),
        "receipt_fingerprint": source["receipt_fingerprint"]})

    definitions = {
        "artifact_identity": ("verified_model_artifacts", "eq", 4),
        "request_coverage": ("recorded_requests", "eq", 72),
        "request_integrity": ("successful_response_rate", "eq", 1.0),
        "context_fit": ("requests_within_route_slot_context", "eq", 72),
        "qwen38_recall": ("qwen38_exact_recall", "ge", 0.90),
        "hauhaucs_recall": ("hauhaucs_exact_recall", "ge", 0.90),
        "fable_recall": ("fable_tc_exact_recall", "ge", 0.90),
        "qwen36_moe_recall": ("qwen36_moe_exact_recall", "ge", 0.90),
        "position_robustness": ("minimum_position_bucket_recall", "ge", 0.80),
        "service_recovery": ("initial_route_and_services_restored", "eq", True),
    }
    gates: dict[str, Any] = {}
    for gate_id, (metric, operator, threshold) in definitions.items():
        actual = metrics[metric]
        passed = actual == threshold if operator == "eq" else actual >= threshold
        gates[gate_id] = {"metric": metric, "operator": operator,
                          "threshold": threshold, "actual": actual, "pass": passed}
    evidence = {"acceptance_gates": "raw/receipt.json", "artifact_hashes": "raw/artifact_hashes.json",
                "case_manifest": "raw/case_manifest.json", "dataset_hashes": "raw/dataset_hashes.json",
                "effective_route": "raw/effective_route.json", "hardware_metrics": "raw/hardware_metrics.json",
                "independent_evaluation": "raw/independent_evaluation.json", "paired_baseline": "raw/paired_baseline.json",
                "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl",
                "receipt_fingerprint": "raw/receipt.json", "recovery_state": "raw/recovery_state.json",
                "service_identity": "raw/service_identity.json", "source_execution_receipt": "raw/source_execution_receipt.json"}
    evidence_files = sorted({raw / value.removeprefix("raw/") for value in evidence.values()
                             if value != "raw/receipt.json"})
    provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(),
        started_at_utc=started_utc, started_monotonic=started_mono,
        input_paths=[*frozen_paths, *evidence_files], packages=[],
        runtime={"execution_mode": "physical_per_slot_context_retrieval",
                 "host_pid": os.getpid(), "requests": len(rows),
                 "models": MODELS, "timing_is_evidence": False})
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(f"incomplete provenance: {errors}")
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID,
               "provenance": provenance, "provenance_complete": True,
               "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    fleet.write_json(raw / "receipt.json", receipt)
    failed = [gate_id for gate_id, gate in gates.items() if not gate["pass"]]
    claim = ("QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_MEASURED_R2" if not failed
             else "QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_NOT_CONFIRMED_R2")
    model_summary = ", ".join(f"{model}={metrics['models'][model]['correct']}/18" for model in MODELS)
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Exact bounded retrieval: {model_summary}. Minimum position-bucket recall "
        f"`{metrics['minimum_position_bucket_recall']:.4f}`. Failed gates: "
        f"`{', '.join(failed) if failed else 'none'}`.\n", encoding="utf-8", newline="\n")
    return receipt


def selfcheck() -> None:
    assert sum(len(TARGETS[model]) * len(POSITIONS) * len(REPLICATES) for model in MODELS) == EXPECTED_REQUESTS
    assert SLOT_CONTEXT["qwen36-moe"] == 73728 // 4
    assert max(TARGETS["qwen36-moe"]) < SLOT_CONTEXT["qwen36-moe"]
    assert exact_recall("`NX-Q38-04000-S0`", "NX-Q38-04000-S0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
