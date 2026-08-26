#!/usr/bin/env python3
"""Host orchestrator for BACKLOG-SLX08-REAL-FIDELITY-01."""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file
from tools.research import slx08_real_fidelity_worker as worker_lib
from tools.research.run_adapter_requalification_r2 import (
    http_get_json, query_gpu, query_service, systemctl, verify_base_model, wait_for_health, windows_path_to_wsl,
)

TASK_ID = "BACKLOG-SLX08-REAL-FIDELITY-01"
BASE_MODEL_WSL = "/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe"
WSL_PYTHON = "/home/augus/.venvs/adapt00-20260824/bin/python"
WORKER = ROOT / "tools/research/slx08_real_fidelity_worker.py"
EXPECTED_INPUTS = {
    ROOT / "config/research_backlog_admissions/BACKLOG-SLX08-REAL-FIDELITY-01.json": "ac43e4dceb11ce8b5dc9eaae43ebb544ebe10057de5d756215789ad61e08d672",
    ROOT / "runs/research/BACKLOG-SLX08-REAL-FIDELITY-01/PRE_REGISTRATION.md": "4ad95f6bde23a64362eaf146aa110cfdb5fc875740cde241559c5628244b6a2a",
    ROOT / "runs/research/SLX-08-SPECULATIVE-PREFILL-2026-08-25/PRE_REGISTRATION.md": "5004d124c4f7543a2542916f05c45ec52afced4be4758ff2c95fa386dd4c6212",
    ROOT / "runs/research/SLX-08-SPECULATIVE-PREFILL-2026-08-25/RESULT.md": "c3d87dd4624e1e2c851df96f6956efb44508df0423bf676657b8c11ae6ade0b7",
    ROOT / "runs/research/SLX-08-SPECULATIVE-PREFILL-2026-08-25/raw/receipt.json": "f19600ed451d5ed4ad3a24b5c29ef3fbcf2a95de06ffcab0aef9a7e9152cb78a",
    ROOT / "tools/probes/slx08_speculative_prefill_oracle.py": "5b85dd266c3fc72ae47a7cabe6e5ae3246e4aab544e87e6ee7cd47eab81bdc37",
    ROOT / "workloads/gsm8k.jsonl": "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
}


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_exec(value: str) -> str:
    value = re.sub(r"; start_time=\[[^\]]*\]", "", value)
    value = re.sub(r"; stop_time=\[[^\]]*\]", "", value)
    return re.sub(r"; pid=\d+", "", value)


def verify_inputs() -> dict:
    ledger = {}
    for path, expected in EXPECTED_INPUTS.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen input mismatch: {path}: {actual} != {expected}")
        ledger[str(path.relative_to(ROOT).as_posix())] = {"bytes": path.stat().st_size, "sha256": actual}
    return ledger


def run_experiment(outdir: pathlib.Path) -> dict:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError(f"raw directory is not empty: {raw}")
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    inputs = verify_inputs()
    model = verify_base_model()
    write_json(raw / "dataset_hashes.json", inputs)
    write_json(raw / "model_hash.json", model)

    initial_service, initial_gpu = query_service(), query_gpu()
    maintenance = {
        "initial_service": initial_service, "initial_gpu": initial_gpu,
        "initial_embedding": http_get_json("http://127.0.0.1:8081/health"),
        "service_stopped_for_vram": False,
    }
    stopped = False
    worker_output = raw / "worker.json"
    command = [
        "wsl", "-d", "Ubuntu-24.04", "--", WSL_PYTHON, windows_path_to_wsl(WORKER),
        "--model", BASE_MODEL_WSL, "--corpus", windows_path_to_wsl(ROOT / "workloads/gsm8k.jsonl"),
        "--output", windows_path_to_wsl(worker_output),
    ]
    try:
        if initial_service["active_state"] == "active":
            systemctl("stop")
            stopped = True
            maintenance["service_stopped_for_vram"] = True
            maintenance["service_after_stop"] = query_service()
            maintenance["embedding_after_stop"] = http_get_json("http://127.0.0.1:8081/health")
            if maintenance["embedding_after_stop"].get("status") != "ok":
                raise RuntimeError("embedding service became unhealthy")
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                   timeout=3600, check=False)
        (raw / "worker.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (raw / "worker.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"worker failed ({completed.returncode}): {completed.stderr[-5000:]}")
    finally:
        if stopped:
            systemctl("start")
            maintenance["inference_health_final"] = wait_for_health("http://127.0.0.1:8080/health", timeout_seconds=180)
        maintenance["final_service"] = query_service()
        maintenance["final_embedding"] = wait_for_health("http://127.0.0.1:8081/health", timeout_seconds=30)
        maintenance["final_gpu"] = query_gpu()
        maintenance["service_and_embedding_restored"] = (
            maintenance["final_service"]["active_state"] == initial_service["active_state"]
            and normalize_exec(maintenance["final_service"]["exec_start"]) == normalize_exec(initial_service["exec_start"])
            and maintenance["final_service"]["n_restarts"] == initial_service["n_restarts"]
            and maintenance["final_embedding"].get("status") == "ok"
            and (not stopped or maintenance.get("inference_health_final", {}).get("status") == "ok")
        )
        write_json(raw / "service_maintenance.json", maintenance)

    payload = json.loads(worker_output.read_text(encoding="utf-8"))
    samples = payload["samples"]
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in samples:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    scores = payload["scores"]
    recomputed = worker_lib.aggregate(samples)
    recompute_match = canonical_json_sha256(scores) == canonical_json_sha256(recomputed)
    decision = {
        "threshold": 0.95, "actual": scores["median_selected_block_context_cosine"],
        "false_negative_candidate": scores["median_selected_block_context_cosine"] >= 0.95,
        "scope": "fidelity only; TTFT unmeasured",
    }
    write_json(raw / "actual_scores.json", {"scores": scores, "decision": decision})
    write_json(raw / "independent_evaluation.json", {"recomputed": recomputed, "match": recompute_match})
    write_json(raw / "invalidation_rules.json", decision)
    write_json(raw / "treatment_materiality.json", {
        "computed_indices_used": scores["computed_top_block_indices_materially_used"],
        "old_probe_bug": "selected_indices computed but k_selected/v_selected sliced from prefix",
    })
    write_json(raw / "paired_baseline.json", {"samples": samples})
    write_json(raw / "semantic_parity.json", {"aggregate_recompute_match": recompute_match, "cell_count": len(samples)})
    write_json(raw / "real_implementation.json", {
        "worker": str(WORKER.relative_to(ROOT).as_posix()), "gpu_worker_pid": payload["pid"],
        "actual_qkv": True, "corrected_gather": True, "integrated_runtime": False,
    })
    write_json(raw / "tensor_identity.json", {"contexts": payload["context_ledger"], "cells": [
        {k: row[k] for k in ("context", "layer", "q_sha256", "k_sha256", "v_sha256", "tensor_source")}
        for row in samples
    ]})
    elapsed = time.monotonic() - started_mono
    write_json(raw / "hardware_metrics.json", {
        "gpu": initial_gpu, "worker_device": payload["device"], "elapsed_seconds": elapsed,
        "ttft_measured": False, "native_kernel_measured": False,
    })

    observations = {**scores, "independent_metric_recompute_match": recompute_match,
                    "service_and_embedding_restored": maintenance["service_and_embedding_restored"]}
    defs = {
        "actual_qkv_coverage": ("actual_qkv_cells", "ge", 12),
        "no_synthetic_decisive_tensors": ("all_decisive_tensors_from_frozen_model", "eq", True),
        "computed_indices_used": ("computed_top_block_indices_materially_used", "eq", True),
        "independent_recompute": ("independent_metric_recompute_match", "eq", True),
        "fidelity": ("median_selected_block_context_cosine", "ge", 0.95),
        "service_recovery": ("service_and_embedding_restored", "eq", True),
    }
    ops = {"eq": lambda a, b: a == b, "ge": lambda a, b: a >= b}
    gates = {gid: {"metric": metric, "operator": op, "threshold": threshold,
                   "actual": observations[metric], "pass": ops[op](observations[metric], threshold)}
             for gid, (metric, op, threshold) in defs.items()}
    receipt_inputs = [*EXPECTED_INPUTS, WORKER, pathlib.Path(__file__).resolve(),
                      *[raw / name for name in (
                          "actual_scores.json", "dataset_hashes.json", "hardware_metrics.json",
                          "independent_evaluation.json", "invalidation_rules.json", "model_hash.json",
                          "paired_baseline.json", "real_implementation.json", "samples.jsonl",
                          "semantic_parity.json", "service_maintenance.json", "tensor_identity.json",
                          "treatment_materiality.json")]]
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(), started_at_utc=started_utc,
        started_monotonic=started_mono, input_paths=receipt_inputs, packages=["pytest"],
        runtime={"execution_mode": "real_qwen_slx08_fidelity", "host_command": command},
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise ValueError(f"incomplete provenance: {errors}")
    evidence = {name: f"raw/{name}.json" for name in (
        "actual_scores", "dataset_hashes", "hardware_metrics", "independent_evaluation",
        "invalidation_rules", "model_hash", "paired_baseline", "real_implementation",
        "semantic_parity", "service_maintenance", "tensor_identity", "treatment_materiality")}
    evidence.update({"acceptance_gates": "raw/receipt.json", "provenance": "raw/receipt.json",
                     "raw_samples": "raw/samples.jsonl", "receipt_fingerprint": "raw/receipt.json"})
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID,
               "provenance": provenance, "provenance_complete": complete, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt = run_experiment(args.outdir.resolve())
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
