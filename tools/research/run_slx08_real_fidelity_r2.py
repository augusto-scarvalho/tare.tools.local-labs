#!/usr/bin/env python3
"""Host orchestrator for retained-context SLX-08 fidelity R2."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)
from tools.research.run_adapter_requalification_r2 import (
    http_get_json,
    query_gpu,
    query_service,
    systemctl,
    verify_base_model,
    wait_for_health,
    windows_path_to_wsl,
)
from tools.research.run_slx08_real_fidelity import normalize_exec, write_json

TASK_ID = "BACKLOG-SLX08-REAL-FIDELITY-02"
BASE_MODEL_WSL = "/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe"
WSL_PYTHON = "/home/augus/.venvs/adapt00-20260824/bin/python"
WORKER = ROOT / "tools/research/slx08_real_fidelity_worker_r2.py"
SCORER = ROOT / "tools/research/slx08_context_scorer.py"
RUNNER = pathlib.Path(__file__).resolve()
EXPECTED_INPUTS = {
    ROOT / "config/research_backlog_admissions/BACKLOG-SLX08-REAL-FIDELITY-02.json": "f8bfd39378849d3e251e9e227187cabec8ef5153d2da4fa01b7425a9a8cc5369",
    ROOT / "runs/research/BACKLOG-SLX08-REAL-FIDELITY-02/PRE_REGISTRATION.md": "ec0e1024a0a8995e5d977b1d72ebd97804dc62503089d3db58a562d4608cd117",
    ROOT / "runs/research/BACKLOG-SLX08-REAL-FIDELITY-01/raw/receipt.json": "6e5212692ff4e8fa3ac50eab13e144a7e08cb933b2c44ecf3bf55568c9b4e660",
    ROOT / "docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md": "a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04",
    ROOT / "tools/research/run_slx08_real_fidelity.py": "0b8e7bc733d7bad7e51ee2edc3586f1259ce1bbcfe4d5f1357ebcc109564efa5",
    ROOT / "tools/research/slx08_real_fidelity_worker.py": "2b628e2fdf864216d40e538ac60cb1f9ac09f6d325aa6c70225bc1c05fa8e1b7",
    ROOT / "workloads/gsm8k.jsonl": "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
}


def verify_inputs() -> dict:
    ledger = {}
    for path, expected in EXPECTED_INPUTS.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen input mismatch: {path}: {actual} != {expected}")
        ledger[str(path.relative_to(ROOT).as_posix())] = {
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    return ledger


def run_checked(command: list[str], stdout_path: pathlib.Path, stderr_path: pathlib.Path) -> None:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {completed.stderr[-5000:]}")


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
        "initial_service": initial_service,
        "initial_gpu": initial_gpu,
        "initial_embedding": http_get_json("http://127.0.0.1:8081/health"),
        "service_stopped_for_vram": False,
    }
    stopped = False
    worker_output = raw / "worker.json"
    bundle = raw / "context_vectors.safetensors"
    evaluation = raw / "context_evaluation.json"
    worker_command = [
        "wsl", "-d", "Ubuntu-24.04", "--", WSL_PYTHON,
        windows_path_to_wsl(WORKER),
        "--model", BASE_MODEL_WSL,
        "--corpus", windows_path_to_wsl(ROOT / "workloads/gsm8k.jsonl"),
        "--output", windows_path_to_wsl(worker_output),
        "--bundle", windows_path_to_wsl(bundle),
    ]
    scorer_command = [
        "wsl", "-d", "Ubuntu-24.04", "--", WSL_PYTHON,
        windows_path_to_wsl(SCORER),
        "--bundle", windows_path_to_wsl(bundle),
        "--worker", windows_path_to_wsl(worker_output),
        "--output", windows_path_to_wsl(evaluation),
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
        run_checked(worker_command, raw / "worker.stdout.log", raw / "worker.stderr.log")
        run_checked(scorer_command, raw / "scorer.stdout.log", raw / "scorer.stderr.log")
    finally:
        if stopped:
            systemctl("start")
            maintenance["inference_health_final"] = wait_for_health(
                "http://127.0.0.1:8080/health", timeout_seconds=180
            )
        maintenance["final_service"] = query_service()
        maintenance["final_embedding"] = wait_for_health(
            "http://127.0.0.1:8081/health", timeout_seconds=30
        )
        maintenance["final_gpu"] = query_gpu()
        maintenance["service_and_embedding_restored"] = (
            maintenance["final_service"]["active_state"] == initial_service["active_state"]
            and normalize_exec(maintenance["final_service"]["exec_start"])
            == normalize_exec(initial_service["exec_start"])
            and maintenance["final_service"]["n_restarts"] == initial_service["n_restarts"]
            and maintenance["final_embedding"].get("status") == "ok"
            and (not stopped or maintenance.get("inference_health_final", {}).get("status") == "ok")
        )
        write_json(raw / "service_maintenance.json", maintenance)

    payload = json.loads(worker_output.read_text(encoding="utf-8"))
    scored = json.loads(evaluation.read_text(encoding="utf-8"))
    samples, scores, summary = payload["samples"], payload["scores"], scored["summary"]
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in samples:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(raw / "actual_scores.json", {"worker": scores, "independent": summary})
    write_json(raw / "independent_evaluation.json", {
        "scorer": str(SCORER.relative_to(ROOT).as_posix()),
        "all_projections_match": summary["recomputed_projection_match_rate"] == 1.0,
        "bundle_sha256": sha256_file(bundle),
        "bundle_bytes": bundle.stat().st_size,
    })
    write_json(raw / "scorer_hashes.json", {
        "worker": {"path": str(WORKER.relative_to(ROOT).as_posix()), "sha256": sha256_file(WORKER)},
        "scorer": {"path": str(SCORER.relative_to(ROOT).as_posix()), "sha256": sha256_file(SCORER)},
        "runner": {"path": str(RUNNER.relative_to(ROOT).as_posix()), "sha256": sha256_file(RUNNER)},
    })
    write_json(raw / "paired_baseline.json", {"samples": samples})
    write_json(raw / "semantic_parity.json", {
        "cell_count": len(samples),
        "projection_match_rate": summary["recomputed_projection_match_rate"],
        "nonfinite_values": summary["nonfinite_values"],
    })
    write_json(raw / "real_implementation.json", {
        "worker": str(WORKER.relative_to(ROOT).as_posix()),
        "scorer": str(SCORER.relative_to(ROOT).as_posix()),
        "gpu_worker_pid": payload["pid"],
        "actual_qkv": True,
        "corrected_gather": True,
        "integrated_runtime": False,
    })
    write_json(raw / "tensor_identity.json", {
        "contexts": payload["context_ledger"],
        "cells": [{
            key: row[key] for key in (
                "cell", "context", "layer", "q_sha256", "k_sha256", "v_sha256",
                "context_vector_sha256", "tensor_source"
            )
        } for row in samples],
        "bundle_sha256": sha256_file(bundle),
    })
    write_json(raw / "treatment_materiality.json", {
        "computed_indices_used": scores["computed_top_block_indices_materially_used"],
        "selected_indices": {row["cell"]: row["selected_block_indices"] for row in samples},
        "causal_limit": "real-QKV substitution and corrected gather are jointly material",
    })
    write_json(raw / "hardware_metrics.json", {
        "gpu": initial_gpu,
        "worker_device": payload["device"],
        "elapsed_seconds": time.monotonic() - started_mono,
        "bundle_bytes": bundle.stat().st_size,
        "ttft_measured": False,
        "native_kernel_measured": False,
    })

    observations = {
        **scores,
        **summary,
        "service_and_embedding_restored": maintenance["service_and_embedding_restored"],
    }
    definitions = {
        "actual_qkv_coverage": ("actual_qkv_cells", "ge", 12),
        "no_synthetic_decisive_tensors": ("all_decisive_tensors_from_frozen_model", "eq", True),
        "computed_indices_used": ("computed_top_block_indices_materially_used", "eq", True),
        "context_bundle_coverage": ("retained_context_cells", "eq", 12),
        "context_projection_match": ("recomputed_projection_match_rate", "eq", 1.0),
        "fidelity": ("recomputed_median_selected_block_context_cosine", "ge", 0.95),
        "service_recovery": ("service_and_embedding_restored", "eq", True),
    }
    operators = {"eq": lambda actual, threshold: actual == threshold,
                 "ge": lambda actual, threshold: actual >= threshold}
    gates = {
        gate_id: {
            "metric": metric,
            "operator": operator,
            "threshold": threshold,
            "actual": observations[metric],
            "pass": operators[operator](observations[metric], threshold),
        }
        for gate_id, (metric, operator, threshold) in definitions.items()
    }
    raw_inputs = [
        raw / name for name in (
            "actual_scores.json", "context_evaluation.json", "context_vectors.safetensors",
            "dataset_hashes.json", "hardware_metrics.json", "independent_evaluation.json",
            "model_hash.json", "paired_baseline.json", "real_implementation.json",
            "samples.jsonl", "scorer.stderr.log", "scorer.stdout.log", "scorer_hashes.json",
            "semantic_parity.json", "service_maintenance.json", "tensor_identity.json",
            "treatment_materiality.json", "worker.json", "worker.stderr.log", "worker.stdout.log",
        )
    ]
    provenance = build_provenance(
        script_path=RUNNER,
        started_at_utc=started_utc,
        started_monotonic=started_mono,
        input_paths=[*EXPECTED_INPUTS, WORKER, SCORER, RUNNER, *raw_inputs],
        packages=["pytest"],
        runtime={
            "execution_mode": "real_qwen_slx08_retained_context_fidelity_r2",
            "worker_command": worker_command,
            "scorer_command": scorer_command,
        },
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise ValueError(f"incomplete provenance: {errors}")
    evidence = {
        "acceptance_gates": "raw/receipt.json",
        "actual_scores": "raw/actual_scores.json",
        "context_bundle": "raw/context_vectors.safetensors",
        "context_evaluation": "raw/context_evaluation.json",
        "dataset_hashes": "raw/dataset_hashes.json",
        "hardware_metrics": "raw/hardware_metrics.json",
        "independent_evaluation": "raw/independent_evaluation.json",
        "model_hash": "raw/model_hash.json",
        "paired_baseline": "raw/paired_baseline.json",
        "provenance": "raw/receipt.json",
        "raw_samples": "raw/samples.jsonl",
        "real_implementation": "raw/real_implementation.json",
        "receipt_fingerprint": "raw/receipt.json",
        "scorer_hashes": "raw/scorer_hashes.json",
        "semantic_parity": "raw/semantic_parity.json",
        "service_maintenance": "raw/service_maintenance.json",
        "tensor_identity": "raw/tensor_identity.json",
        "treatment_materiality": "raw/treatment_materiality.json",
    }
    receipt = {
        "schema": "local-labs-backlog-receipt-v1",
        "task_id": TASK_ID,
        "provenance": provenance,
        "provenance_complete": complete,
        "gates": gates,
        "evidence": evidence,
    }
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
