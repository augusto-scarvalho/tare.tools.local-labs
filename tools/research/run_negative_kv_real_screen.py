#!/usr/bin/env python3
"""Host orchestrator for BACKLOG-NEGATIVE-KV-REAL-SCREEN-01."""
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

from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)
from tools.research import negative_kv_real_worker as worker_lib
from tools.research.run_adapter_requalification_r2 import (
    http_get_json,
    query_gpu,
    query_service,
    systemctl,
    verify_base_model,
    wait_for_health,
    windows_path_to_wsl,
)

TASK_ID = "BACKLOG-NEGATIVE-KV-REAL-SCREEN-01"
BASE_MODEL_WSL = "/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe"
WSL_PYTHON = "/home/augus/.venvs/adapt00-20260824/bin/python"
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-NEGATIVE-KV-REAL-SCREEN-01.json"
PREREG = ROOT / "runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-01/PRE_REGISTRATION.md"
CORPUS = ROOT / "workloads/gsm8k.jsonl"
WORKER = ROOT / "tools/research/negative_kv_real_worker.py"
EXPECTED_INPUTS = {
    ADMISSION: "16d845d3debc066fb9aa2852e9249605c7fc96d1005b089338846ae0b4dbff68",
    PREREG: "dce466a9cdfc8a34b7baf1afe52f15752891bd063a32ac3d2f28450d7aebfe11",
    CORPUS: "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    ROOT / "runs/research/RSH-01-FIBQUANT-2026-08-25/RESULT.md": "f68d18cd0113f0e36a3b1146e4840b2f4fb7cfff8913c21898d0d7da40626d7e",
    ROOT / "runs/research/REP-03-KVARN-OFFLINE-2026-08-25/RESULT.md": "3ec9a7f93659511d04a1254a05f9ad792a3ffd3ac543b244a40401305e0b68d1",
    ROOT / "runs/research/RSH-03-KVLINC-COMPENSATION-2026-08-25/RESULT.md": "4e6fb2bdd5eb6949fb1a7a57e3861c5517f34b97e7dec78f92f0bc0a9f0ee014",
    ROOT / "runs/research/RSH-04-RABITQ-CACHE-2026-08-25/RESULT.md": "9873ce73c576e2a2a055c3dd29983c213736c510a1de1cfc2da212ba23147e48",
    ROOT / "runs/research/REP-06-ENTROPY-PRECISION-2026-08-25/RESULT.md": "98d45bad61c76787d585e61e1aa9fc7742922f8b2ebd27f0b935c0202795530b",
    ROOT / "tools/probes/rsh01_fibquant_simulation.py": "8a0cd28fe10d3c6b50a7bcf5df4740442a043b3717b1807099dfae4ec6bfa80d",
    ROOT / "tools/probes/rep03_kvarn_codec.py": "b27ff1cd5df40e82f58c1f6719a9e992fec6df5948a307688f643690e26f1973",
    ROOT / "tools/probes/rsh03_kvlinc_compensation.py": "a77eee4852575149186d54eff35ff9afa7e937a5adea9c15482d37613068e346",
    ROOT / "tools/probes/rsh04_rabitq_cache.py": "ced47b5c945bc496d53eff40f12cf34bb36ad0826bf9a315a09dfe5e517926ea",
    ROOT / "tools/probes/rep06_dynamic_entropy_precision.py": "709be19a12535435cde12aecac792f3175c132892cbba938f31b01f503af879d",
}


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_exec_start(value: str) -> str:
    value = re.sub(r"; start_time=\[[^\]]*\]", "", value)
    value = re.sub(r"; stop_time=\[[^\]]*\]", "", value)
    value = re.sub(r"; pid=\d+", "", value)
    return value


def verify_inputs(expected_inputs: dict[pathlib.Path, str] = EXPECTED_INPUTS) -> dict:
    ledger = {}
    for path, expected in expected_inputs.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen input mismatch: {path}: {actual} != {expected}")
        ledger[str(path.relative_to(ROOT).as_posix())] = {"bytes": path.stat().st_size, "sha256": actual}
    return ledger


def candidate_decisions(scores: dict) -> dict:
    return {
        "RSH-01": {
            "pass": scores["rsh01_fib_mse_ratio_vs_uniform"] <= 0.70
            and scores["rsh01_fib_sqnr_gain_db"] >= 2.5
            and scores["rsh01_fib_cosine_similarity"] >= 0.995,
            "rules": ["mse_ratio<=0.70", "sqnr_gain_db>=2.5", "cosine>=0.995"],
        },
        "REP-03": {
            "pass": scores["rep03_hadamard_mse_reduction"] >= 0.50
            and scores["rep03_hadamard_attention_cosine"] >= 0.99,
            "rules": ["mse_reduction>=0.50", "attention_cosine>=0.99"],
        },
        "RSH-03": {
            "pass": scores["rsh03_rank4_mse_recovery"] >= 0.50
            and scores["rsh03_rank4_output_cosine"] >= 0.998
            and scores["rsh03_rank4_parameter_overhead"] <= 0.01,
            "rules": ["mse_recovery>=0.50", "output_cosine>=0.998", "parameter_overhead<=0.01"],
        },
        "RSH-04": {
            "pass": scores["rsh04_binary_top_block_recall"] >= 0.90
            and scores["rsh04_retained_fraction"] <= 0.30,
            "rules": ["top_block_recall>=0.90", "retained_fraction<=0.30"],
        },
        "REP-06": {
            "pass": scores["rep06_average_bits_per_element"] <= 7.0
            and scores["rep06_dynamic_attention_cosine"] >= 0.992
            and scores["rep06_dynamic_beats_static_int4"] is True,
            "rules": ["average_bits<=7.0", "attention_cosine>=0.992", "beats_static_int4=true"],
        },
    }


def build_gates(scores: dict, recompute_match: bool, service_restored: bool,
                continuation_verified: bool | None = None) -> dict:
    definitions = {
        "actual_activation_coverage": ("actual_model_activation_cells", "ge", 18),
        "actual_weight_coverage": ("actual_model_weight_matrices", "ge", 12),
        "candidate_coverage": ("candidate_hypotheses_evaluated", "eq", 5),
        "no_synthetic_decisive_tensors": ("all_decisive_tensors_from_frozen_model", "eq", True),
        "independent_recompute": ("independent_metric_recompute_match", "eq", True),
        "rsh01_mse": ("rsh01_fib_mse_ratio_vs_uniform", "le", 0.70),
        "rsh01_sqnr": ("rsh01_fib_sqnr_gain_db", "ge", 2.5),
        "rsh01_cosine": ("rsh01_fib_cosine_similarity", "ge", 0.995),
        "rep03_mse": ("rep03_hadamard_mse_reduction", "ge", 0.50),
        "rep03_attention": ("rep03_hadamard_attention_cosine", "ge", 0.99),
        "rsh03_recovery": ("rsh03_rank4_mse_recovery", "ge", 0.50),
        "rsh03_cosine": ("rsh03_rank4_output_cosine", "ge", 0.998),
        "rsh03_overhead": ("rsh03_rank4_parameter_overhead", "le", 0.01),
        "rsh04_recall": ("rsh04_binary_top_block_recall", "ge", 0.90),
        "rsh04_dram": ("rsh04_retained_fraction", "le", 0.30),
        "rep06_bits": ("rep06_average_bits_per_element", "le", 7.0),
        "rep06_attention": ("rep06_dynamic_attention_cosine", "ge", 0.992),
        "rep06_beats_static": ("rep06_dynamic_beats_static_int4", "eq", True),
        "service_recovery": ("service_and_embedding_restored", "eq", True),
    }
    observations = {**scores, "independent_metric_recompute_match": recompute_match,
                    "service_and_embedding_restored": service_restored}
    if continuation_verified is not None:
        definitions = {"continuation_integrity":
                       ("frozen_failed_predecessor_verified", "eq", True), **definitions}
        observations["frozen_failed_predecessor_verified"] = continuation_verified
    operators = {
        "eq": lambda a, b: a == b, "ge": lambda a, b: a >= b,
        "le": lambda a, b: a <= b,
    }
    return {
        gate_id: {"metric": metric, "operator": op, "threshold": threshold,
                  "actual": observations[metric], "pass": operators[op](observations[metric], threshold)}
        for gate_id, (metric, op, threshold) in definitions.items()
    }


def run_experiment(outdir: pathlib.Path, *, task_id: str = TASK_ID,
                   expected_inputs: dict[pathlib.Path, str] = EXPECTED_INPUTS,
                   continuation_inputs: dict[pathlib.Path, str] | None = None) -> dict:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError(f"raw output directory is not empty: {raw}")
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    input_ledger = verify_inputs(expected_inputs)
    model_ledger = verify_base_model()
    write_json(raw / "dataset_hashes.json", {"inputs": input_ledger})
    write_json(raw / "model_hash.json", model_ledger)
    continuation_verified = None
    if continuation_inputs is not None:
        continuation_ledger = {
            str(path.relative_to(ROOT).as_posix()): {
                "bytes": path.stat().st_size, "sha256": sha256_file(path), "expected_sha256": expected,
            }
            for path, expected in continuation_inputs.items()
        }
        continuation_verified = all(row["sha256"] == row["expected_sha256"]
                                    for row in continuation_ledger.values())
        if not continuation_verified:
            raise ValueError("frozen failed predecessor did not verify")
        write_json(raw / "continuation_ledger.json", continuation_ledger)

    initial_service = query_service()
    initial_gpu = query_gpu()
    initial_embedding = http_get_json("http://127.0.0.1:8081/health")
    maintenance = {
        "initial_service": initial_service, "initial_gpu": initial_gpu,
        "initial_embedding": initial_embedding, "service_stopped_for_vram": False,
    }
    service_stopped = False
    worker_output = raw / "worker.json"
    command = [
        "wsl", "-d", "Ubuntu-24.04", "--", WSL_PYTHON,
        windows_path_to_wsl(WORKER), "--model", BASE_MODEL_WSL,
        "--corpus", windows_path_to_wsl(CORPUS),
        "--output", windows_path_to_wsl(worker_output),
    ]
    try:
        if initial_service["active_state"] == "active":
            systemctl("stop")
            service_stopped = True
            maintenance["service_stopped_for_vram"] = True
            maintenance["service_after_stop"] = query_service()
            maintenance["embedding_after_stop"] = http_get_json("http://127.0.0.1:8081/health")
            if maintenance["embedding_after_stop"].get("status") != "ok":
                raise RuntimeError("embedding service became unhealthy")
        completed = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=3600, check=False,
        )
        (raw / "worker.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (raw / "worker.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"GPU worker failed ({completed.returncode}): {completed.stderr[-5000:]}")
    finally:
        if service_stopped:
            systemctl("start")
            maintenance["inference_health_final"] = wait_for_health("http://127.0.0.1:8080/health", timeout_seconds=180)
        maintenance["final_service"] = query_service()
        maintenance["final_embedding"] = wait_for_health("http://127.0.0.1:8081/health", timeout_seconds=30)
        maintenance["final_gpu"] = query_gpu()
        maintenance["service_and_embedding_restored"] = (
            maintenance["final_service"]["active_state"] == initial_service["active_state"]
            and normalize_exec_start(maintenance["final_service"]["exec_start"])
            == normalize_exec_start(initial_service["exec_start"])
            and maintenance["final_service"]["n_restarts"] == initial_service["n_restarts"]
            and maintenance["final_embedding"].get("status") == "ok"
            and (not service_stopped or maintenance.get("inference_health_final", {}).get("status") == "ok")
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
    decisions = candidate_decisions(scores)
    write_json(raw / "actual_scores.json", {"scores": scores, "candidate_decisions": decisions})
    write_json(raw / "independent_evaluation.json", {
        "recomputed_scores": recomputed, "independent_metric_recompute_match": recompute_match,
        "worker_score_sha256": canonical_json_sha256(scores),
        "recomputed_score_sha256": canonical_json_sha256(recomputed),
    })
    write_json(raw / "invalidation_rules.json", decisions)
    write_json(raw / "tensor_identity.json", {
        "context_ledger": payload["context_ledger"],
        "cells": [{key: row[key] for key in row if key in {
            "candidate", "context", "layer", "slice", "tensor_sha256", "tensor_shape", "tensor_source"
        }} for row in samples],
    })
    write_json(raw / "real_implementation.json", {
        "worker": str(WORKER.relative_to(ROOT).as_posix()),
        "host_orchestrator": str(pathlib.Path(__file__).resolve().relative_to(ROOT).as_posix()),
        "gpu_worker_pid": payload["pid"], "model_class": payload["transformers_model_class"],
        "synthetic_decisive_inputs": False,
        "treatment_only_randomness": {"candidate": "RSH-04", "projection_seeds": list(worker_lib.PROJECTION_SEEDS)},
    })
    write_json(raw / "semantic_parity.json", {
        "individual_rows_retained": len(samples), "aggregate_recompute_match": recompute_match,
        "candidate_row_counts": {candidate: sum(row["candidate"] == candidate for row in samples)
                                 for candidate in sorted({row["candidate"] for row in samples})},
    })
    write_json(raw / "paired_baseline.json", {"candidate_decisions": decisions, "scores": scores})
    elapsed = time.monotonic() - started_mono
    write_json(raw / "hardware_metrics.json", {
        "initial_gpu": initial_gpu, "worker_device": payload["device"],
        "elapsed_seconds": elapsed, "physical_packing_measured": False,
        "native_kernel_throughput_measured": False,
    })

    gates = build_gates(scores, recompute_match, maintenance["service_and_embedding_restored"],
                        continuation_verified)
    receipt_inputs = [*expected_inputs.keys(), WORKER, pathlib.Path(__file__).resolve(),
                      raw / "actual_scores.json", raw / "dataset_hashes.json",
                      raw / "hardware_metrics.json", raw / "independent_evaluation.json",
                      raw / "invalidation_rules.json", raw / "model_hash.json",
                      raw / "paired_baseline.json", raw / "real_implementation.json",
                      raw / "samples.jsonl", raw / "semantic_parity.json",
                      raw / "service_maintenance.json", raw / "tensor_identity.json"]
    if continuation_inputs is not None:
        receipt_inputs.append(raw / "continuation_ledger.json")
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(), started_at_utc=started_utc,
        started_monotonic=started_mono, input_paths=receipt_inputs,
        packages=["pytest"], runtime={"execution_mode": "real_qwen_negative_kv_screen", "host_command": command},
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise ValueError(f"incomplete provenance: {errors}")
    receipt = {
        "schema": "local-labs-backlog-receipt-v1", "task_id": task_id,
        "provenance": provenance, "provenance_complete": complete, "gates": gates,
        "evidence": {
            "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
            "dataset_hashes": "raw/dataset_hashes.json", "hardware_metrics": "raw/hardware_metrics.json",
            "independent_evaluation": "raw/independent_evaluation.json",
            "invalidation_rules": "raw/invalidation_rules.json", "model_hash": "raw/model_hash.json",
            "paired_baseline": "raw/paired_baseline.json", "provenance": "raw/receipt.json",
            "raw_samples": "raw/samples.jsonl", "real_implementation": "raw/real_implementation.json",
            "receipt_fingerprint": "raw/receipt.json", "semantic_parity": "raw/semantic_parity.json",
            "service_maintenance": "raw/service_maintenance.json", "tensor_identity": "raw/tensor_identity.json",
        },
    }
    if continuation_inputs is not None:
        receipt["evidence"]["continuation_ledger"] = "raw/continuation_ledger.json"
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
