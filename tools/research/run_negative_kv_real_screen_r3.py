#!/usr/bin/env python3
"""Harness-bound R3 negative-KV screen with replayable tensor retention."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_lifecycle.experiment_harness import ExperimentRun
from tools.analysis.experiment_provenance import (
    build_provenance,
    provenance_complete,
    sha256_file,
)
from tools.research import run_negative_kv_real_screen as r1
from tools.research import run_negative_kv_real_screen_r2 as r2

TASK_ID = "BACKLOG-NEGATIVE-KV-REAL-SCREEN-03"
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-NEGATIVE-KV-REAL-SCREEN-03.json"
PREREG = ROOT / "runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-03/PRE_REGISTRATION.md"
WORKER = ROOT / "tools/research/negative_kv_real_worker_r3.py"
EXPECTED = dict(r2.EXPECTED_INPUTS)
EXPECTED.update({
    ADMISSION: "a9f11108e986e7ccdb1c7fa134d67214e13cf84cfb0e08e9e59dd7eb39548234",
    PREREG: "c99972053af918a38e8f367a6ceb6af7003ce8f34a45962d63cc39c5ef7f60ff",
    ROOT / "runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-02/raw/receipt.json": "18fac16ec34a64258fec6e83f36aed713803eb0766fb4f4e382bacc1fc57fc4e",
    ROOT / "runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-02/REVIEW.json": "fe967a16c32a16a49d59bc511ebfcf1d961b374ad96a4cbe35eb1bf897080252",
    ROOT / "tools/research/run_negative_kv_real_screen_r2.py": "579e8225464fd0b8f9d21968c08fe9d7f6d12b86445b641ccb11b37d2c5d46b7",
    ROOT / "tools/research/run_negative_kv_real_screen.py": "5295afff2c3f8e4fe0ce9f6c85c9409ffb1ae8285f54085d25a95beabdfec97a",
    ROOT / "tools/research/negative_kv_real_worker.py": "a3d059da1f80592d4a0a3c35c6a8f36e3b9deb5118b3200dd0402505b08534c4",
})


def write_json(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def evaluate_gates(metrics: dict) -> dict:
    definitions = {
        "r2_audit_bound": ("blocked_r2_review_verified", "eq", True),
        "activation_coverage": ("actual_model_activation_cells", "ge", 18),
        "weight_coverage": ("actual_model_weight_matrices", "ge", 12),
        "candidate_coverage": ("candidate_hypotheses_evaluated", "eq", 5),
        "sample_retention": ("retained_candidate_cells", "eq", 78),
        "tensor_retention": ("retained_decisive_tensor_files", "eq", 39),
        "tensor_hashes": ("retained_tensor_hashes_verified", "eq", True),
        "service_recovery": ("service_and_embedding_restored", "eq", True),
    }
    operators = {"eq": lambda a, b: a == b, "ge": lambda a, b: a >= b}
    return {
        gate_id: {
            "metric": metric, "operator": operator, "threshold": threshold,
            "actual": metrics[metric], "pass": operators[operator](metrics[metric], threshold),
        }
        for gate_id, (metric, operator, threshold) in definitions.items()
    }


def execute(outdir: pathlib.Path) -> tuple[dict, dict, dict]:
    raw = outdir / "raw"
    inputs = {}
    for path, expected in EXPECTED.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen input mismatch: {path}: {actual} != {expected}")
        inputs[path.relative_to(ROOT).as_posix()] = actual
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()

    with ExperimentRun(raw, TASK_ID, inputs, requires_restoration=True) as run:
        run.checkpoint("inputs_verified", {"frozen_inputs": len(inputs)})
        physical_outdir = raw / "physical"
        (physical_outdir / "raw").mkdir(parents=True)
        original_worker = r1.WORKER
        try:
            r1.WORKER = WORKER
            physical_receipt = r1.run_experiment(
                physical_outdir,
                task_id=TASK_ID,
                expected_inputs=EXPECTED,
            )
        finally:
            r1.WORKER = original_worker

        physical_raw = physical_outdir / "raw"
        worker_payload = json.loads((physical_raw / "worker.json").read_text(encoding="utf-8"))
        samples = worker_payload["samples"]
        tensor_ledger = worker_payload["retained_tensors"]
        tensor_dir = physical_raw / "tensors"
        verified = len(tensor_ledger) == 39 and all(
            (tensor_dir / row["file"]).is_file()
            and (tensor_dir / row["file"]).stat().st_size == row["bytes"]
            and sha256_file(tensor_dir / row["file"]) == row["sha256"]
            for row in tensor_ledger
        )
        if len(samples) != 78 or not verified:
            raise RuntimeError("retained tensor/sample bundle is incomplete")
        for sample in samples:
            run.record(sample)

        scores = worker_payload["scores"]
        decisions = r1.candidate_decisions(scores)
        maintenance = json.loads((physical_raw / "service_maintenance.json").read_text(encoding="utf-8"))
        restored = maintenance["service_and_embedding_restored"]
        run.restored({
            "service_and_embedding_restored": restored,
            "tensor_files": len(tensor_ledger),
        }, ok=restored)
        metrics = {
            "blocked_r2_review_verified": True,
            "actual_model_activation_cells": scores["actual_model_activation_cells"],
            "actual_model_weight_matrices": scores["actual_model_weight_matrices"],
            "candidate_hypotheses_evaluated": scores["candidate_hypotheses_evaluated"],
            "retained_candidate_cells": len(samples),
            "retained_decisive_tensor_files": len(tensor_ledger),
            "retained_tensor_hashes_verified": verified,
            "service_and_embedding_restored": restored,
        }
        gates = evaluate_gates(metrics)
        write_json(raw / "actual_scores.json", {"metrics": metrics, "scores": scores, "candidate_decisions": decisions})
        write_json(raw / "artifact_hashes.json", {"retained_tensors": tensor_ledger})
        write_json(raw / "retained_tensors.json", {
            "schema": "negative-kv-retained-tensors-v1",
            "root": "raw/physical/raw/tensors",
            "count": len(tensor_ledger),
            "files": tensor_ledger,
        })
        write_json(raw / "independent_evaluation.json", {
            "status": "PENDING_INDEPENDENT_REPLAY",
            "replayable_candidate_cells": len(samples),
            "retained_decisive_tensor_files": len(tensor_ledger),
        })
        write_json(raw / "scorer_hashes.json", {
            "frozen_metric_implementation": sha256_file(ROOT / "tools/research/negative_kv_real_worker.py"),
            "retention_worker": sha256_file(WORKER),
        })

        tensor_paths = [tensor_dir / row["file"] for row in tensor_ledger]
        provenance_inputs = [
            *EXPECTED.keys(), pathlib.Path(__file__).resolve(), WORKER,
            physical_raw / "receipt.json", raw / "actual_scores.json",
            raw / "artifact_hashes.json", raw / "retained_tensors.json",
            raw / "independent_evaluation.json", raw / "scorer_hashes.json",
            *tensor_paths,
        ]
        provenance = build_provenance(
            script_path=pathlib.Path(__file__).resolve(),
            started_at_utc=started_utc,
            started_monotonic=started_mono,
            input_paths=provenance_inputs,
            packages=["pytest"],
            runtime={
                "execution_mode": "real_qwen_negative_kv_retained_tensor_screen",
                "physical_receipt": "raw/physical/raw/receipt.json",
            },
        )
        complete, errors = provenance_complete(provenance)
        if not complete:
            raise ValueError(f"incomplete provenance: {errors}")
        evidence = {
            "acceptance_gates": "raw/receipt.json",
            "actual_scores": "raw/actual_scores.json",
            "artifact_hashes": "raw/artifact_hashes.json",
            "dataset_hashes": "raw/physical/raw/dataset_hashes.json",
            "hardware_metrics": "raw/physical/raw/hardware_metrics.json",
            "independent_evaluation": "raw/independent_evaluation.json",
            "invalidation_rules": "raw/physical/raw/invalidation_rules.json",
            "model_hash": "raw/physical/raw/model_hash.json",
            "paired_baseline": "raw/physical/raw/paired_baseline.json",
            "provenance": "raw/receipt.json",
            "raw_samples": "raw/samples.jsonl",
            "real_implementation": "raw/physical/raw/real_implementation.json",
            "receipt_fingerprint": "raw/receipt.json",
            "scorer_hashes": "raw/scorer_hashes.json",
            "service_maintenance": "raw/physical/raw/service_maintenance.json",
            "tensor_identity": "raw/physical/raw/tensor_identity.json",
            "retained_tensors": "raw/retained_tensors.json",
        }
        receipt = run.seal({
            "schema": "local-labs-backlog-receipt-v1",
            "task_id": TASK_ID,
            "provenance": provenance,
            "provenance_complete": True,
            "gates": gates,
            "evidence": evidence,
            "physical_receipt_sha256": sha256_file(physical_raw / "receipt.json"),
        })
    return receipt, scores, decisions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt, scores, decisions = execute(args.outdir.resolve())
    reversed_candidates = [candidate for candidate, row in decisions.items() if row["pass"]]
    claim = (
        f"{reversed_candidates[0].replace('-', '')}_FALSE_NEGATIVE_CANDIDATE_R3"
        if len(reversed_candidates) == 1
        else "NEGATIVE_KV_REAL_SCREEN_VERIFIED_R3"
    )
    (args.outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent tensor replay. "
        f"Retained 78 candidate cells and 39 decisive tensor files. Candidate reversals: "
        f"{', '.join(reversed_candidates) if reversed_candidates else 'none'}. "
        "No packed-byte, VRAM, kernel, throughput or deployment claim.\n",
        encoding="utf-8",
    )
    print(json.dumps({"claim": claim, "scores": scores, "gates": receipt["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
