#!/usr/bin/env python3
"""Harness-bound longitudinal replication of the qualified text fleet."""
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
from tools.analysis.experiment_provenance import build_provenance, provenance_complete, sha256_file
from tools.research import run_fleet_regression_screen as base

TASK_ID = "BACKLOG-FLEET-REGRESSION-SCREEN-03"
PRE_REG_SHA256 = "1d6a63c372846402718ee93f2bdf97ab26e6caa8b3b16eeb711c407da64ac852"
SOURCE_HASHES = {
    "config/research_backlog_admissions/BACKLOG-FLEET-REGRESSION-SCREEN-03.json": "8746a25d050af19e49c5e8e68499a72cc31e3b8f130ea143681e85c0530cd0c6",
    "runs/research/BACKLOG-FLEET-REGRESSION-SCREEN-02/raw/receipt.json": "23f3f94634e3e9be5451bd0daafcb7e28dd7d338ddf9bddbe6b81b15a267e598",
    "runs/research/BACKLOG-FLEET-REGRESSION-SCREEN-02/REVIEW.json": "e0fd0c20f510bb70b50694a828450b6e47bd9fd86ae798059e34a3ae5836be27",
    "tools/research/run_fleet_regression_screen_r2.py": "7f577e3e3ab9054b7dd48f0279852b033fd4ba323c53b9d3ee06cdb710303fd7",
    "tools/research/run_fleet_regression_screen.py": "7cbf942375c120970ac78fb40f7f15050ebddb7ec5372cb40bd721009faf1de3",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    "workloads/gsm8k.jsonl": "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    "runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl": "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
    "tools/benchmarks/agent_suite_v2.py": "14d0a1b76d4d729228678f215ecefa3254aef214eb65ac9d8d7061bccc0dc59e",
}


def write_json(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def evaluate_gates(metrics: dict) -> dict:
    definitions = {
        "r2_binding": ("promoted_r2_verified", "eq", True),
        "route_coverage": ("route_models_completed", "eq", 4),
        "request_coverage": ("recorded_requests", "eq", 448),
        "request_integrity": ("successful_response_rate", "eq", 1.0),
        "route_identity": ("route_identity_verified", "eq", True),
        "repeatability": ("exact_repeat_rate", "ge", 0.95),
        "request_retention": ("retained_request_payloads", "eq", 448),
        "service_integrity": ("service_restarts", "eq", 0),
        "embedding_integrity": ("embedding_health", "eq", 200),
        "service_recovery": ("initial_model_restored", "eq", True),
    }
    operators = {"eq": lambda actual, expected: actual == expected, "ge": lambda actual, expected: actual >= expected}
    return {
        gate_id: {
            "metric": metric,
            "operator": operator,
            "threshold": threshold,
            "actual": metrics[metric],
            "pass": operators[operator](metrics[metric], threshold),
        }
        for gate_id, (metric, operator, threshold) in definitions.items()
    }


def execute(outdir: pathlib.Path) -> tuple[dict, dict]:
    raw = outdir / "raw"
    inputs = {}
    for relative, expected in SOURCE_HASHES.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen input mismatch: {relative}: {actual} != {expected}")
        inputs[relative] = actual
    prereg = outdir / "PRE_REGISTRATION.md"
    if sha256_file(prereg) != PRE_REG_SHA256:
        raise ValueError("preregistration mismatch")
    inputs[prereg.relative_to(ROOT).as_posix()] = PRE_REG_SHA256
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()

    with ExperimentRun(raw, TASK_ID, inputs, requires_restoration=True) as run:
        run.checkpoint("host_inputs_verified", {"input_count": len(inputs), "expected_requests": 448})
        physical_outdir = raw / "physical"
        (physical_outdir / "raw").mkdir(parents=True)
        original = (base.TASK_ID, base.PRE_REG_SHA256, base.SOURCE_HASHES, base.__file__)
        try:
            base.TASK_ID = TASK_ID
            base.PRE_REG_SHA256 = PRE_REG_SHA256
            base.SOURCE_HASHES = SOURCE_HASHES
            base.__file__ = __file__
            physical_receipt = base.execute(physical_outdir)
        finally:
            base.TASK_ID, base.PRE_REG_SHA256, base.SOURCE_HASHES, base.__file__ = original

        physical_raw = physical_outdir / "raw"
        rows = base.read_jsonl(physical_raw / "samples.jsonl")
        metrics = json.loads((physical_raw / "actual_scores.json").read_text(encoding="utf-8"))
        metrics["promoted_r2_verified"] = True
        metrics["retained_request_payloads"] = sum(isinstance(row.get("request"), dict) for row in rows)
        for row in rows:
            run.record(row)
        restored = (
            metrics["initial_model_restored"] is True
            and metrics["embedding_health"] == 200
            and metrics["service_restarts"] == 0
        )
        run.restored({
            "initial_model_restored": metrics["initial_model_restored"],
            "embedding_health": metrics["embedding_health"],
            "service_restarts": metrics["service_restarts"],
        }, ok=restored)
        gates = evaluate_gates(metrics)

        r2_scores = json.loads((ROOT / "runs/research/BACKLOG-FLEET-REGRESSION-SCREEN-02/raw/actual_scores.json").read_text(encoding="utf-8"))
        write_json(raw / "actual_scores.json", metrics)
        write_json(raw / "artifact_hashes.json", json.loads((physical_raw / "frozen_inputs.json").read_text(encoding="utf-8")))
        write_json(raw / "independent_evaluation.json", {
            "r2": {
                "successful_response_rate": r2_scores["successful_response_rate"],
                "exact_repeat_rate": r2_scores["exact_repeat_rate"],
            },
            "r3": {
                "successful_response_rate": metrics["successful_response_rate"],
                "exact_repeat_rate": metrics["exact_repeat_rate"],
            },
            "quality_scores_are_descriptive": True,
        })
        write_json(raw / "source_execution_receipt.json", {
            "r2_receipt_sha256": SOURCE_HASHES["runs/research/BACKLOG-FLEET-REGRESSION-SCREEN-02/raw/receipt.json"],
            "physical_r3_receipt_sha256": sha256_file(physical_raw / "receipt.json"),
        })

        nested_evidence = [path for path in physical_raw.rglob("*") if path.is_file()]
        provenance = build_provenance(
            script_path=pathlib.Path(__file__).resolve(),
            started_at_utc=started_utc,
            started_monotonic=started_mono,
            input_paths=[
                *[ROOT / relative for relative in SOURCE_HASHES],
                prereg,
                pathlib.Path(__file__).resolve(),
                raw / "actual_scores.json",
                raw / "artifact_hashes.json",
                raw / "independent_evaluation.json",
                raw / "source_execution_receipt.json",
                *nested_evidence,
            ],
            packages=["pytest"],
            runtime={
                "execution_mode": "qualified_fleet_longitudinal_replication",
                "requests": len(rows),
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
            "effective_route": "raw/physical/raw/effective_route.json",
            "environment": "raw/physical/raw/environment.json",
            "hardware_metrics": "raw/physical/raw/hardware_metrics.json",
            "independent_evaluation": "raw/independent_evaluation.json",
            "paired_baseline": "raw/physical/raw/paired_baseline.json",
            "provenance": "raw/receipt.json",
            "raw_samples": "raw/samples.jsonl",
            "receipt_fingerprint": "raw/receipt.json",
            "recovery_state": "raw/physical/raw/recovery_state.json",
            "service_identity": "raw/physical/raw/service_identity.json",
            "service_maintenance": "raw/physical/raw/service_maintenance.json",
            "source_execution_receipt": "raw/source_execution_receipt.json",
            "treatment_controls": "raw/physical/raw/treatment_controls.json",
        }
        receipt = run.seal({
            "schema": "local-labs-backlog-receipt-v1",
            "task_id": TASK_ID,
            "provenance": provenance,
            "provenance_complete": True,
            "gates": gates,
            "evidence": evidence,
            "physical_receipt_fingerprint": physical_receipt["receipt_fingerprint"],
        })
    return receipt, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt, metrics = execute(args.outdir.resolve())
    passed = all(row["pass"] for row in receipt["gates"].values())
    claim = "QUALIFIED_TEXT_FLEET_LONGITUDINAL_REPLICATION_R3" if passed else "QUALIFIED_TEXT_FLEET_LONGITUDINAL_REJECTED_R3"
    failures = [gate for gate, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review. "
        f"Recorded {metrics['recorded_requests']} requests; success {metrics['successful_response_rate']:.6f}; "
        f"exact repeat {metrics['exact_repeat_rate']:.6f}; failed gates: {', '.join(failures) if failures else 'none'}. "
        "No model-quality or production-SLO claim.\n",
        encoding="utf-8",
    )
    print(json.dumps({"claim": claim, "gates": receipt["gates"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
