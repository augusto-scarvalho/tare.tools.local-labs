#!/usr/bin/env python3
"""Run GDN02 R2 with retained vectors, independent scoring, and harness seal."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model_lifecycle.experiment_harness import ExperimentRun
from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)
from tools.research.run_gdn02_learned_state import service_state, wsl_path

TASK_ID = "BACKLOG-GDN02-LEARNED-STATE-02"
MODEL = pathlib.Path("/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe")
MODEL_SHA256 = "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c"
CORPUS = ROOT / "workloads/gsm8k.jsonl"
EXPECTED = {
    ROOT / "config/research_backlog_admissions/BACKLOG-GDN02-LEARNED-STATE-02.json": "a76fbb57414473b7cb60e45c064c33349c70d39f33476821e27053c37011d446",
    ROOT / "runs/research/BACKLOG-GDN02-LEARNED-STATE-02/PRE_REGISTRATION.md": "44f6a37087e039f2060cb7b9a884639e90af0800c8a6e30e1bf7ad578eb55b51",
    ROOT / "runs/research/BACKLOG-GDN02-LEARNED-STATE-01/raw/receipt.json": "3222aceaa925b48fda0b9eb684e32f5dac917e4e86a80c6fc65279cddbb7f236",
    ROOT / "docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md": "a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04",
    ROOT / "tools/research/run_gdn02_learned_state.py": "c6f37fccf63cff34d14a4b76595acd9bf50835ec2b3400f18c4742611d4e0fcb",
    ROOT / "tools/research/gdn02_learned_state_worker.py": "9d09cf2b93c870f14dadf20662a7f8298b14e60bedf91e2ca78ccca17d5970cf",
    CORPUS: "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
}


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_process(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False, timeout=timeout,
    )


def gate_rows(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    definitions = {
        "learned_states": ("learned_gdn_layer_cells", "ge", 3),
        "retained_cells": ("retained_decisive_layer_cells", "eq", 3),
        "retained_collateral": ("retained_collateral_cosines", "eq", 147),
        "independent_recompute": ("recomputed_metric_match_rate", "eq", 1.0),
        "target_leakage": ("median_old_fact_leakage_pct", "le", 5.0),
        "collateral_retention": ("median_collateral_retention_pct", "ge", 90.0),
        "update_fidelity": ("median_updated_fact_fidelity_pct", "ge", 95.0),
        "state_materiality": ("distinct_recurrent_state_conditions", "ge", 3),
    }
    operators = {
        "eq": lambda actual, threshold: actual == threshold,
        "ge": lambda actual, threshold: actual >= threshold,
        "le": lambda actual, threshold: actual <= threshold,
    }
    return {
        gate: {
            "metric": metric, "operator": operator, "threshold": threshold,
            "actual": metrics[metric], "pass": operators[operator](metrics[metric], threshold),
        }
        for gate, (metric, operator, threshold) in definitions.items()
    }


def execute(outdir: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = outdir / "raw"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    monotonic = time.monotonic()
    for path, expected in EXPECTED.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen input mismatch: {path}: {actual}")
    inputs = {path.relative_to(ROOT).as_posix(): digest for path, digest in EXPECTED.items()}
    worker = ROOT / "tools/research/gdn02_learned_state_worker_r2.py"
    scorer = ROOT / "tools/research/gdn02_retained_scorer.py"
    python = "/home/augus/.venvs/adapt00-20260824/bin/python"

    with ExperimentRun(raw, TASK_ID, inputs) as run:
        before = service_state()
        worker_json = raw / "worker.json"
        bundle = raw / "state_vectors.safetensors"
        worker_command = [
            "wsl", "-d", "Ubuntu-24.04", "--", python, wsl_path(worker),
            "--model", str(MODEL), "--corpus", wsl_path(CORPUS),
            "--output", wsl_path(worker_json), "--bundle", wsl_path(bundle),
            "--batch-size", "5",
        ]
        completed = run_process(worker_command, 1800)
        (raw / "worker.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (raw / "worker.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"worker failed {completed.returncode}: {completed.stderr[-4000:]}")
        worker_result = json.loads(worker_json.read_text(encoding="utf-8"))
        if worker_result["model_file_sha256"] != MODEL_SHA256:
            raise ValueError("worker model identity mismatch")
        if worker_result["corpus_sha256"] != EXPECTED[CORPUS]:
            raise ValueError("worker corpus identity mismatch")
        if worker_result["bundle_sha256"] != sha256_file(bundle):
            raise ValueError("worker bundle identity mismatch")

        score_path = raw / "recomputed_scores.json"
        score_command = [
            "wsl", "-d", "Ubuntu-24.04", "--", python, wsl_path(scorer),
            "--worker", wsl_path(worker_json), "--bundle", wsl_path(bundle),
            "--output", wsl_path(score_path),
        ]
        scored = run_process(score_command, 300)
        (raw / "scorer.stdout.log").write_text(scored.stdout, encoding="utf-8")
        (raw / "scorer.stderr.log").write_text(scored.stderr, encoding="utf-8")
        if scored.returncode != 0:
            raise RuntimeError(f"scorer failed {scored.returncode}: {scored.stderr[-4000:]}")
        score = json.loads(score_path.read_text(encoding="utf-8"))
        metrics = score["metrics"]
        after = service_state()
        if before["systemd"].get("MainPID") != after["systemd"].get("MainPID"):
            raise RuntimeError("serving process changed during representation experiment")
        if after["systemd"].get("NRestarts") != "0" or after["health"] != {"inference": 200, "embedding": 200}:
            raise RuntimeError(f"serving baseline unhealthy after worker: {after}")

        write_json(raw / "actual_scores.json", metrics)
        write_json(raw / "artifact_hashes.json", {
            "frozen_inputs": inputs,
            "model_file_sha256": MODEL_SHA256,
            "worker_sha256": sha256_file(worker),
            "scorer_sha256": sha256_file(scorer),
            "bundle_sha256": sha256_file(bundle),
            "bundle_bytes": bundle.stat().st_size,
        })
        write_json(raw / "dataset_hashes.json", {
            "corpus_sha256": EXPECTED[CORPUS],
            "records_semantic_sha256": canonical_json_sha256(worker_result["records"]),
        })
        write_json(raw / "failure_reproduction.json", {
            "r1_metrics": json.loads((ROOT / "runs/research/BACKLOG-GDN02-LEARNED-STATE-01/raw/actual_scores.json").read_text(encoding="utf-8")),
            "r2_recomputed_metrics": metrics,
        })
        write_json(raw / "falsifiable_hypothesis.json", {
            "leakage_max_pct": 5.0, "retention_min_pct": 90.0,
            "fidelity_min_pct": 95.0, "exact_recompute_rate": 1.0,
        })
        write_json(raw / "hardware_metrics.json", worker_result["hardware"] | {"elapsed_seconds": worker_result["elapsed_seconds"]})
        write_json(raw / "independent_evaluation.json", {
            "scorer": "gdn02_retained_scorer.py", "rows": score["rows"],
            "recomputed_metric_match_rate": metrics["recomputed_metric_match_rate"],
        })
        write_json(raw / "invalidation_rules.json", {
            "all_three_cells_required": True, "all_147_cosines_required": True,
            "scorer_agreement_required": True, "all_gates_required": True,
        })
        write_json(raw / "invariant_controls.json", {
            "layers": [0, 1, 2], "records": 50, "target_index": 5,
            "old_value": "41", "new_value": "42", "token_lengths": worker_result["token_lengths"],
        })
        write_json(raw / "paired_baseline.json", {
            "baseline": "old plus reaffirm-old", "treatment": "old plus correction-new",
            "oracle": "new plus reaffirm-new", "same_templates_and_lengths": True,
        })
        write_json(raw / "real_implementation.json", {
            "module_classes": [cell["module_class"] for cell in worker_result["cells"]],
            "official_recurrent_kernel": "torch_chunk_gated_delta_rule",
            "learned_checkpoint": MODEL_SHA256,
        })
        write_json(raw / "retained_state_bundle.json", {
            "path": "raw/state_vectors.safetensors", "sha256": sha256_file(bundle),
            "bytes": bundle.stat().st_size, "keys": worker_result["bundle_keys"],
            "decisive_layer_cells": 3, "collateral_cosines": 147,
        })
        write_json(raw / "semantic_parity.json", {
            "collateral_retention_by_layer": [row["collateral_retention_pct"] for row in score["rows"]]
        })
        write_json(raw / "source_execution_receipt.json", {
            "r1_receipt_sha256": EXPECTED[ROOT / "runs/research/BACKLOG-GDN02-LEARNED-STATE-01/raw/receipt.json"]
        })
        write_json(raw / "target_materiality.json", {
            "baseline_oracle_distances": [row["baseline_oracle_distance"] for row in score["rows"]],
            "all_material": all(row["baseline_oracle_distance"] >= 1e-4 for row in score["rows"]),
        })
        write_json(raw / "service_maintenance.json", {"before": before, "after": after, "service_untouched": True})
        for row in score["rows"]:
            run.record(row)
        run.checkpoint("retained_recompute_complete", {
            "cells": metrics["retained_decisive_layer_cells"],
            "collateral_cosines": metrics["retained_collateral_cosines"],
            "match_rate": metrics["recomputed_metric_match_rate"],
        })
        evidence = {
            "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
            "artifact_hashes": "raw/artifact_hashes.json", "dataset_hashes": "raw/dataset_hashes.json",
            "failure_reproduction": "raw/failure_reproduction.json", "falsifiable_hypothesis": "raw/falsifiable_hypothesis.json",
            "hardware_metrics": "raw/hardware_metrics.json", "independent_evaluation": "raw/independent_evaluation.json",
            "invalidation_rules": "raw/invalidation_rules.json", "invariant_controls": "raw/invariant_controls.json",
            "paired_baseline": "raw/paired_baseline.json", "provenance": "raw/receipt.json",
            "raw_samples": "raw/samples.jsonl", "real_implementation": "raw/real_implementation.json",
            "receipt_fingerprint": "raw/receipt.json", "retained_state_bundle": "raw/retained_state_bundle.json",
            "semantic_parity": "raw/semantic_parity.json", "source_execution_receipt": "raw/source_execution_receipt.json",
            "target_materiality": "raw/target_materiality.json",
        }
        provenance_inputs = [
            *EXPECTED, worker, scorer, worker_json, bundle, score_path,
            *[raw / path.removeprefix("raw/") for path in evidence.values() if path != "raw/receipt.json"],
        ]
        provenance = build_provenance(
            script_path=pathlib.Path(__file__).resolve(), started_at_utc=started,
            started_monotonic=monotonic, input_paths=provenance_inputs,
            packages=["torch", "transformers", "safetensors", "numpy"],
            runtime={"execution_mode": "retained_learned_gdn_state", "worker_command": worker_command, "scorer_command": score_command},
        )
        complete, errors = provenance_complete(provenance)
        if not complete:
            raise ValueError(errors)
        receipt = {
            "schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID,
            "provenance": provenance, "provenance_complete": True,
            "gates": gate_rows(metrics), "evidence": evidence,
        }
        sealed = run.seal(receipt)
    return sealed, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt, metrics = execute(args.outdir.resolve())
    passed = all(row["pass"] for row in receipt["gates"].values())
    claim = "GDN02_LEARNED_STATE_WITH_RETAINED_TENSORS_QUALIFIED_R2" if passed else "GDN02_LEARNED_STATE_WITH_RETAINED_TENSORS_REJECTED_R2"
    failed = [name for name, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Retained `{metrics['retained_decisive_layer_cells']}` cells and `{metrics['retained_collateral_cosines']}` collateral cosines; "
        f"independent match `{metrics['recomputed_metric_match_rate']:.4%}`; failed gates: `{', '.join(failed) if failed else 'none'}`. "
        "Scope remains learned recurrent-state representations.\n", encoding="utf-8"
    )
    print(json.dumps({"claim": claim, "gates": receipt["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
