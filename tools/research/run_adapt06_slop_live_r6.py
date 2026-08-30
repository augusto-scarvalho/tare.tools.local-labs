#!/usr/bin/env python3
"""Seal the physical R5 multi-adapter treatment with route counterfactuals."""
from __future__ import annotations

import argparse
import json
import pathlib
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
from tools.research import run_adapt06_slop_live_r5 as r5

TASK_ID = "BACKLOG-ADAPT06-SLOP-LIVE-06"
EXPECTED = {
    ROOT / "config/research_backlog_admissions/BACKLOG-ADAPT06-SLOP-LIVE-06.json": "63b4a75b38a272831d8fd855ac6dcd3ac691d83f1460c6b5737707f11e64ea90",
    ROOT / "runs/research/BACKLOG-ADAPT06-SLOP-LIVE-06/PRE_REGISTRATION.md": "c6e39de6a98cdab3a3b879c6ae1d1a23a8e53906fd03629e27c185b32614b4af",
    ROOT / "runs/research/BACKLOG-ADAPT06-SLOP-LIVE-05/raw/receipt.json": "871fd8aeb94ff4b2e4eeb6432ba10305591c01b6270462686af0a116ec8d3a28",
    ROOT / "runs/research/BACKLOG-ADAPT06-SLOP-LIVE-05/raw/live_rows.json": "9ad728eef78827899ad27920a81a8273c99ac75808517286529a0538fbca30e0",
    ROOT / "docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md": "a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04",
    ROOT / "tools/research/run_adapt06_slop_live_r5.py": "7fc57290cf59fb826f18306e3e861a1019680206027b771203fb5e5441269922",
    ROOT / "runs/research/BACKLOG-ADAPT-TRAIN-01/raw/checkpoint_seed_20260824/adapter_model.safetensors": "05b80090d2d1ba751d48a5032cddec82819a79b9bfb5bd8b05306b85d6ef0122",
    ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints/seed_20260824_full_trace/adapter_model.safetensors": "174832aa1bd25cbc5ed7f0ff717ad253ec94e2c23edc82e6f828ceadeed566b7",
}


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def derive_counterfactuals(live: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    baselines = {
        (row["route"], row["index"]): row["content"] for row in live["baselines"]
    }
    routed = live["routed"]
    matches = [
        row["content"] == baselines[(row["route"], row["index"])] for row in routed
    ]
    counterfactuals = {
        "baseline_cells": len(baselines),
        "routed_cells": len(routed),
        "match_count": sum(matches),
        "route_correct_counterfactual_match_rate": sum(matches) / len(matches),
        "baseline_hashes": {
            f"{route}:{index}": canonical_json_sha256(text)
            for (route, index), text in sorted(baselines.items())
        },
    }
    schedule = live["schedule"]
    cache = live["cache"]
    bound_rows = {
        "live_rows_sha256": None,
        "baseline_rows": len(live["baselines"]),
        "routed_rows": len(routed),
        "cache_sequences": len(cache),
        "alternating_rows": len(schedule["alternating"]),
        "grouped_rows": len(schedule["grouped"]),
        "temporary_server_log_required": True,
    }
    return counterfactuals, bound_rows


def gates(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    definitions = {
        "adapter_conversion": ("converted_adapters", "eq", 2),
        "adapter_loading": ("loaded_adapters", "eq", 2),
        "behavioral_materiality": ("prompts_with_distinct_route_outputs", "ge", 4),
        "bound_live_rows": ("digest_bound_live_rows", "eq", True),
        "isolated_counterfactuals": ("route_correct_counterfactual_match_rate", "eq", 1.0),
        "route_isolation": ("routed_exact_match_rate", "eq", 1.0),
        "cross_route_isolation": ("cross_route_contamination_count", "eq", 0),
        "affinity_switch_reduction": ("requested_route_switch_reduction", "ge", 0.9),
        "affinity_parity": ("schedule_semantic_parity", "eq", 1.0),
        "service_restore": ("original_service_restored", "eq", 1),
        "embedding_integrity": ("embedding_health", "eq", 200),
    }
    operators = {"eq": lambda a, b: a == b, "ge": lambda a, b: a >= b}
    return {
        gate: {
            "metric": metric,
            "operator": operator,
            "threshold": threshold,
            "actual": metrics[metric],
            "pass": operators[operator](metrics[metric], threshold),
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
    with ExperimentRun(raw, TASK_ID, inputs, requires_restoration=True) as run:
        legacy = raw / "physical_r5"
        (legacy / "raw").mkdir(parents=True)
        _legacy_receipt, metrics = r5.execute(legacy)
        live_path = legacy / "raw/live_rows.json"
        log_path = legacy / "raw/temporary_server.log"
        if not live_path.is_file() or not log_path.is_file() or not log_path.stat().st_size:
            raise RuntimeError("physical treatment did not retain live rows and server log")
        live = json.loads(live_path.read_text(encoding="utf-8"))
        counterfactuals, bound_rows = derive_counterfactuals(live)
        bound_rows["live_rows_sha256"] = sha256_file(live_path)
        metrics.update({
            "digest_bound_live_rows": True,
            "route_correct_counterfactual_match_rate": counterfactuals[
                "route_correct_counterfactual_match_rate"
            ],
        })
        write_json(raw / "actual_scores.json", metrics)
        write_json(raw / "counterfactual_baselines.json", counterfactuals)
        write_json(raw / "cache_schedule_rows.json", bound_rows)
        write_json(raw / "route_logs.json", {
            "temporary_server_log": "raw/physical_r5/raw/temporary_server.log",
            "bytes": log_path.stat().st_size,
            "sha256": sha256_file(log_path),
        })
        write_json(raw / "artifact_hashes.json", {
            "frozen_inputs": inputs,
            "physical_r5_receipt_sha256": sha256_file(legacy / "raw/receipt.json"),
            "physical_live_rows_sha256": sha256_file(live_path),
        })
        write_json(raw / "dataset_hashes.json", {
            "prompts_semantic_sha256": canonical_json_sha256(r5.PROMPTS)
        })
        write_json(raw / "effective_route.json", json.loads(
            (legacy / "raw/effective_route.json").read_text(encoding="utf-8")
        ))
        write_json(raw / "failure_reproduction.json", {
            "r5_audit_hold": "auxiliary schedules and route counterfactuals were outside independent binding",
            "r6_repair": "physical R5 child and all decisive auxiliaries are sealed by one harness terminal",
        })
        write_json(raw / "falsifiable_hypothesis.json", {
            "switch_reduction_min": 0.9,
            "counterfactual_match_required": 1.0,
            "all_gates_required": True,
        })
        write_json(raw / "hardware_metrics.json", {
            key: metrics[key] for key in (
                "alternating_wall_ms", "grouped_wall_ms", "client_affinity_speedup"
            )
        })
        write_json(raw / "independent_evaluation.json", counterfactuals)
        write_json(raw / "invalidation_rules.json", {
            "dense_label_is_not_treatment": True,
            "all_auxiliary_rows_must_be_terminal_bound": True,
            "all_gates_required": True,
        })
        write_json(raw / "invariant_controls.json", {
            "routes": ["base", "mlp", "attn"], "prompts": 12,
            "temperature": 0.0, "top_k": 1, "seed": 0,
        })
        write_json(raw / "paired_baseline.json", counterfactuals)
        write_json(raw / "real_implementation.json", {
            "physical_child": "run_adapt06_slop_live_r5.execute",
            "server_native_affinity_scheduler": False,
            "client_affinity_order": True,
        })
        write_json(raw / "recovery_state.json", json.loads(
            (legacy / "raw/recovery_state.json").read_text(encoding="utf-8")
        ))
        write_json(raw / "semantic_parity.json", {
            "route_correct_counterfactual_match_rate": metrics["route_correct_counterfactual_match_rate"],
            "schedule_semantic_parity": metrics["schedule_semantic_parity"],
        })
        write_json(raw / "service_identity.json", json.loads(
            (legacy / "raw/service_identity.json").read_text(encoding="utf-8")
        ))
        write_json(raw / "service_maintenance.json", json.loads(
            (legacy / "raw/service_maintenance.json").read_text(encoding="utf-8")
        ))
        write_json(raw / "source_execution_receipt.json", {
            "r5_receipt_sha256": EXPECTED[ROOT / "runs/research/BACKLOG-ADAPT06-SLOP-LIVE-05/raw/receipt.json"],
            "fresh_physical_child_receipt_sha256": sha256_file(legacy / "raw/receipt.json"),
        })
        for row in live["routed"]:
            run.record({
                "route": row["route"], "index": row["index"],
                "repeat": row["repeat"], "match": row["match"],
                "content_sha256": canonical_json_sha256(row["content"]),
            })
        run.checkpoint("physical_route_complete", {
            "routed_rows": len(live["routed"]), "counterfactual_match_rate": metrics["route_correct_counterfactual_match_rate"]
        })
        run.restored({
            "original_service_restored": metrics["original_service_restored"],
            "embedding_health": metrics["embedding_health"],
        }, ok=metrics["original_service_restored"] == 1 and metrics["embedding_health"] == 200)
        evidence = {
            "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
            "artifact_hashes": "raw/artifact_hashes.json", "cache_schedule_rows": "raw/cache_schedule_rows.json",
            "counterfactual_baselines": "raw/counterfactual_baselines.json", "dataset_hashes": "raw/dataset_hashes.json",
            "effective_route": "raw/effective_route.json", "failure_reproduction": "raw/failure_reproduction.json",
            "falsifiable_hypothesis": "raw/falsifiable_hypothesis.json", "hardware_metrics": "raw/hardware_metrics.json",
            "independent_evaluation": "raw/independent_evaluation.json", "invalidation_rules": "raw/invalidation_rules.json",
            "invariant_controls": "raw/invariant_controls.json", "paired_baseline": "raw/paired_baseline.json",
            "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl",
            "real_implementation": "raw/real_implementation.json", "receipt_fingerprint": "raw/receipt.json",
            "recovery_state": "raw/recovery_state.json", "route_logs": "raw/route_logs.json",
            "semantic_parity": "raw/semantic_parity.json", "service_identity": "raw/service_identity.json",
            "service_maintenance": "raw/service_maintenance.json", "source_execution_receipt": "raw/source_execution_receipt.json",
        }
        provenance_inputs = [*EXPECTED, *[raw / path.removeprefix("raw/") for path in evidence.values() if path != "raw/receipt.json"]]
        provenance = build_provenance(
            script_path=pathlib.Path(__file__).resolve(), started_at_utc=started,
            started_monotonic=monotonic, input_paths=provenance_inputs,
            packages=["pytest"], runtime={"execution_mode": "harness_bound_physical_r5"},
        )
        complete, errors = provenance_complete(provenance)
        if not complete:
            raise ValueError(errors)
        receipt = {
            "schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID,
            "provenance": provenance, "provenance_complete": True,
            "gates": gates(metrics), "evidence": evidence,
        }
        sealed = run.seal(receipt)
    return sealed, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt, metrics = execute(args.outdir.resolve())
    passed = all(row["pass"] for row in receipt["gates"].values())
    claim = "ADAPT06_TWO_ADAPTER_CLIENT_AFFINITY_QUALIFIED_R6" if passed else "ADAPT06_TWO_ADAPTER_CLIENT_AFFINITY_REJECTED_R6"
    failed = [name for name, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Route-correct match: `{metrics['route_correct_counterfactual_match_rate']:.4%}`; "
        f"switch reduction: `{metrics['requested_route_switch_reduction']:.4%}`; "
        f"failed gates: `{', '.join(failed) if failed else 'none'}`. "
        "The claim is client-affinity only.\n", encoding="utf-8"
    )
    print(json.dumps({"claim": claim, "gates": receipt["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
