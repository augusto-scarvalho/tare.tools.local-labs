#!/usr/bin/env python3
"""Trace-verbosity repeat of the SLX-03 instrumented runtime crossover."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file
from tools.research import run_slx03_gdn_fusion_runtime as r1

TASK_ID = "BACKLOG-SLX03-GDN-FUSION-RUNTIME-02"
HOST_INPUTS = {
    "config/research_backlog_admissions/BACKLOG-SLX03-GDN-FUSION-RUNTIME-02.json": "d1eb4e6a33c199d99aaaddaa4234c0aa85c5a556b5e480ba63c5d69eecfc1cb4",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-RUNTIME-02/PRE_REGISTRATION.md": "c1a5fefee51ca787a8b6358838be645967e6900a3878500bddc6c95845479357",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-RUNTIME-01/raw/receipt.json": "8261893e08281f190bd84d18970de46c52a8ddeb2ebefa8a7880a06784feb933",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-RUNTIME-01/REVIEW.json": "46aec851ee15e81b110b478a845a804b220913aeabd0b477063fdaf818c44f5d",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-INSTRUMENTED-01/raw/receipt.json": "1d8fbaeac548e83e5df3360338e3e6dedb4143fc55ee72f9902b645ee784b80b",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-INSTRUMENTED-01/REVIEW.json": "008b1321b2b01f9ddd05f5807e4e7edfec308ca68c0dcae32b0178d6f64680bf",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
}


def execute(outdir: pathlib.Path) -> dict:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    monotonic = time.monotonic()
    r1.TASK_ID = TASK_ID
    r1.HOST_INPUTS = HOST_INPUTS
    r1.SERVER_ARGS = [*r1.SERVER_ARGS, "-lv", "4"]
    r1.execute(outdir)

    raw = outdir / "raw"
    metrics_path = raw / "actual_scores.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    logs = sorted((raw / "logs").glob("*.log"))
    verbosity = {path.name: "verbosity = 4" in path.read_text(encoding="utf-8", errors="replace") for path in logs}
    metrics["trace_verbosity_verified"] = len(verbosity) == 4 and all(verbosity.values())
    r1.write_json(metrics_path, metrics)

    exact_artifacts = {}
    for name in ("samples.jsonl", "paired_metrics.json", "blocks.jsonl"):
        path = raw / name
        exact_artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    r1.write_json(raw / "end_to_end_artifact.json", {"exact_files": exact_artifacts, "hash_semantics": "raw file bytes"})
    r1.write_json(raw / "independent_evaluation.json", {"executor_metrics": metrics, "independent_review_pending": True, "claim_boundary": "runtime route and exact parity only"})

    definitions = {
        "binary_model_identity": ("binary_and_model_identity_verified", True), "trace_logging": ("trace_verbosity_verified", True),
        "treatment_identity": ("explicit_fusion_controls_verified", True), "balanced_crossover": ("valid_abba_blocks", 4),
        "request_integrity": ("successful_response_rate", 1.0), "runtime_route": ("on_blocks_with_marker", 2),
        "negative_control": ("off_blocks_without_marker", 2), "semantic_parity": ("exact_output_parity_rate", 1.0),
        "service_recovery": ("service_gateway_embedding_restored", True),
    }
    gates = {name: {"metric": metric, "operator": "eq", "threshold": threshold, "actual": metrics[metric], "pass": metrics[metric] == threshold} for name, (metric, threshold) in definitions.items()}
    _, frozen_paths = r1.verify_inputs()
    evidence_files = sorted(path for path in raw.rglob("*") if path.is_file() and path.name != "receipt.json")
    provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(), started_at_utc=started, started_monotonic=monotonic, input_paths=[*frozen_paths, *evidence_files], packages=[], runtime={"execution_mode": "instrumented_runtime_abba_trace_verbosity", "blocks": 4, "requests": metrics["recorded_requests"], "model": r1.MODEL, "verbosity": 4})
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(errors)
    evidence = {
        "acceptance_gates": "raw/receipt.json", "binary_identity": "raw/binary_identity.json", "block_logs": "raw/logs", "dependency_hashes": "raw/dependency_hashes.json",
        "effective_route": "raw/effective_route.json", "end_to_end_artifact": "raw/end_to_end_artifact.json", "environment": "raw/environment.json", "hardware_metrics": "raw/hardware_metrics.json",
        "independent_evaluation": "raw/independent_evaluation.json", "paired_baseline": "raw/paired_baseline.json", "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl",
        "receipt_fingerprint": "raw/receipt.json", "recovery_state": "raw/recovery_state.json", "service_identity": "raw/service_identity.json", "service_maintenance": "raw/service_maintenance.json", "treatment_controls": "raw/treatment_controls.json",
    }
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    r1.write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = "SLX03_GDN_FUSION_RUNTIME_ROUTE_CONFIRMED_R2" if not failed else "SLX03_GDN_FUSION_RUNTIME_ROUTE_NOT_CONFIRMED_R2"
    (outdir / "RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\nTrace verbosity `{metrics['trace_verbosity_verified']}`; ON blocks with marker `{metrics['on_blocks_with_marker']}/2`; OFF blocks without marker `{metrics['off_blocks_without_marker']}/2`; exact output parity `{metrics['exact_output_parity_rate']:.4f}` over 16 pairs; HTTP success `{metrics['successful_response_rate']:.4f}`; service restored `{metrics['service_gateway_embedding_restored']}`. Failed gates: `{', '.join(failed) if failed else 'none'}`. No performance, write-reduction or deployment claim is made.\n", encoding="utf-8", newline="\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    execute(args.outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
