#!/usr/bin/env python3
"""PID-bound startup and request-journal repeat of the SLX-03 runtime crossover."""
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
from tools.research import run_slx03_gdn_fusion_runtime_r2 as r2

TASK_ID = "BACKLOG-SLX03-GDN-FUSION-RUNTIME-03"
HOST_INPUTS = {
    "config/research_backlog_admissions/BACKLOG-SLX03-GDN-FUSION-RUNTIME-03.json": "90a582505bd80d9c5b4126792f2a53a54b46b2ef886272635f3a8dd8597ac0c7",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-RUNTIME-03/PRE_REGISTRATION.md": "92e7c6c32d15651f740da79a5f2607413f6e1b589b5f31eb9a877c8da8ef5364",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-RUNTIME-02/raw/receipt.json": "d9c409805a4dc09f1c437a4452004f01f2bb28a4934b28d8a18aec09d2d29716",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-RUNTIME-02/REVIEW.json": "fe7bee68bb88c13da11e323b4f62f18bbf359b7cbb91f36b9b62bbe399a4adba",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-INSTRUMENTED-01/raw/receipt.json": "1d8fbaeac548e83e5df3360338e3e6dedb4143fc55ee72f9902b645ee784b80b",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-INSTRUMENTED-01/REVIEW.json": "008b1321b2b01f9ddd05f5807e4e7edfec308ca68c0dcae32b0178d6f64680bf",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
}
STARTUP_DIR: pathlib.Path


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8", newline="\n")


def execute(outdir: pathlib.Path) -> dict:
    global STARTUP_DIR
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    monotonic = time.monotonic()
    raw = outdir / "raw"
    STARTUP_DIR = raw / "startup_logs"
    STARTUP_DIR.mkdir(parents=True, exist_ok=True)
    original_start = r1.start_block

    def start_with_pid_bound_startup(block: dict):
        unit, launch = original_start(block)
        pid = int(launch["state"]["main_pid"])
        startup = r1.infra.wsl("journalctl", f"_PID={pid}", "--no-pager", "-o", "short-iso", root=True, timeout=180)
        path = STARTUP_DIR / f"{block['id']}.log"
        path.write_text(startup["stdout"], encoding="utf-8", newline="\n")
        verified = startup["returncode"] == 0 and "verbosity = 4" in startup["stdout"]
        launch["startup_trace"] = {"pid": pid, "path": str(path.relative_to(outdir)).replace("\\", "/"), "sha256": sha256_file(path), "verbosity_4": verified}
        if not verified:
            r1.infra.wsl("systemctl", "stop", unit, root=True, timeout=180)
            raise RuntimeError(f"PID-bound startup verbosity missing for {block['id']} pid={pid}")
        return unit, launch

    r1.start_block = start_with_pid_bound_startup
    r2.TASK_ID = TASK_ID
    r2.HOST_INPUTS = HOST_INPUTS
    r2.execute(outdir)

    blocks_path = raw / "blocks.jsonl"
    blocks = r1.read_jsonl(blocks_path)
    request_dir = raw / "request_logs"
    request_dir.mkdir(parents=True, exist_ok=True)
    for block in blocks:
        pid = int(block["launch"]["state"]["main_pid"])
        request = r1.infra.wsl("journalctl", f"_PID={pid}", "--no-pager", "-o", "short-iso", root=True, timeout=300)
        path = request_dir / f"{block['block_id']}.log"
        path.write_text(request["stdout"], encoding="utf-8", newline="\n")
        block["pid_bound_request_log"] = {"pid": pid, "path": str(path.relative_to(outdir)).replace("\\", "/"), "sha256": sha256_file(path), "returncode": request["returncode"]}
        block["marker_count"] = request["stdout"].count(r1.MARKER)
    write_jsonl(blocks_path, blocks)

    metrics_path = raw / "actual_scores.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.pop("trace_verbosity_verified", None)
    metrics["pid_bound_startup_verbosity_verified"] = len(blocks) == 4 and all(block["launch"]["startup_trace"]["verbosity_4"] for block in blocks)
    metrics["on_blocks_with_marker"] = sum(block["arm"] == "on" and block["marker_count"] > 0 for block in blocks)
    metrics["off_blocks_without_marker"] = sum(block["arm"] == "off" and block["marker_count"] == 0 for block in blocks)
    r1.write_json(metrics_path, metrics)
    r1.write_json(raw / "effective_route.json", {"blocks": blocks, "order": [block["arm"] for block in blocks], "marker": r1.MARKER, "journal_scope": "current PID, uncapped"})
    r1.write_json(raw / "hardware_metrics.json", {"per_block_gpu": [{"block_id": block["block_id"], "gpu": block["gpu"], "marker_count": block["marker_count"]} for block in blocks], "performance_claimed": False})
    r1.write_json(raw / "independent_evaluation.json", {"executor_metrics": metrics, "independent_review_pending": True, "claim_boundary": "runtime route and exact parity only"})

    exact = {}
    artifact_paths = [raw / "samples.jsonl", raw / "paired_metrics.json", blocks_path, *sorted(STARTUP_DIR.glob("*.log")), *sorted(request_dir.glob("*.log"))]
    for path in artifact_paths:
        relative = str(path.relative_to(raw)).replace("\\", "/")
        exact[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    r1.write_json(raw / "end_to_end_artifact.json", {"exact_files": exact, "hash_semantics": "raw file bytes"})

    definitions = {
        "binary_model_identity": ("binary_and_model_identity_verified", True), "trace_logging": ("pid_bound_startup_verbosity_verified", True),
        "treatment_identity": ("explicit_fusion_controls_verified", True), "balanced_crossover": ("valid_abba_blocks", 4),
        "request_integrity": ("successful_response_rate", 1.0), "runtime_route": ("on_blocks_with_marker", 2),
        "negative_control": ("off_blocks_without_marker", 2), "semantic_parity": ("exact_output_parity_rate", 1.0),
        "service_recovery": ("service_gateway_embedding_restored", True),
    }
    gates = {name: {"metric": metric, "operator": "eq", "threshold": threshold, "actual": metrics[metric], "pass": metrics[metric] == threshold} for name, (metric, threshold) in definitions.items()}
    _, frozen_paths = r1.verify_inputs()
    evidence_files = sorted(path for path in raw.rglob("*") if path.is_file() and path.name != "receipt.json")
    provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(), started_at_utc=started, started_monotonic=monotonic, input_paths=[*frozen_paths, *evidence_files], packages=[], runtime={"execution_mode": "instrumented_runtime_abba_pid_bound", "blocks": 4, "requests": metrics["recorded_requests"], "model": r1.MODEL, "verbosity": 4})
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(errors)
    evidence = {
        "acceptance_gates": "raw/receipt.json", "binary_identity": "raw/binary_identity.json", "block_logs": "raw/request_logs", "dependency_hashes": "raw/dependency_hashes.json",
        "effective_route": "raw/effective_route.json", "end_to_end_artifact": "raw/end_to_end_artifact.json", "environment": "raw/environment.json", "hardware_metrics": "raw/hardware_metrics.json",
        "independent_evaluation": "raw/independent_evaluation.json", "paired_baseline": "raw/paired_baseline.json", "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl",
        "receipt_fingerprint": "raw/receipt.json", "recovery_state": "raw/recovery_state.json", "service_identity": "raw/service_identity.json", "service_maintenance": "raw/service_maintenance.json", "treatment_controls": "raw/treatment_controls.json",
    }
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    r1.write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = "SLX03_GDN_FUSION_RUNTIME_ROUTE_CONFIRMED_R3" if not failed else "SLX03_GDN_FUSION_RUNTIME_ROUTE_NOT_CONFIRMED_R3"
    (outdir / "RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\nPID-bound startup verbosity `{metrics['pid_bound_startup_verbosity_verified']}`; ON markers `{metrics['on_blocks_with_marker']}/2`; OFF without markers `{metrics['off_blocks_without_marker']}/2`; exact parity `{metrics['exact_output_parity_rate']:.4f}` over 16 pairs; HTTP `{metrics['successful_response_rate']:.4f}`; service restored `{metrics['service_gateway_embedding_restored']}`. Failed gates: `{', '.join(failed) if failed else 'none'}`. No performance, write-reduction or deployment claim is made.\n", encoding="utf-8", newline="\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    execute(args.outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
