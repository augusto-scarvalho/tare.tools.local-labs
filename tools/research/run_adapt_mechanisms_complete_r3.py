#!/usr/bin/env python3
"""Correct R2's seed-evidence false negative without changing its estimand."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)


TASK_ID = "BACKLOG-ADAPT-MECHANISMS-COMPLETE-03"
R2 = ROOT / "runs/research/BACKLOG-ADAPT-MECHANISMS-COMPLETE-02"
HOST_INPUTS = {
    "config/research_backlog_admissions/BACKLOG-ADAPT-MECHANISMS-COMPLETE-03.json": "a5439f086144673700d5d0ae241aefad6a2f261b59224149aca9017773149346",
    "runs/research/BACKLOG-ADAPT-MECHANISMS-COMPLETE-03/PRE_REGISTRATION.md": "1f2eea846567aadde9f8370a58169e34912f5e55cc8a46dda92791d1fd4ea2b2",
    "runs/research/BACKLOG-ADAPT-MECHANISMS-COMPLETE-02/raw/receipt.json": "fee03cc40a3bae4c7701775a44b59f9da837e6e6d752fefcc3c6e033ba2e1aa2",
    "runs/research/BACKLOG-ADAPT-MECHANISMS-COMPLETE-02/raw/samples.jsonl": "dfd74427278fff8a718d24be2f5c016ffcbb36553e622da1fcab487efc3bfdc3",
    "runs/research/BACKLOG-ADAPT-MECHANISMS-COMPLETE-02/raw/actual_scores.json": "0b860bc296ab78c9dfa27f2171055b5ee43f2e2dd13ea3a18b5f310210704737",
    "runs/research/BACKLOG-ADAPT-MECHANISMS-COMPLETE-02/raw/artifact_hashes.json": "69ec7655feae95de07c39a06c0836de36d95501b038e6179ddd0f978cbb6f280",
    "runs/research/BACKLOG-ADAPT-MECHANISMS-COMPLETE-02/raw/scorer_hashes.json": "e992db2e67ba926bdffac4f53cde0d5400594ea47e31479857f2686f0ec7eec3",
    "runs/research/BACKLOG-ADAPT-MECHANISMS-COMPLETE-02/raw/service_maintenance.json": "76b1544f7b968a65193296aded06e0853cc44e61c4398ca147d3ecb6ea8416b1",
    "runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/seed.json": "d9ad3488127806b15a2ff081e1ca898fc9945716920a49e5bfba0d9497b5d13b",
}


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()
    input_paths: list[pathlib.Path] = []
    ledger: dict[str, Any] = {}
    for relative, expected in HOST_INPUTS.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen input mismatch: {relative}: {actual} != {expected}")
        input_paths.append(path)
        ledger[relative] = {"bytes": path.stat().st_size, "sha256": actual}

    r2_receipt = read_json(R2 / "raw/receipt.json")
    unexpected_failures = [gate_id for gate_id, gate in r2_receipt["gates"].items()
                           if not gate["pass"] and gate_id != "seed_control"]
    if unexpected_failures:
        raise ValueError(f"R2 has non-seed gate failures: {unexpected_failures}")
    rows = read_jsonl(R2 / "raw/samples.jsonl")
    groups: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    keys: set[tuple[str, str, str, str]] = set()
    for row in rows:
        row_id = row["task_id"] if row["panel"] == "math" else row["id"]
        key = (row["mechanism"], row["arm"], row["panel"], row_id)
        if key in keys:
            raise ValueError(f"duplicate key: {key}")
        keys.add(key)
        groups[(row["mechanism"], row["arm"])][row["panel"]] += 1
        if row.get("stored_score_match") is not True:
            raise ValueError(f"R2 independent score mismatch: {key}")
    if len(rows) != 768 or len(groups) != 16:
        raise ValueError(f"R2 coverage mismatch: rows={len(rows)} groups={len(groups)}")
    if any(counts.get("math") != 32 or counts.get("qa") != 16 for counts in groups.values()):
        raise ValueError(f"R2 per-arm coverage mismatch: {dict(groups)}")

    r2_scores = read_json(R2 / "raw/actual_scores.json")
    seed_receipt = read_json(ROOT / "runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/seed.json")
    service = read_json(R2 / "raw/service_maintenance.json")
    artifacts = read_json(R2 / "raw/artifact_hashes.json")
    scorer_hashes = read_json(R2 / "raw/scorer_hashes.json")
    metrics = {
        "r2_sources_verified": True,
        "fresh_mechanisms_completed": r2_scores["fresh_mechanisms_completed"],
        "fresh_training_arms": r2_scores["fresh_training_arms"],
        "fresh_scored_generations": len(rows),
        "complete_arm_instances": len(groups),
        "frozen_execution_seed": seed_receipt["seed"],
        "independent_score_match": all(row["stored_score_match"] for row in rows),
        "hashed_adapter_artifacts": r2_scores["hashed_adapter_artifacts"],
        "normalized_original_service_restored": service["normalized_original_service_restored"],
        "embedding_health": r2_scores["embedding_health"],
    }
    definitions = {
        "source_integrity": ("r2_sources_verified", "eq", True),
        "mechanism_coverage": ("fresh_mechanisms_completed", "eq", 5),
        "training_coverage": ("fresh_training_arms", "eq", 12),
        "evaluation_coverage": ("fresh_scored_generations", "eq", 768),
        "arm_coverage": ("complete_arm_instances", "eq", 16),
        "seed_control": ("frozen_execution_seed", "eq", 20260827),
        "independent_aggregate": ("independent_score_match", "eq", True),
        "artifact_identity": ("hashed_adapter_artifacts", "ge", 13),
        "service_restore": ("normalized_original_service_restored", "eq", True),
        "embedding_integrity": ("embedding_health", "eq", 200),
    }
    gates: dict[str, Any] = {}
    for gate_id, (metric, operator, threshold) in definitions.items():
        actual = metrics[metric]
        passed = actual == threshold if operator == "eq" else actual >= threshold
        gates[gate_id] = {"metric": metric, "operator": operator, "threshold": threshold,
                          "actual": actual, "pass": passed}

    write_json(raw / "actual_scores.json", {**metrics, "arm_summaries": r2_scores["arm_summaries"]})
    write_json(raw / "artifact_hashes.json", artifacts)
    write_json(raw / "dataset_hashes.json", {"row_keys_sha256": canonical_json_sha256(sorted(keys)), "rows": len(rows)})
    write_json(raw / "independent_evaluation.json", {"rows": len(rows), "all_r2_score_matches": True,
               "r2_non_seed_gate_failures": unexpected_failures})
    write_json(raw / "samples.json", {"source": "BACKLOG-ADAPT-MECHANISMS-COMPLETE-02/raw/samples.jsonl",
               "sha256": sha256_file(R2 / "raw/samples.jsonl"), "rows": len(rows)})
    write_json(raw / "scorer_hashes.json", scorer_hashes)
    write_json(raw / "service_maintenance.json", service)
    write_json(raw / "source_execution_receipt.json", {"task_id": "BACKLOG-ADAPT-MECHANISMS-COMPLETE-02",
               "receipt_sha256": sha256_file(R2 / "raw/receipt.json"),
               "receipt_fingerprint": r2_receipt["receipt_fingerprint"],
               "corrected_gate": "seed_control"})
    write_json(raw / "training_trace.json", {"execution_seed_receipt": seed_receipt,
               "seed_receipt_sha256": sha256_file(ROOT / "runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/seed.json")})
    evidence = {
        "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
        "artifact_hashes": "raw/artifact_hashes.json", "dataset_hashes": "raw/dataset_hashes.json",
        "independent_evaluation": "raw/independent_evaluation.json", "provenance": "raw/receipt.json",
        "raw_samples": "raw/samples.json", "receipt_fingerprint": "raw/receipt.json",
        "scorer_hashes": "raw/scorer_hashes.json", "service_maintenance": "raw/service_maintenance.json",
        "source_execution_receipt": "raw/source_execution_receipt.json", "training_trace": "raw/training_trace.json",
    }
    evidence_files = sorted(path for path in raw.rglob("*") if path.is_file())
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(), started_at_utc=started,
        started_monotonic=mono, input_paths=[*input_paths, *evidence_files], packages=[],
        runtime={"execution_mode": "read_only_seed_evidence_correction", "corrected_gate": "seed_control",
                 "estimand_changed": False, "new_training": False, "new_inference": False},
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(f"incomplete provenance: {errors}")
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID,
               "provenance": provenance, "provenance_complete": True,
               "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = "ADAPT01_05_MECHANISMS_COMPLETED_R3" if not failed else "ADAPT01_05_MECHANISMS_MIXED_R3"
    lokr = r2_scores["arm_summaries"]["adapt01/lokr_5ep"]
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Preserved 768 rows/16 arms and rebound only the execution seed to `{seed_receipt['seed']}`. "
        f"lokr_5ep remains math `{lokr['math_correct']}/32`, QA `{lokr['qa_pass']}/16`. "
        f"Failed gates: `{', '.join(failed) if failed else 'none'}`.\n",
        encoding="utf-8", newline="\n",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
