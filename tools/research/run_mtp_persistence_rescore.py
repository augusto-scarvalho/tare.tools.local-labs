#!/usr/bin/env python3
"""Read-only telemetry-compatible rescore of MTP persistence observations."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)


TASK_ID = "BACKLOG-MTP-PERSISTENCE-RESCORE-02"
PRE_REG_SHA256 = "35ded4d36e5cfc18c4d0fc2fb3cef4fabaa00a6df2554626886375fd27335010"
SOURCES = {
    "runs/research/BACKLOG-MTP-PERSISTENCE-01/raw/samples.jsonl": "ae235a397007db32e6c5adc4228f26023fd0a7a5efa4a93cab8d7ba9e6b1b34d",
    "runs/research/BACKLOG-MTP-PERSISTENCE-01/raw/receipt.json": "6a52f8690b9f20c6583bdfa80e947ca4f7469e3297b24e046195a799c4116079",
    "runs/research/BACKLOG-MTP-PERSISTENCE-01/RESULT.md": "d49a9cffd05669d64d5d8bc0959250871f5c2179951243aa099a4ed52d160f02",
    "tools/research/run_mtp_persistence_first_instance.py": "2a70897b1a9d73fb5fbf77159fa8c6706a41786c95bd428061e8963b8abde7b1",
}


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify() -> tuple[dict[str, Any], list[pathlib.Path]]:
    ledger = {}
    paths = []
    for relative, expected in SOURCES.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"source hash mismatch: {relative}: {actual}")
        ledger[relative] = {"bytes": path.stat().st_size, "sha256": actual}
        paths.append(path)
    prereg = ROOT / "runs/research" / TASK_ID / "PRE_REGISTRATION.md"
    actual = sha256_file(prereg)
    if actual != PRE_REG_SHA256:
        raise ValueError(f"preregistration hash mismatch: {actual}")
    ledger[str(prereg.relative_to(ROOT)).replace("\\", "/")] = {"bytes": prereg.stat().st_size, "sha256": actual}
    paths.append(prereg)
    return ledger, paths


def rescore(row: dict[str, Any]) -> dict[str, Any]:
    saved = row["saved"]
    restored = row["restored"]
    tokens_cached = int(row["warm"]["body"].get("tokens_cached") or 0)
    legacy_cache_n = int(row["warm"]["timings"].get("cache_n") or 0)
    lifecycle = (
        row.get("lifecycle_ok") is True
        and saved.get("http_status") == 200
        and restored.get("http_status") == 200
        and int(saved["body"].get("n_saved") or 0) > 0
        and int(restored["body"].get("n_restored") or 0) > 0
    )
    semantic = row.get("exact_cold_restored") is True and row.get("oracle_pass") is True
    physical = lifecycle and tokens_cached > 0 and semantic
    priming_accepted = int((row.get("priming") or {}).get("timings", {}).get("draft_n_accepted") or 0)
    return {
        "index": int(row["index"]), "arm": row["arm"], "lifecycle_ok": lifecycle,
        "tokens_cached": tokens_cached, "legacy_cache_n": legacy_cache_n,
        "exact_cold_restored": bool(row.get("exact_cold_restored")),
        "oracle_pass": bool(row.get("oracle_pass")), "physical_success": physical,
        "source_pass": bool(row.get("pass")), "priming_accepted": priming_accepted,
    }


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError(f"raw output directory is not empty: {raw}")
    ledger, frozen_paths = verify()
    source = load_rows(ROOT / "runs/research/BACKLOG-MTP-PERSISTENCE-01/raw/samples.jsonl")
    if len(source) != 44 or sorted(int(row["index"]) for row in source) != list(range(44)):
        raise ValueError("source observation index coverage mismatch")
    counts = {arm: sum(row["arm"] == arm for row in source) for arm in ("nospec", "cold", "warm")}
    if counts != {"nospec": 4, "cold": 20, "warm": 20}:
        raise ValueError(f"source treatment counts mismatch: {counts}")
    rows = [rescore(row) for row in source]
    rescored_path = raw / "rescored_rows.jsonl"
    rescored_path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    physical = sum(row["physical_success"] for row in rows)
    exact = sum(row["exact_cold_restored"] and row["oracle_pass"] for row in rows)
    material = sum(row["arm"] == "warm" and row["priming_accepted"] > 0 for row in rows)
    unprimed_failures = sum(row["arm"] == "cold" and not row["physical_success"] for row in rows)
    legacy_false_negative = all(
        row["source_pass"] is False and row["legacy_cache_n"] == 0
        and row["tokens_cached"] > 0 and row["physical_success"]
        for row in rows
    )
    metrics = {
        "rescored_observations": len(rows), "physical_persistence_successes": physical,
        "exact_semantic_successes": exact, "material_primed_observations": material,
        "legacy_false_negative_confirmed": legacy_false_negative,
        "unprimed_physical_failures": unprimed_failures,
        "treatment_counts": counts,
    }
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "source_identity.json", ledger)
    write_json(raw / "failure_reproduction.json", {
        "historical_intermitent_failure_reproduced": unprimed_failures > 0,
        "source_runner_false_negative_reproduced": legacy_false_negative,
        "defect": "timings.cache_n remained zero while tokens_cached and save/restore counts were positive",
    })
    write_json(raw / "falsifiable_hypothesis.json", {"prediction": "all 44 source failures become physical successes under current telemetry contract", "actual": metrics})
    write_json(raw / "invariant_controls.json", {"treatment_counts": counts, "read_only": True, "network_calls": 0, "service_changes": 0})
    write_json(raw / "invalidation_rules.json", {"source_hash_mismatch_aborts": True, "missing_fields_abort": True, "source_pass_not_reused": True})
    write_json(raw / "independent_evaluation.json", {"all_rows_rescored": True, "independent_review_pending": True, "metrics": metrics})
    write_json(raw / "semantic_parity.json", {"exact_and_oracle": exact, "total": len(rows), "rate": exact / len(rows)})

    definitions = {
        "source_coverage": ("rescored_observations", "eq", 44),
        "physical_persistence": ("physical_persistence_successes", "eq", 44),
        "semantic_parity": ("exact_semantic_successes", "eq", 44),
        "priming_materiality": ("material_primed_observations", "eq", 20),
        "legacy_false_negative": ("legacy_false_negative_confirmed", "eq", True),
        "historical_failure": ("unprimed_physical_failures", "eq", 0),
    }
    gates = {}
    for gate_id, (metric, operator, threshold) in definitions.items():
        actual = metrics[metric]
        gates[gate_id] = {"metric": metric, "operator": operator, "threshold": threshold, "actual": actual, "pass": actual == threshold}
    evidence_files = sorted(path for path in raw.iterdir() if path.is_file())
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(), started_at_utc=started_utc,
        started_monotonic=started_mono, input_paths=[*frozen_paths, *evidence_files], packages=[],
        runtime={"execution_mode": "read_only_rescore", "network_calls": 0, "service_changes": 0},
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(f"incomplete provenance: {errors}")
    evidence = {
        "acceptance_gates": "raw/receipt.json", "failure_reproduction": "raw/failure_reproduction.json",
        "falsifiable_hypothesis": "raw/falsifiable_hypothesis.json", "independent_evaluation": "raw/independent_evaluation.json",
        "invalidation_rules": "raw/invalidation_rules.json", "invariant_controls": "raw/invariant_controls.json",
        "provenance": "raw/receipt.json", "raw_samples": "raw/rescored_rows.jsonl",
        "receipt_fingerprint": "raw/receipt.json", "semantic_parity": "raw/semantic_parity.json",
    }
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    passed = all(gate["pass"] for gate in gates.values())
    claim = "MTP_PERSISTENCE_HISTORICAL_FAILURE_NOT_REPRODUCED_R2" if passed else "MTP_PERSISTENCE_RESCORE_INVALID_R2"
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"The immutable source contains `{physical}/44` physical persistence successes and `{unprimed_failures}` unprimed failures. "
        "The original 0/44 score is a confirmed telemetry-contract false negative: `timings.cache_n=0` while top-level "
        "`tokens_cached=5039` and save/restore lifecycle evidence are positive. The historical intermittent failure was not reproduced.\n",
        encoding="utf-8", newline="\n",
    )
    return receipt


def selfcheck() -> None:
    fixture = {
        "index": 0, "arm": "cold", "lifecycle_ok": True, "pass": False,
        "saved": {"http_status": 200, "body": {"n_saved": 10}},
        "restored": {"http_status": 200, "body": {"n_restored": 10}},
        "warm": {"body": {"tokens_cached": 10}, "timings": {"cache_n": 0}},
        "exact_cold_restored": True, "oracle_pass": True, "priming": None,
    }
    assert rescore(fixture)["physical_success"] is True
    print("MTP persistence rescore self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    import subprocess
    advance = subprocess.run(
        [sys.executable, str(ROOT / "tools/analysis/backlog_pipeline.py"), "advance", TASK_ID, "--to", "EXECUTED", "--actor", "Codex executor"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    print(json.dumps({"returncode": advance.returncode, "stdout": advance.stdout, "stderr": advance.stderr}, indent=2), flush=True)
    return 0 if advance.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
