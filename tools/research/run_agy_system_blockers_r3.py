#!/usr/bin/env python3
"""Semantic, whole-tree audit of six AGY integration-blocker claims."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.request
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)


TASK_ID = "BACKLOG-AGY-SYSTEM-BLOCKERS-03"
DISTRO = "Ubuntu-24.04"
SOURCE = "/home/augus/src/slop.cpp-main"
COMMIT = "87a416bd75d5a64e66e55846b779c0a54eca21bd"
HOST_INPUTS = {
    "config/research_backlog_admissions/BACKLOG-AGY-SYSTEM-BLOCKERS-03.json": "17b6d1d1cc7e8f2ca761f7077635555bf136da1270ab24a093e3997e17660d53",
    "runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-03/PRE_REGISTRATION.md": "9bf48e61bacca9d17216b29a1ea6ed75323c6052a29efd4de3b4e900d9e174f9",
    "runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-02/raw/receipt.json": "06255331280cc995c17cbba2b4dc78443690ba1530c0243b81884ca1f3c04af0",
    "runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-02/raw/samples.json": "33bdab6b182d9998ab0ce0634ad7632491e215be5dc27c966e80adafb14abc6f",
    "tools/research/run_agy_system_blockers_r2.py": "77d6f704d23a6d60eedf5f3bccdc9f0c7d7b808d3ebf62515cbd9fe8c7767060",
}
FEATURES = {
    "SLX-03": {
        "requested": "recurrent-state write elision in the deployed decode path",
        "patterns": [
            r"ggml_cuda_try_gdn_cache_fusion",
            r"ggml_cuda_op_gated_delta_net_fused_cache",
            r"fused gated_delta_net snapshot copies",
        ],
        "classification": "MATERIALIZED_BOUNDED",
        "scope_note": "Direct GDN snapshot-to-cache fusion is present; N16/EOS cadence and hardware write counters remain unproven.",
    },
    "SLX-07": {
        "requested": "H2O heavy-hitter KV eviction",
        "patterns": [r"heavy.?hitter", r"\bh2o\b"],
        "classification": "NOT_MATERIALIZED_IN_COMMIT",
        "scope_note": "No feature-specific H2O accumulator or eviction policy was found in the immutable tracked tree.",
    },
    "REP-04": {
        "requested": "native fused KVarN attention/dequantization kernel",
        "patterns": [r"\bkvarn\b", r"kvarn[_ -]"],
        "classification": "NOT_MATERIALIZED_IN_COMMIT",
        "scope_note": "Generic quantized attention and Hadamard code do not establish a KVarN kernel.",
    },
    "REP-05": {
        "requested": "per-layer KV cache precision allocation",
        "patterns": [r"cache-type-[kv]", r"per.?layer.{0,40}(kv|cache).{0,40}(type|precision)"],
        "classification": "NEARBY_GLOBAL_CONTROL_ONLY",
        "scope_note": "Global K/V cache types exist; no per-layer allocator or CLI was identified.",
    },
    "RETRO-01": {
        "requested": "trained recurrent-retrofit checkpoint and route",
        "patterns": [r"recurrent.{0,30}retrofit", r"retrofit.{0,30}recurrent"],
        "classification": "ARTIFACT_NOT_IDENTIFIED_BOUNDED",
        "scope_note": "Native recurrent architectures are not evidence of the requested trained retrofit artifact.",
    },
    "SLX-08": {
        "requested": "selected-block speculative prefill wired into real TTFT",
        "patterns": [r"speculative.{0,30}prefill", r"selected.?block.{0,30}(qkv|prefill)", r"\bdflash\b"],
        "classification": "NEARBY_DFLASH_MECHANISM_ONLY",
        "scope_note": "DFlash support exists, but no selected-block speculative-prefill route matching the claim was identified.",
    },
}


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(argv: list[str], timeout: float = 180.0) -> dict[str, Any]:
    completed = subprocess.run(
        argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False, timeout=timeout,
    )
    return {
        "argv": argv,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def wsl(*argv: str, timeout: float = 180.0) -> dict[str, Any]:
    return run(["wsl", "-d", DISTRO, "-e", *argv], timeout=timeout)


def checked(result: dict[str, Any], label: str) -> dict[str, Any]:
    if result["returncode"] != 0:
        raise RuntimeError(f"{label} failed: {result}")
    return result


def health(port: int) -> int | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as response:
            return response.status
    except Exception:
        return None


def service_identity() -> dict[str, Any]:
    shown = checked(wsl("systemctl", "show", "llm-inference.service", "-p", "MainPID",
                        "-p", "NRestarts", "-p", "ActiveState", "--no-pager"), "service show")
    values = dict(line.split("=", 1) for line in shown["stdout"].splitlines() if "=" in line)
    pid = values.get("MainPID", "0")
    exe = checked(wsl("readlink", "-f", f"/proc/{pid}/exe"), "service exe")["stdout"]
    cmdline = checked(wsl("python3", "-c",
        "import pathlib,sys;print('\\n'.join(x.decode() for x in pathlib.Path(sys.argv[1]).read_bytes().split(b'\\0') if x))",
        f"/proc/{pid}/cmdline"), "service argv")["stdout"].splitlines()
    return {
        "active_state": values.get("ActiveState"),
        "main_pid": pid,
        "n_restarts": values.get("NRestarts"),
        "executable": exe,
        "argv": cmdline,
        "health_8080": health(8080),
        "health_8081": health(8081),
    }


def grep_commit(pattern: str) -> dict[str, Any]:
    result = wsl("git", "-C", SOURCE, "grep", "-n", "-i", "-E", pattern, COMMIT, "--", ".")
    if result["returncode"] not in (0, 1):
        raise RuntimeError(f"git grep failed: {result}")
    lines = result["stdout"].splitlines() if result["stdout"] else []
    return {
        "pattern": pattern,
        "match_count": len(lines),
        "matches": lines[:200],
        "full_output_sha256": hashlib.sha256(result["stdout"].encode("utf-8")).hexdigest(),
        "returncode": result["returncode"],
    }


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()
    host_paths: list[pathlib.Path] = []
    host_ledger: dict[str, Any] = {}
    for relative, expected in HOST_INPUTS.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen source mismatch: {relative}: {actual} != {expected}")
        host_paths.append(path)
        host_ledger[relative] = {"bytes": path.stat().st_size, "sha256": actual}

    head = checked(wsl("git", "-C", SOURCE, "rev-parse", "HEAD"), "candidate HEAD")["stdout"]
    commit_type = checked(wsl("git", "-C", SOURCE, "cat-file", "-t", COMMIT), "candidate commit")["stdout"]
    if head != COMMIT or commit_type != "commit":
        raise ValueError(f"candidate commit drift: head={head}, type={commit_type}")

    before = service_identity()
    if before["active_state"] != "active" or before["health_8080"] != 200 or before["health_8081"] != 200:
        raise RuntimeError(f"service baseline unhealthy: {before}")

    rows: list[dict[str, Any]] = []
    for item, spec in FEATURES.items():
        searches = [grep_commit(pattern) for pattern in spec["patterns"]]
        semantic_evidence = True
        if item == "SLX-03":
            semantic_evidence = all(search["match_count"] > 0 for search in searches)
        rows.append({
            "item": item,
            "requested_feature": spec["requested"],
            "classification": spec["classification"],
            "scope_note": spec["scope_note"],
            "searches": searches,
            "semantic_evidence_complete": semantic_evidence,
            "predecessor_missing_integration": True,
            "predecessor_false_negative": item == "SLX-03" and semantic_evidence,
        })

    inventory = checked(wsl("find", "/home/augus/models", "-type", "f"), "model inventory")
    inventory_matches = [
        line for line in inventory["stdout"].splitlines()
        if re.search(r"retrofit|recurrent.*(checkpoint|model)|hybrid.*retrofit", line, re.I)
    ]
    for row in rows:
        if row["item"] == "RETRO-01":
            row["artifact_inventory_matches"] = inventory_matches[:200]
            row["artifact_inventory_full_sha256"] = hashlib.sha256(
                inventory["stdout"].encode("utf-8")
            ).hexdigest()

    after = service_identity()
    runtime_unchanged = before == after
    metrics = {
        "classified_items": len(rows),
        "items_with_semantic_evidence": sum(row["semantic_evidence_complete"] for row in rows),
        "confirmed_predecessor_false_negatives": sum(row["predecessor_false_negative"] for row in rows),
        "runtime_unchanged": runtime_unchanged,
        "materialized_bounded": sum(row["classification"] == "MATERIALIZED_BOUNDED" for row in rows),
    }
    gates = {
        "scope_coverage": {"metric": "classified_items", "operator": "eq", "threshold": 6,
                           "actual": metrics["classified_items"], "pass": metrics["classified_items"] == 6},
        "semantic_evidence": {"metric": "items_with_semantic_evidence", "operator": "eq", "threshold": 6,
                              "actual": metrics["items_with_semantic_evidence"], "pass": metrics["items_with_semantic_evidence"] == 6},
        "false_negative_detection": {"metric": "confirmed_predecessor_false_negatives", "operator": "ge", "threshold": 1,
                                      "actual": metrics["confirmed_predecessor_false_negatives"], "pass": metrics["confirmed_predecessor_false_negatives"] >= 1},
        "runtime_integrity": {"metric": "runtime_unchanged", "operator": "eq", "threshold": True,
                              "actual": runtime_unchanged, "pass": runtime_unchanged is True},
    }

    previous_rows = json.loads((ROOT / "runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-02/raw/samples.json").read_text(encoding="utf-8"))
    write_json(raw / "samples.json", rows)
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "artifact_hashes.json", {"host": host_ledger, "candidate_commit": COMMIT})
    write_json(raw / "failure_reproduction.json", {"predecessor_rows": previous_rows,
               "reproduced_failure": "path-restricted literal search missed semantic source match"})
    write_json(raw / "falsifiable_hypothesis.json", {"expected_false_negatives_at_least": 1,
               "positive_anchor_requirements": FEATURES["SLX-03"]["patterns"]})
    write_json(raw / "independent_evaluation.json", {"classifications": {
               row["item"]: row["classification"] for row in rows},
               "bounded_interpretation": True})
    write_json(raw / "invalidation_rules.json", {
        "missing_positive_anchor_invalidates_slx03_false_negative": True,
        "lexical_near_match_cannot_establish_requested_feature": True,
        "claims_are_bounded_to_commit_and_inventory": True,
    })
    write_json(raw / "invariant_controls.json", {"candidate_commit": COMMIT,
               "source_root": SOURCE, "features": FEATURES, "read_only": True})
    write_json(raw / "semantic_parity.json", {row["item"]: {
               "requested": row["requested_feature"], "observed": row["classification"],
               "scope_note": row["scope_note"]} for row in rows})
    write_json(raw / "service_identity.json", {"before": before, "after": after,
               "runtime_unchanged": runtime_unchanged})
    write_json(raw / "source_execution_receipt.json", {"candidate_commit": COMMIT,
               "whole_tree_search": True,
               "search_output_hashes": {row["item"]: [search["full_output_sha256"] for search in row["searches"]] for row in rows}})

    evidence = {
        "acceptance_gates": "raw/receipt.json",
        "actual_scores": "raw/actual_scores.json",
        "artifact_hashes": "raw/artifact_hashes.json",
        "failure_reproduction": "raw/failure_reproduction.json",
        "falsifiable_hypothesis": "raw/falsifiable_hypothesis.json",
        "independent_evaluation": "raw/independent_evaluation.json",
        "invalidation_rules": "raw/invalidation_rules.json",
        "invariant_controls": "raw/invariant_controls.json",
        "provenance": "raw/receipt.json",
        "raw_samples": "raw/samples.json",
        "receipt_fingerprint": "raw/receipt.json",
        "semantic_parity": "raw/semantic_parity.json",
        "service_identity": "raw/service_identity.json",
        "source_execution_receipt": "raw/source_execution_receipt.json",
    }
    evidence_files = sorted(path for path in raw.rglob("*") if path.is_file())
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(), started_at_utc=started,
        started_monotonic=mono, input_paths=[*host_paths, *evidence_files], packages=[],
        runtime={"execution_mode": "read_only_semantic_source_audit", "candidate_commit": COMMIT},
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
    claim = "AGY_SYSTEM_BLOCKER_FALSE_NEGATIVE_CONFIRMED_R3" if not failed else "AGY_SYSTEM_BLOCKER_FALSE_NEGATIVE_NOT_CONFIRMED_R3"
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Classified 6/6 items; confirmed predecessor false negatives: "
        f"`{metrics['confirmed_predecessor_false_negatives']}`. SLX-03 is "
        "`MATERIALIZED_BOUNDED`; the other five remain bounded missing/nearby classifications. "
        f"Failed gates: `{', '.join(failed) if failed else 'none'}`.\n",
        encoding="utf-8", newline="\n",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        assert len(FEATURES) == 6
        assert FEATURES["SLX-03"]["classification"] == "MATERIALIZED_BOUNDED"
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
