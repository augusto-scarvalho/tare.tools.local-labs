#!/usr/bin/env python3
"""Read-only completion of the ADAPT five-mechanism matrix."""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from collections import defaultdict
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/benchmarks"))

from normal_qa_ab import grade, load_tasks
from tools.analysis.a2_stats import gsm8k_extract, numeric_equal
from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)


TASK_ID = "BACKLOG-ADAPT-MECHANISMS-COMPLETE-02"
R1 = ROOT / "runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01"
MISSING = ROOT / "runs/research/BACKLOG-ADAPT01-640-EVAL-01"
HOST_INPUTS = {
    "config/research_backlog_admissions/BACKLOG-ADAPT-MECHANISMS-COMPLETE-02.json": "2c850cc5990b0b6f86b5e3af576067748dc12bd0230e097984736ce72e5f1ff2",
    "runs/research/BACKLOG-ADAPT-MECHANISMS-COMPLETE-02/PRE_REGISTRATION.md": "19b6fff623566412de1b27eaae44201c6e8b79d3bc92c6b7435658065378fa59",
    "runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/receipt.json": "dd975197993bab7943ea2407a664f20fda927bb6fc581714eba093f7e93be0c6",
    "runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/samples.jsonl": "76089e6911d55a8cad5ba75162c898bdf6ad62e8fe561749553235f06de90bc6",
    "runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/training_trace.json": "332ccf6d14e8e45ef5c7edeb7cbff8e2d0654e1ded45509c5a8f72bd87314a48",
    "runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/artifact_hashes.json": "7d357cd1928066e45eb519825a0c89dc7d9dfed631fc3afaf1fb9b1dfb39519d",
    "runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/checkpoint_hashes.json": "a24103ed2f9ce792c410289155b957927333da37ef556d789afc11c7e4a835b6",
    "runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/service_maintenance.json": "db3b8f9f686a65543a5e268e5a6f5e5194412823911d6302ed963f42daa90dd3",
    "runs/research/BACKLOG-ADAPT01-640-EVAL-01/raw/receipt.json": "cb0180cc424db42e5a8a81fd6bf60c66b5b61c5a2c029631e909418276e8d170",
    "runs/research/BACKLOG-ADAPT01-640-EVAL-01/raw/samples.jsonl": "8b5b5a870b3a6f20ea8adadca76f093dcae9e1a725cf98a4d37e70f38caa1d33",
    "runs/research/BACKLOG-ADAPT01-640-EVAL-01/raw/service_maintenance.json": "b4915cd2cee96eb8998627ae61f019da87639bf795eea196dbeb7930d042f497",
    "tools/analysis/a2_stats.py": "d63e4c0e5fcb820d912c2492fa1e4f50c94b2488970c8fa1278c749e6b0bd459",
    "tools/benchmarks/normal_qa_ab.py": "b249c4efd4d2d52ed2da748dbaba30ceb53833e60de15fea79e4b41070d3f641",
    "runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl": "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
}


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: pathlib.Path, value: object) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalized_execstart(value: str) -> dict[str, str | None]:
    path = re.search(r"\bpath=([^ ;]+)", value)
    argv = re.search(r"\bargv\[\]=(.*?)\s+;\s+ignore_errors=", value)
    return {"path": path.group(1) if path else None, "argv": argv.group(1) if argv else None}


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

    rows = read_jsonl(R1 / "raw/samples.jsonl")
    missing_rows = read_jsonl(MISSING / "raw/samples.jsonl")
    for row in missing_rows:
        rows.append({"mechanism": "adapt01", "arm": "lokr_5ep", **row})
    qa_path = ROOT / "runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl"
    qa_tasks = {task["id"]: task for task in load_tasks(qa_path)}
    seen: set[tuple[str, str, str, str]] = set()
    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    scorer_matches = []
    for row in rows:
        row_id = row["task_id"] if row["panel"] == "math" else row["id"]
        key = (row["mechanism"], row["arm"], row["panel"], row_id)
        if key in seen:
            raise ValueError(f"duplicate sample key: {key}")
        seen.add(key)
        if row["panel"] == "math":
            rescored = bool(numeric_equal(gsm8k_extract(row["text"]), row["gold"]))
            stored = bool(row["correct"])
        else:
            rescored, _ = grade(qa_tasks[row["id"]], row["text"])
            stored = bool(row["pass"])
        scorer_matches.append(rescored == stored)
        output_row = {**row, "independent_rescore": bool(rescored), "stored_score_match": rescored == stored}
        append_jsonl(raw / "samples.jsonl", output_row)
        groups[(row["mechanism"], row["arm"])][row["panel"]].append(output_row)
    if len(rows) != 768:
        raise ValueError(f"expected 768 rows, got {len(rows)}")
    incomplete = {
        f"{mechanism}/{arm}": {panel: len(panel_rows) for panel, panel_rows in panels.items()}
        for (mechanism, arm), panels in groups.items()
        if len(panels["math"]) != 32 or len(panels["qa"]) != 16
    }
    if incomplete:
        raise ValueError(f"incomplete arms: {incomplete}")

    summaries: dict[str, Any] = {}
    for (mechanism, arm), panels in sorted(groups.items()):
        math = panels["math"]
        qa = panels["qa"]
        summaries[f"{mechanism}/{arm}"] = {
            "math_correct": sum(row["independent_rescore"] for row in math),
            "math_n": len(math), "qa_pass": sum(row["independent_rescore"] for row in qa),
            "qa_n": len(qa), "natural_eos": sum(bool(row["natural_eos"]) for row in math + qa),
        }

    training = json.loads((R1 / "raw/training_trace.json").read_text(encoding="utf-8"))
    checkpoints = json.loads((R1 / "raw/checkpoint_hashes.json").read_text(encoding="utf-8"))
    r1_service = json.loads((R1 / "raw/service_maintenance.json").read_text(encoding="utf-8"))
    missing_service = json.loads((MISSING / "raw/service_maintenance.json").read_text(encoding="utf-8"))
    before_values, after_values = r1_service["before"]["values"], r1_service["after"]["values"]
    normalized_restore = (
        normalized_execstart(before_values["ExecStart"]) == normalized_execstart(after_values["ExecStart"])
        and before_values.get("ActiveState") == after_values.get("ActiveState") == "active"
        and before_values.get("NRestarts") == after_values.get("NRestarts") == "0"
        and r1_service["before"]["health"] == r1_service["after"]["health"] == {"8080": 200, "8081": 200}
        and missing_service.get("service_untouched") is True
    )
    seeds = {arm["metrics"].get("seed") for arm in training["arms"]}
    metrics = {
        "source_receipts_and_artifacts_verified": True,
        "fresh_mechanisms_completed": len({mechanism for mechanism, _ in groups}),
        "fresh_training_arms": len(training["arms"]),
        "fresh_scored_generations": len(rows),
        "complete_arm_instances": len(groups),
        "fresh_seed_verified": 20260827 if seeds == {20260827} else None,
        "independent_score_match": all(scorer_matches),
        "hashed_adapter_artifacts": len(checkpoints),
        "normalized_original_service_restored": normalized_restore,
        "embedding_health": 200 if normalized_restore else None,
    }
    definitions = {
        "source_integrity": ("source_receipts_and_artifacts_verified", "eq", True),
        "mechanism_coverage": ("fresh_mechanisms_completed", "eq", 5),
        "training_coverage": ("fresh_training_arms", "eq", 12),
        "evaluation_coverage": ("fresh_scored_generations", "eq", 768),
        "arm_coverage": ("complete_arm_instances", "eq", 16),
        "seed_control": ("fresh_seed_verified", "eq", 20260827),
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

    r1_receipt = json.loads((R1 / "raw/receipt.json").read_text(encoding="utf-8"))
    missing_receipt = json.loads((MISSING / "raw/receipt.json").read_text(encoding="utf-8"))
    write_json(raw / "actual_scores.json", {**metrics, "arm_summaries": summaries})
    write_json(raw / "artifact_hashes.json", {"inputs": ledger, "adapter_checkpoints": checkpoints})
    write_json(raw / "dataset_hashes.json", {"combined_sample_keys_sha256": canonical_json_sha256(sorted(seen)),
               "rows": len(rows)})
    write_json(raw / "independent_evaluation.json", {"all_score_matches": all(scorer_matches),
               "matched_rows": sum(scorer_matches), "rows": len(rows), "arm_summaries": summaries})
    write_json(raw / "scorer_hashes.json", {"math": ledger["tools/analysis/a2_stats.py"],
               "qa": ledger["tools/benchmarks/normal_qa_ab.py"], "qa_tasks": ledger["runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl"]})
    write_json(raw / "service_maintenance.json", {"r1": r1_service, "missing_arm": missing_service,
               "normalized_before": normalized_execstart(before_values["ExecStart"]),
               "normalized_after": normalized_execstart(after_values["ExecStart"]),
               "normalized_original_service_restored": normalized_restore})
    write_json(raw / "source_execution_receipt.json", {
        "r1": {"sha256": sha256_file(R1 / "raw/receipt.json"), "fingerprint": r1_receipt["receipt_fingerprint"]},
        "missing_arm": {"sha256": sha256_file(MISSING / "raw/receipt.json"), "fingerprint": missing_receipt["receipt_fingerprint"]},
    })
    write_json(raw / "training_trace.json", training)
    evidence = {
        "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
        "artifact_hashes": "raw/artifact_hashes.json", "dataset_hashes": "raw/dataset_hashes.json",
        "independent_evaluation": "raw/independent_evaluation.json", "provenance": "raw/receipt.json",
        "raw_samples": "raw/samples.jsonl", "receipt_fingerprint": "raw/receipt.json",
        "scorer_hashes": "raw/scorer_hashes.json", "service_maintenance": "raw/service_maintenance.json",
        "source_execution_receipt": "raw/source_execution_receipt.json", "training_trace": "raw/training_trace.json",
    }
    evidence_files = sorted(path for path in raw.rglob("*") if path.is_file())
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(), started_at_utc=started,
        started_monotonic=mono, input_paths=[*input_paths, *evidence_files], packages=[],
        runtime={"execution_mode": "read_only_artifact_synthesis", "new_training": False,
                 "new_inference": False, "source_rows": {"r1": 720, "missing_arm": 48}},
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
    claim = "ADAPT01_05_MECHANISMS_COMPLETED_R2" if not failed else "ADAPT01_05_MECHANISMS_MIXED_R2"
    lokr = summaries["adapt01/lokr_5ep"]
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Joined 720+48 physical rows into `{len(groups)}` complete arms and `{len(rows)}` generations. "
        f"The missing lokr_5ep arm scored math `{lokr['math_correct']}/32`, QA `{lokr['qa_pass']}/16`; "
        f"normalized historical service restoration `{normalized_restore}`. Failed gates: "
        f"`{', '.join(failed) if failed else 'none'}`.\n",
        encoding="utf-8", newline="\n",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        assert normalized_execstart("{ path=/x ; argv[]=/x --a b ; ignore_errors=no ; pid=1 }") == {"path": "/x", "argv": "/x --a b"}
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
