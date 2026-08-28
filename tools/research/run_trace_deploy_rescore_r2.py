#!/usr/bin/env python3
"""Fixture-validated rescore of the frozen trace deployment finalist outputs."""
from __future__ import annotations

import argparse
import inspect
import json
import pathlib
import random
import statistics
import sys
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.a2_stats import numeric_equal
from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)
from tools.analysis.final_numeric_answer import (
    extract_final_numeric,
    extract_final_numeric_for_question,
)


TASK_ID = "BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-02"
SOURCE = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01"
BOOTSTRAP_SEED = 2026082711
BOOTSTRAP_REPLICATES = 20_000
HOST_INPUTS = {
    "config/research_backlog_admissions/BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-02.json": "6fc64b6edaa63656509d795e83d732603cd02f996b28cb1ba984d9165a87550f",
    "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-02/PRE_REGISTRATION.md": "d1913954b36609a5e1089dd8ee79c10af981ebb3c94b12270c6b407c2e0e8910",
    "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/receipt.json": "b4fc924a1542e4913c3c1d70fdf77f8bb9be0e2662b8757d0d06f82b60d3f521",
    "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/student_samples.json": "288270e4faa780bbd905b593193bf9c9edc595d84bf41cc2ef3fd72ba53663c9",
    "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/actual_scores.json": "c2ba817d9919d7c58e4d9ca33f6ce3105c9f25e3ccf105e96d94631944f3a18a",
    "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/dataset_hashes.json": "f3bd82ee0aef9b7eb7669d7ed6bc5549412c8a5ebdec33bb4496bb869c95661c",
    "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/checkpoint_hashes.json": "d13a971c7e27509142dfd0f15a61574bc5f7ebf7f32742b2df008643987014c2",
    "tools/analysis/final_numeric_answer.py": "16fa779e53b6ceaa9a26507aa42c780171ba748003ffe9c8049bab310ae47905",
    "tests/fixtures/final_numeric_answer_cases.json": "414c86c25be9b49483475ea784297d9e6c693903fb0843d01fcbf40bbc595975",
    "tests/test_final_numeric_answer.py": "c1f59bbe0782d6395fbf9d0c282e02a6a00ab3bb4254161ddec66aea283272e4",
}


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: pathlib.Path, value: object) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def paired_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {
        arm: {row["task_id"]: int(row["rescored_correct"]) for row in rows if row["arm"] == arm}
        for arm in ("answer_only", "full_trace")
    }
    if len(by_arm["answer_only"]) != 256 or set(by_arm["answer_only"]) != set(by_arm["full_trace"]):
        raise ValueError("paired coverage is incomplete")
    task_ids = sorted(by_arm["answer_only"])
    differences = [by_arm["full_trace"][task] - by_arm["answer_only"][task] for task in task_ids]
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = [
        sum(differences[rng.randrange(256)] for _ in range(256)) / 256
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    estimates.sort()
    return {
        "point": statistics.mean(differences),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "lower_95": estimates[int(0.025 * BOOTSTRAP_REPLICATES)],
        "upper_95": estimates[int(0.975 * BOOTSTRAP_REPLICATES)],
        "trace_only_correct": sum(value == 1 for value in differences),
        "answer_only_correct": sum(value == -1 for value in differences),
    }


def validate_fixtures() -> dict[str, Any]:
    cases = json.loads((ROOT / "tests/fixtures/final_numeric_answer_cases.json").read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        extraction = (
            extract_final_numeric_for_question(case["question"], case["text"])
            if case.get("question") else extract_final_numeric(case["text"])
        )
        passed = extraction.value == case["expected"] and extraction.method == case["method"]
        rows.append({"id": case["id"], "expected": case["expected"],
                     "actual": extraction.value, "method": extraction.method, "pass": passed})
    return {"cases": len(rows), "passed": sum(row["pass"] for row in rows), "rows": rows}


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

    fixture = validate_fixtures()
    if fixture["passed"] != fixture["cases"]:
        raise RuntimeError(f"external fixture validation failed: {fixture}")
    source_scores = json.loads((SOURCE / "raw/actual_scores.json").read_text(encoding="utf-8"))
    source_samples = json.loads((SOURCE / "raw/student_samples.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for arm_record in source_samples:
        arm = arm_record["arm"]
        for sample in arm_record["math_samples"]:
            key = (arm, sample["task_id"])
            if key in seen:
                raise ValueError(f"duplicate arm/task: {key}")
            seen.add(key)
            extraction = extract_final_numeric_for_question(sample["prompt"], sample["output_text"])
            correct = bool(numeric_equal(extraction.value, sample["gold"]))
            row = {
                "arm": arm,
                "task_id": sample["task_id"],
                "gold": sample["gold"],
                "original_extracted": sample["extracted"],
                "original_correct": bool(sample["correct"]),
                "rescored_extracted": extraction.value,
                "rescored_method": extraction.method,
                "rescored_correct": correct,
                "natural_eos": bool(sample["natural_eos"]),
                "new_tokens": sample["new_tokens"],
            }
            rows.append(row)
            append_jsonl(raw / "rescored_samples.jsonl", row)
    if len(rows) != 512:
        raise ValueError(f"expected 512 rows, got {len(rows)}")

    comparison = paired_bootstrap(rows)
    answer_correct = sum(row["rescored_correct"] for row in rows if row["arm"] == "answer_only")
    trace_correct = sum(row["rescored_correct"] for row in rows if row["arm"] == "full_trace")
    signature = list(inspect.signature(extract_final_numeric_for_question).parameters)
    scorer_blind = signature == ["question", "text"]
    metrics = {
        "frozen_deployment_sources_verified": True,
        "external_fixture_cases": fixture["cases"],
        "external_fixture_pass_rate": fixture["passed"] / fixture["cases"],
        "selected_seed": source_scores["selected_seed"],
        "third_panel_disjoint_from_training_and_prior_panels": source_scores["third_panel_disjoint_from_training_and_prior_panels"],
        "rescored_generations": len(rows),
        "answer_math_correct": answer_correct,
        "trace_math_correct": trace_correct,
        "answer_accuracy": answer_correct / 256,
        "trace_third_panel_accuracy": trace_correct / 256,
        "trace_minus_answer": (trace_correct - answer_correct) / 256,
        "paired_bootstrap": comparison,
        "paired_bootstrap_95ci_lower_trace_minus_answer": comparison["lower_95"],
        "imported_selected_seed_qa_regression": source_scores["imported_selected_seed_qa_regression"],
        "scorer_does_not_receive_gold": scorer_blind,
        "changed_labels": sum(row["original_correct"] != row["rescored_correct"] for row in rows),
    }
    definitions = {
        "source_integrity": ("frozen_deployment_sources_verified", "eq", True),
        "fixture_validation": ("external_fixture_pass_rate", "eq", 1.0),
        "fixture_coverage": ("external_fixture_cases", "ge", 18),
        "selection_reproducibility": ("selected_seed", "eq", 20260832),
        "panel_isolation": ("third_panel_disjoint_from_training_and_prior_panels", "eq", True),
        "evaluation_coverage": ("rescored_generations", "eq", 512),
        "finalist_gain": ("paired_bootstrap_95ci_lower_trace_minus_answer", "gt", 0.0),
        "finalist_absolute": ("trace_third_panel_accuracy", "ge", 0.40),
        "protected_retention": ("imported_selected_seed_qa_regression", "le", 0.05),
        "scorer_blinding": ("scorer_does_not_receive_gold", "eq", True),
    }
    gates: dict[str, Any] = {}
    for gate_id, (metric, operator, threshold) in definitions.items():
        actual = metrics[metric]
        passed = actual == threshold if operator == "eq" else actual >= threshold if operator == "ge" else actual > threshold if operator == "gt" else actual <= threshold
        gates[gate_id] = {"metric": metric, "operator": operator, "threshold": threshold,
                          "actual": actual, "pass": passed}

    old_receipt = json.loads((SOURCE / "raw/receipt.json").read_text(encoding="utf-8"))
    checkpoint_hashes = json.loads((SOURCE / "raw/checkpoint_hashes.json").read_text(encoding="utf-8"))
    dataset_hashes = json.loads((SOURCE / "raw/dataset_hashes.json").read_text(encoding="utf-8"))
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "checkpoint_hashes.json", checkpoint_hashes)
    write_json(raw / "dataset_hashes.json", dataset_hashes)
    write_json(raw / "external_fixture_validation.json", fixture)
    write_json(raw / "independent_evaluation.json", {"comparison": comparison,
               "answer_correct": answer_correct, "trace_correct": trace_correct,
               "label_changes": metrics["changed_labels"]})
    write_json(raw / "model_hash.json", checkpoint_hashes)
    write_json(raw / "paired_baseline.json", {"baseline": "answer_only", "treatment": "full_trace",
               "paired_tasks": 256, "comparison": comparison})
    write_json(raw / "scorer_hashes.json", {"implementation": ledger["tools/analysis/final_numeric_answer.py"],
               "fixtures": ledger["tests/fixtures/final_numeric_answer_cases.json"],
               "signature": signature, "gold_argument_absent": scorer_blind})
    write_json(raw / "source_execution_receipt.json", {"task_id": "BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01",
               "receipt_sha256": sha256_file(SOURCE / "raw/receipt.json"),
               "receipt_fingerprint": old_receipt["receipt_fingerprint"]})
    write_json(raw / "student_samples.json", {"source": "raw/rescored_samples.jsonl", "rows": len(rows)})
    write_json(raw / "teacher_samples.json", {"teacher_rows_used": 0, "rescore_only": True})

    evidence = {
        "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
        "checkpoint_hashes": "raw/checkpoint_hashes.json", "dataset_hashes": "raw/dataset_hashes.json",
        "external_fixture_validation": "raw/external_fixture_validation.json",
        "independent_evaluation": "raw/independent_evaluation.json", "model_hash": "raw/model_hash.json",
        "paired_baseline": "raw/paired_baseline.json", "provenance": "raw/receipt.json",
        "raw_samples": "raw/rescored_samples.jsonl", "receipt_fingerprint": "raw/receipt.json",
        "scorer_hashes": "raw/scorer_hashes.json", "source_execution_receipt": "raw/source_execution_receipt.json",
        "student_samples": "raw/student_samples.json", "teacher_samples": "raw/teacher_samples.json",
    }
    evidence_files = sorted(path for path in raw.rglob("*") if path.is_file())
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(), started_at_utc=started,
        started_monotonic=mono, input_paths=[*input_paths, *evidence_files], packages=[],
        runtime={"execution_mode": "offline_frozen_output_rescore", "new_inference": False,
                 "new_training": False, "seed_reselection": False, "fourth_panel": False},
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
    claim = "TRACE_DISTILLATION_DEPLOYMENT_FINALIST_RESCORED_R2" if not failed else "TRACE_DISTILLATION_DEPLOYMENT_FINALIST_NOT_CONFIRMED_R2"
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Answer-only/trace rescored `{answer_correct}/256` and `{trace_correct}/256`; "
        f"delta `{metrics['trace_minus_answer']:.6f}`, paired-bootstrap 95% interval "
        f"`[{comparison['lower_95']:.6f}, {comparison['upper_95']:.6f}]`. "
        f"External fixtures `{fixture['passed']}/{fixture['cases']}`. "
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
        fixture = validate_fixtures()
        assert fixture["cases"] >= 18 and fixture["passed"] == fixture["cases"]
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
