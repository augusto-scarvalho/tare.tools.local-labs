#!/usr/bin/env python3
"""Conservative offline rescore of the frozen trace-deployment outputs."""
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
    build_provenance, canonical_json_sha256, provenance_complete, sha256_file,
)
from tools.analysis.final_numeric_answer_v2 import extract_concluded_numeric_for_question

TASK_ID = "BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-03"
SOURCE = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01"
BOOTSTRAP_SEED = 2026082811
REPLICATES = 20_000
HOST_INPUTS = {
    "config/research_backlog_admissions/BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-03.json": "7edf7413d9c4e1adb55fde84b6060084f9252c09a35b186e0236310d647a6dda",
    "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-03/PRE_REGISTRATION.md": "b6f424169219780f13a2983ab328400b403553ffc043c058a4815d9a3f45816d",
    "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/receipt.json": "b4fc924a1542e4913c3c1d70fdf77f8bb9be0e2662b8757d0d06f82b60d3f521",
    "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/student_samples.json": "288270e4faa780bbd905b593193bf9c9edc595d84bf41cc2ef3fd72ba53663c9",
    "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/actual_scores.json": "c2ba817d9919d7c58e4d9ca33f6ce3105c9f25e3ccf105e96d94631944f3a18a",
    "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-02/REVIEW.json": "a0347bcbc61e9c64fc660170732eaf684edec8fbd0e3b6b8d5e010242f1da890",
    "tools/analysis/final_numeric_answer_v2.py": "51bb1bb3af967eb1d6638f2816d89d90c7251f533f17274ac33efd3e25f7cf35",
    "tests/fixtures/final_numeric_answer_v2_cases.json": "5e6494486f9caec3b40e072896e7b574911f188d4f11db4d6d90056565641184",
    "tests/test_final_numeric_answer_v2.py": "876ff43e96f6320f362e50abef83a877bd43ce2d1c1e97171786a9105c50683a",
}


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def paired_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = {arm: {r["task_id"]: int(r["rescored_correct"]) for r in rows if r["arm"] == arm}
            for arm in ("answer_only", "full_trace")}
    if len(arms["answer_only"]) != 256 or set(arms["answer_only"]) != set(arms["full_trace"]):
        raise ValueError("paired coverage is incomplete")
    diffs = [arms["full_trace"][task] - arms["answer_only"][task] for task in sorted(arms["answer_only"])]
    rng = random.Random(BOOTSTRAP_SEED)
    samples = sorted(sum(diffs[rng.randrange(256)] for _ in range(256)) / 256 for _ in range(REPLICATES))
    return {"point": statistics.mean(diffs), "replicates": REPLICATES, "seed": BOOTSTRAP_SEED,
            "lower_95": samples[int(.025 * REPLICATES)], "upper_95": samples[int(.975 * REPLICATES)],
            "trace_only_correct": sum(x == 1 for x in diffs), "answer_only_correct": sum(x == -1 for x in diffs)}


def validate_fixtures() -> dict[str, Any]:
    cases = json.loads((ROOT / "tests/fixtures/final_numeric_answer_v2_cases.json").read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        got = extract_concluded_numeric_for_question(case["question"], case["text"])
        rows.append({"id": case["id"], "expected": case["expected"], "actual": got.value,
                     "method": got.method, "pass": got.value == case["expected"] and got.method == case["method"]})
    return {"cases": len(rows), "passed": sum(r["pass"] for r in rows), "rows": rows}


def retained_regressions(samples: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {
        ("answer_only", "gsm8k/665"): "15", ("answer_only", "gsm8k/711"): "4",
        ("answer_only", "gsm8k/838"): "15", ("full_trace", "gsm8k/665"): "360",
        ("full_trace", "gsm8k/711"): "1", ("full_trace", "gsm8k/838"): "0",
    }
    rows = []
    for arm in samples:
        for sample in arm["math_samples"]:
            key = (arm["arm"], sample["task_id"])
            if key in expected:
                got = extract_concluded_numeric_for_question(sample["prompt"], sample["output_text"])
                rows.append({"arm": key[0], "task_id": key[1], "expected": expected[key],
                             "actual": got.value, "pass": got.value == expected[key]})
    return {"cases": len(rows), "passed": sum(r["pass"] for r in rows), "rows": rows}


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    started, mono = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), time.monotonic()
    ledger, inputs = {}, []
    for relative, expected in HOST_INPUTS.items():
        path, actual = ROOT / relative, sha256_file(ROOT / relative)
        if actual != expected:
            raise ValueError(f"frozen input mismatch: {relative}: {actual} != {expected}")
        ledger[relative] = {"bytes": path.stat().st_size, "sha256": actual}; inputs.append(path)

    fixture = validate_fixtures()
    samples = json.loads((SOURCE / "raw/student_samples.json").read_text(encoding="utf-8"))
    regressions = retained_regressions(samples)
    if fixture["passed"] != fixture["cases"] or regressions["passed"] != regressions["cases"]:
        raise RuntimeError("fixture or retained regression validation failed")
    rows, seen = [], set()
    for arm in samples:
        for sample in arm["math_samples"]:
            key = (arm["arm"], sample["task_id"])
            if key in seen: raise ValueError(f"duplicate row: {key}")
            seen.add(key)
            got = extract_concluded_numeric_for_question(sample["prompt"], sample["output_text"])
            rows.append({"arm": key[0], "task_id": key[1], "gold": sample["gold"],
                         "original_extracted": sample["extracted"], "original_correct": bool(sample["correct"]),
                         "rescored_extracted": got.value, "rescored_method": got.method,
                         "rescored_correct": bool(numeric_equal(got.value, sample["gold"])),
                         "new_tokens": sample["new_tokens"], "natural_eos": bool(sample["natural_eos"])})
    if len(rows) != 512: raise ValueError(f"expected 512 rows, got {len(rows)}")
    with (raw / "rescored_samples.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows: stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    source_scores = json.loads((SOURCE / "raw/actual_scores.json").read_text(encoding="utf-8"))
    comparison = paired_bootstrap(rows)
    answer = sum(r["rescored_correct"] for r in rows if r["arm"] == "answer_only")
    trace = sum(r["rescored_correct"] for r in rows if r["arm"] == "full_trace")
    signature = list(inspect.signature(extract_concluded_numeric_for_question).parameters)
    metrics = {
        "frozen_deployment_sources_verified": True, "external_fixture_cases": fixture["cases"],
        "external_fixture_pass_rate": fixture["passed"] / fixture["cases"],
        "retained_regression_cases": regressions["cases"],
        "retained_regression_pass_rate": regressions["passed"] / regressions["cases"],
        "rescored_generations": len(rows), "answer_math_correct": answer, "trace_math_correct": trace,
        "answer_accuracy": answer / 256, "trace_third_panel_accuracy": trace / 256,
        "trace_minus_answer": (trace - answer) / 256, "paired_bootstrap": comparison,
        "paired_bootstrap_95ci_lower_trace_minus_answer": comparison["lower_95"],
        "imported_selected_seed_qa_regression": source_scores["imported_selected_seed_qa_regression"],
        "scorer_does_not_receive_gold": signature == ["question", "text"],
        "answer_extraction_coverage": sum(r["rescored_extracted"] is not None for r in rows if r["arm"] == "answer_only") / 256,
        "trace_extraction_coverage": sum(r["rescored_extracted"] is not None for r in rows if r["arm"] == "full_trace") / 256,
        "changed_labels": sum(r["original_correct"] != r["rescored_correct"] for r in rows),
    }
    definitions = {
        "source_integrity": ("frozen_deployment_sources_verified", "eq", True),
        "fixture_validation": ("external_fixture_pass_rate", "eq", 1.0),
        "fixture_coverage": ("external_fixture_cases", "ge", 15),
        "retained_regressions": ("retained_regression_pass_rate", "eq", 1.0),
        "evaluation_coverage": ("rescored_generations", "eq", 512),
        "finalist_gain": ("paired_bootstrap_95ci_lower_trace_minus_answer", "gt", 0.0),
        "finalist_absolute": ("trace_third_panel_accuracy", "ge", .40),
        "protected_retention": ("imported_selected_seed_qa_regression", "le", .05),
        "scorer_blinding": ("scorer_does_not_receive_gold", "eq", True),
    }
    gates = {}
    for gate, (metric, op, threshold) in definitions.items():
        actual = metrics[metric]
        passed = actual == threshold if op == "eq" else actual >= threshold if op == "ge" else actual > threshold if op == "gt" else actual <= threshold
        gates[gate] = {"metric": metric, "operator": op, "threshold": threshold, "actual": actual, "pass": passed}
    write_json(raw / "actual_scores.json", metrics); write_json(raw / "external_fixture_validation.json", fixture)
    write_json(raw / "retained_regression_validation.json", regressions)
    write_json(raw / "independent_evaluation.json", {"comparison": comparison, "answer_correct": answer, "trace_correct": trace})
    write_json(raw / "paired_baseline.json", {"baseline": "answer_only", "treatment": "full_trace", "paired_tasks": 256, "comparison": comparison})
    write_json(raw / "scorer_hashes.json", {"implementation": ledger["tools/analysis/final_numeric_answer_v2.py"],
               "fixtures": ledger["tests/fixtures/final_numeric_answer_v2_cases.json"], "signature": signature})
    source_receipt = json.loads((SOURCE / "raw/receipt.json").read_text(encoding="utf-8"))
    write_json(raw / "source_execution_receipt.json", {"task_id": SOURCE.name,
               "receipt_sha256": sha256_file(SOURCE / "raw/receipt.json"), "receipt_fingerprint": source_receipt["receipt_fingerprint"]})
    write_json(raw / "dataset_hashes.json", {"source_samples": ledger["runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/student_samples.json"]})
    write_json(raw / "model_hash.json", {"rescore_only": True, "source_receipt": ledger["runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/receipt.json"]})
    write_json(raw / "student_samples.json", {"source": "raw/rescored_samples.jsonl", "rows": len(rows)})
    write_json(raw / "teacher_samples.json", {"teacher_rows_used": 0, "rescore_only": True})
    evidence = {"acceptance_gates":"raw/receipt.json", "actual_scores":"raw/actual_scores.json",
        "dataset_hashes":"raw/dataset_hashes.json", "external_fixture_validation":"raw/external_fixture_validation.json",
        "independent_evaluation":"raw/independent_evaluation.json", "model_hash":"raw/model_hash.json",
        "paired_baseline":"raw/paired_baseline.json", "provenance":"raw/receipt.json",
        "raw_samples":"raw/rescored_samples.jsonl", "receipt_fingerprint":"raw/receipt.json",
        "scorer_hashes":"raw/scorer_hashes.json", "source_execution_receipt":"raw/source_execution_receipt.json",
        "student_samples":"raw/student_samples.json", "teacher_samples":"raw/teacher_samples.json"}
    evidence_files = sorted(p for p in raw.rglob("*") if p.is_file())
    provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(), started_at_utc=started,
        started_monotonic=mono, input_paths=[*inputs, *evidence_files], packages=[],
        runtime={"execution_mode":"offline_conservative_rescore", "new_inference":False, "new_training":False})
    complete, errors = provenance_complete(provenance)
    if not complete: raise RuntimeError(f"incomplete provenance: {errors}")
    receipt = {"schema":"local-labs-backlog-receipt-v1", "task_id":TASK_ID, "provenance":provenance,
               "provenance_complete":True, "gates":gates, "evidence":evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt); write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = "TRACE_DISTILLATION_DEPLOYMENT_FINALIST_RESCORED_R3" if not failed else "TRACE_DISTILLATION_DEPLOYMENT_FINALIST_NOT_CONFIRMED_R3"
    (outdir / "RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Answer-only/trace: `{answer}/256` and `{trace}/256`; delta `{metrics['trace_minus_answer']:.6f}`, "
        f"95% CI `[{comparison['lower_95']:.6f}, {comparison['upper_95']:.6f}]`. Failed gates: `{', '.join(failed) if failed else 'none'}`.\n",
        encoding="utf-8", newline="\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true"); args = parser.parse_args()
    if args.selfcheck:
        fixture = validate_fixtures(); assert fixture["cases"] >= 15 and fixture["passed"] == fixture["cases"]; return 0
    receipt = execute(args.outdir.resolve()); print(json.dumps(receipt["gates"], separators=(",", ":")), flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
