#!/usr/bin/env python3
"""Confirm SLX08 relevance selection with a balanced exact-bound campaign."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import run_slx08_relevance_prefill_r6 as r6

TASK_ID = "BACKLOG-SLX08-RELEVANCE-PREFILL-11"
PAIRS = 126
CASE_OFFSET = 126
PRE_REG_SHA256 = "149324cba2066e19ee11c939e20af64b36ffb5377221d00fa3b31e16adde19d5"
SOURCE_HASHES = {
    "runs/research/BACKLOG-SLX08-RELEVANCE-PREFILL-10/raw/receipt.json": "7f573c2ae762772aadf2368a1945adfcf29f45feef0f35153453538e3c87324a",
    "runs/research/BACKLOG-SLX08-RELEVANCE-PREFILL-10/REVIEW.json": "6d018172936bc13870b4a2e6dfe107fc497aebd7cdfcc2e77e5d09b276f3560c",
    "tools/research/run_slx08_physical_prefill_r4.py": "42141bffd6c51635f1b0ec6e1ff3b531f4c7ee2b8eca933ee3371b673b86bf6d",
    "tools/research/run_slx08_physical_prefill_r5.py": "42a0c405bedd7003a6d28b17152261ea690bf1fb5cd6ca6f3e18f6451fd94848",
    "tools/research/run_slx08_relevance_prefill_r6.py": "fccac2347d78c3307448fe30c4cdc25363863e01a935286263a86df034f847e2",
}


def binomial_cdf(k: int, n: int, probability: float) -> float:
    return sum(
        math.comb(n, value) * probability**value * (1.0 - probability) ** (n - value)
        for value in range(k + 1)
    )


def exact_upper_failure_bound(failures: int, opportunities: int, alpha: float = 0.05) -> float:
    if opportunities <= 0 or failures < 0 or failures > opportunities:
        raise ValueError("invalid exact-bound counts")
    if failures == opportunities:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if binomial_cdf(failures, opportunities, middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def freeze_fixtures() -> list[dict]:
    status, gateway = r6.base.health(8080)
    if status != 200 or not isinstance(gateway, dict) or gateway.get("current_model") != "qwen38" or not gateway.get("backend_healthy"):
        raise RuntimeError(f"healthy qwen38 tokenizer backend required: {status}: {gateway}")
    backend_port = gateway.get("backend_port")
    if not isinstance(backend_port, int):
        raise RuntimeError(f"gateway did not expose an integer backend port: {gateway}")
    original_base = r6.base.TEMP_BASE
    original_builder = r6.build_fixture
    try:
        r6.base.TEMP_BASE = f"http://127.0.0.1:{backend_port}"
        fixtures = [original_builder(case_id + CASE_OFFSET) for case_id in range(PAIRS)]
    finally:
        r6.base.TEMP_BASE = original_base
    if len(fixtures) != PAIRS or sorted({fixture["case_id"] % 14 for fixture in fixtures}) != list(range(14)):
        raise RuntimeError("R11 fixtures do not cover every evidence position")
    if any(sum(fixture["case_id"] % 14 == position for fixture in fixtures) != 9 for position in range(14)):
        raise RuntimeError("R11 evidence positions are not exactly balanced")
    if any(sum(fixture["case_id"] % 3 == period for fixture in fixtures) != 42 for period in range(3)):
        raise RuntimeError("R11 arm-order periods are not exactly balanced")
    return fixtures


def evaluate_gates(metrics: dict) -> dict:
    definitions = {
        "dense_control": ("dense_requests", "ge", PAIRS),
        "naive_control": ("naive_requests", "ge", PAIRS),
        "relevance_treatment": ("relevance_requests", "ge", PAIRS),
        "route_observation": ("relevance_route_observation_rate", "eq", 1.0),
        "evidence_retention": ("relevance_evidence_retention_rate", "eq", 1.0),
        "retained_fraction": ("relevance_median_retained_fraction", "eq", 0.5),
        "dense_semantic_floor": ("dense_accuracy", "ge", 0.9),
        "relevance_semantic_floor": ("relevance_accuracy", "ge", 0.9),
        "semantic_noninferiority": ("relevance_vs_dense_exact_delta_ci95_low", "ge", -0.03),
        "selector_value": ("relevance_vs_naive_accuracy_delta", "ge", 0.2),
        "ttft_gain": ("relevance_vs_dense_p50_ttft_speedup", "ge", 1.1),
        "tail_safety": ("relevance_vs_dense_p95_ttft_speedup", "ge", 1.0),
        "service_restore": ("original_service_restored", "eq", 1),
        "embedding_integrity": ("embedding_health", "eq", 200),
    }
    operators = {"eq": lambda actual, threshold: actual == threshold, "ge": lambda actual, threshold: actual >= threshold}
    return {
        gate: {"metric": metric, "operator": operator, "threshold": threshold, "actual": metrics[metric], "pass": operators[operator](metrics[metric], threshold)}
        for gate, (metric, operator, threshold) in definitions.items()
    }


def execute(outdir: pathlib.Path) -> tuple[dict, dict]:
    fixtures = freeze_fixtures()
    r6.TASK_ID = TASK_ID
    r6.PRE_REG_SHA256 = PRE_REG_SHA256
    r6.SOURCE_HASHES = SOURCE_HASHES
    r6.base.PAIRS = PAIRS
    r6.build_fixture = lambda case_id: fixtures[case_id]

    original_score = r6.score
    original_gates = r6.evaluate_gates
    original_provenance = r6.build_provenance
    original_write_json = r6.base.write_json
    runner_path = pathlib.Path(__file__).resolve()

    def exact_score(pairs: list[dict], restored: bool, embedding_status: int | None) -> dict:
        metrics = original_score(pairs, restored, embedding_status)
        opportunities = sum(pair["dense"]["correct"] for pair in pairs)
        failures = sum(pair["dense"]["correct"] and not pair["relevance"]["correct"] for pair in pairs)
        upper = exact_upper_failure_bound(failures, opportunities)
        metrics.update({
            "relevance_vs_dense_failure_opportunities": opportunities,
            "relevance_vs_dense_failures": failures,
            "relevance_vs_dense_exact_failure_rate_ci95_high": upper,
            "relevance_vs_dense_exact_delta_ci95_low": -upper,
        })
        return metrics

    def bound_provenance(**kwargs):
        kwargs["script_path"] = runner_path
        inputs = list(kwargs["input_paths"])
        if runner_path not in inputs:
            inputs.append(runner_path)
        kwargs["input_paths"] = inputs
        return original_provenance(**kwargs)

    def corrected_evidence(path: pathlib.Path, value) -> None:
        if path.name == "falsifiable_hypothesis.json":
            value.update({"fixtures": PAIRS, "case_offset": CASE_OFFSET, "noninferiority": "one-sided exact binomial", "alpha": 0.05})
        elif path.name == "treatment_materiality.json":
            value["requests_per_arm"] = PAIRS
        elif path.name == "independent_evaluation.json":
            value.update({"noninferiority": "one-sided exact binomial upper bound on relevance failures among dense-success opportunities", "balanced_positions": 14, "balanced_order_periods": 3})
        original_write_json(path, value)

    r6.score = exact_score
    r6.evaluate_gates = evaluate_gates
    r6.build_provenance = bound_provenance
    r6.base.write_json = corrected_evidence
    try:
        return r6.execute(outdir)
    finally:
        r6.score = original_score
        r6.evaluate_gates = original_gates
        r6.build_provenance = original_provenance
        r6.base.write_json = original_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt, metrics = execute(args.outdir.resolve())
    passed = all(gate["pass"] for gate in receipt["gates"].values())
    claim = "SLX08_RELEVANCE_SELECTED_PREFILL_QUALIFIED_R11" if passed else "SLX08_RELEVANCE_SELECTED_PREFILL_REJECTED_R11"
    failed = [gate for gate, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review. Dense/naive/relevance accuracy: "
        f"{metrics['dense_accuracy']:.4f}/{metrics['naive_accuracy']:.4f}/{metrics['relevance_accuracy']:.4f}; exact delta CI95 low "
        f"{metrics['relevance_vs_dense_exact_delta_ci95_low']:.6f}; relevance p50 TTFT speedup {metrics['relevance_vs_dense_p50_ttft_speedup']:.4f}x; "
        f"failed gates: {', '.join(failed) if failed else 'none'}. Bounded to the frozen R11 panel.\n",
        encoding="utf-8",
    )
    print(json.dumps({"claim": claim, "metrics": metrics, "gates": receipt["gates"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
