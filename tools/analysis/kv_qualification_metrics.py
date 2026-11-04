#!/usr/bin/env python3
"""Codec-independent full-distribution metrics for BEE-L2 KV qualification.

Input files are JSON arrays with one row per identical token position:
``{"id": "prompt/position", "log_probs": [full-vocabulary log probabilities]}``.
Top-k responses are deliberately rejected because they cannot establish full-support
KL or Jensen-Shannon divergence.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics


def normalize(log_probs: list[float]) -> list[float]:
    if not log_probs or not all(math.isfinite(value) for value in log_probs):
        raise ValueError("log_probs must be a non-empty finite full-vocabulary vector")
    peak = max(log_probs)
    total = sum(math.exp(value - peak) for value in log_probs)
    log_z = peak + math.log(total)
    return [value - log_z for value in log_probs]


def kl_divergence(p_log: list[float], q_log: list[float]) -> float:
    if len(p_log) != len(q_log):
        raise ValueError("distribution dimensions differ")
    p, q = normalize(p_log), normalize(q_log)
    return sum(math.exp(lp) * (lp - lq) for lp, lq in zip(p, q))


def jensen_shannon(p_log: list[float], q_log: list[float]) -> float:
    if len(p_log) != len(q_log):
        raise ValueError("distribution dimensions differ")
    p, q = normalize(p_log), normalize(q_log)
    m = []
    for lp, lq in zip(p, q):
        peak = max(lp, lq)
        m.append(peak + math.log(math.exp(lp - peak) + math.exp(lq - peak)) -
                 math.log(2.0))
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty metric vector")
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def compare(reference_rows: list[dict], candidate_rows: list[dict]) -> dict:
    reference = {row["id"]: row for row in reference_rows}
    candidate = {row["id"]: row for row in candidate_rows}
    if set(reference) != set(candidate):
        missing = sorted(set(reference) - set(candidate))[:5]
        extra = sorted(set(candidate) - set(reference))[:5]
        raise ValueError(f"unpaired token positions: missing={missing} extra={extra}")
    rows = []
    for row_id in sorted(reference):
        p = reference[row_id]["log_probs"]
        q = candidate[row_id]["log_probs"]
        if len(p) < 1000:
            raise ValueError(
                f"{row_id}: vector has {len(p)} entries; top-k/truncated logits rejected")
        rows.append({
            "id": row_id,
            "kl_reference_candidate": kl_divergence(p, q),
            "kl_candidate_reference": kl_divergence(q, p),
            "jensen_shannon": jensen_shannon(p, q),
            "top1_equal": max(range(len(p)), key=p.__getitem__) ==
                          max(range(len(q)), key=q.__getitem__),
        })
    js = [row["jensen_shannon"] for row in rows]
    return {
        "n": len(rows),
        "median_jensen_shannon": statistics.median(js),
        "p95_jensen_shannon": percentile(js, 0.95),
        "max_jensen_shannon": max(js),
        "top1_agreement": sum(row["top1_equal"] for row in rows) / len(rows),
        "rows": rows,
    }


def selfcheck() -> int:
    size = 1024
    a = [-20.0] * size
    b = list(a)
    a[3], b[3] = 0.0, 0.0
    same = compare([{"id": "x/0", "log_probs": a}],
                   [{"id": "x/0", "log_probs": b}])
    assert same["median_jensen_shannon"] == 0.0
    assert same["top1_agreement"] == 1.0
    b[3], b[4] = -1.0, 0.0
    changed = compare([{"id": "x/0", "log_probs": a}],
                      [{"id": "x/0", "log_probs": b}])
    assert changed["median_jensen_shannon"] > 0
    assert changed["top1_agreement"] == 0.0
    try:
        compare([{"id": "x", "log_probs": [0.0, -1.0]}],
                [{"id": "x", "log_probs": [0.0, -1.0]}])
    except ValueError as error:
        assert "top-k" in str(error)
    else:
        raise AssertionError("truncated logits must fail closed")
    print("kv qualification metrics self-check: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=pathlib.Path)
    parser.add_argument("--candidate", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        return selfcheck()
    if not args.reference or not args.candidate or not args.output:
        parser.error("--reference, --candidate, and --output are required")
    result = compare(json.loads(args.reference.read_text(encoding="utf-8")),
                     json.loads(args.candidate.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
