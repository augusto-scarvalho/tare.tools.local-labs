#!/usr/bin/env python3
"""Score an MBPP+ subset with the official EvalPlus executor, fail-closed.

EvalPlus requires all benchmark ids.  Missing ids are padded with guaranteed-failing
empty solutions, stale result caches are removed, and the reported denominator contains
only the originally selected ids.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from benchmark_harness_qa import bust_stale_results, pad_subset, wilson_interval  # noqa: E402
from evalplus.data import get_mbpp_plus  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.executable} {sys.argv[0]} <subset_samples.jsonl>", file=sys.stderr)
        return 2
    samples = pathlib.Path(sys.argv[1]).resolve()
    mine = {}
    for line_number, line in enumerate(samples.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row.get("task_id")
        if task_id in mine:
            raise ValueError(f"duplicate task_id {task_id!r} at line {line_number}")
        mine[task_id] = row.get("solution", "")
    if not mine:
        raise ValueError("empty subset")
    all_ids = list(get_mbpp_plus())
    unknown = sorted(set(mine) - set(all_ids))
    if unknown:
        raise ValueError(f"unknown task ids: {unknown}")
    padded = samples.with_suffix(".padded.jsonl")
    with padded.open("w", encoding="utf-8") as handle:
        for row in pad_subset(mine, all_ids):
            handle.write(json.dumps(row) + "\n")
    results_path = padded.with_name(padded.stem + "_eval_results.json")
    bust_stale_results(results_path)
    subprocess.run([sys.executable, "-m", "evalplus.evaluate", "--dataset", "mbpp",
                    "--samples", str(padded), "--parallel", "4"], check=True)
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    evaluated = payload["eval"]
    base_ok = plus_ok = 0
    failures = []
    for task_id in mine:
        value = evaluated.get(task_id)
        entry = value[0] if isinstance(value, list) and value else (value or {})
        base_pass = entry.get("base_status") == "pass"
        plus_pass = entry.get("plus_status") == "pass"
        base_ok += int(base_pass)
        plus_ok += int(plus_pass)
        if not plus_pass:
            failures.append({"task_id": task_id, "base_status": entry.get("base_status"),
                             "plus_status": entry.get("plus_status")})
    n = len(mine)
    base_ci = wilson_interval(base_ok, n)
    plus_ci = wilson_interval(plus_ok, n)
    report = {"benchmark": "MBPP+", "subset_samples": str(samples), "n": n,
              "base_pass": base_ok, "base_pass_at_1": base_ok / n,
              "base_wilson_95": list(base_ci), "plus_pass": plus_ok,
              "plus_pass_at_1": plus_ok / n, "plus_wilson_95": list(plus_ci),
              "failures": failures, "padded_n": len(all_ids),
              "stale_result_cache_busted": True}
    report_path = samples.with_name(samples.stem + "_score.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"evidence: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
