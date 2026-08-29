#!/usr/bin/env python3
"""Versioned rebind runner using the historical canonical prompt digest."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import canonical_json_sha256
from tools.research import run_retained_fleet_rebind as base


def context_metrics(
    rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    registry: dict[str, Any],
    artifacts: dict[str, Any],
    generator: Callable[[int, str, str], str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute R1 metrics, replacing only its incorrect prompt digest."""
    metrics, analysis = base.context_metrics(rows, cases, registry, artifacts, generator)
    case_by_key = {
        (row["model"], row["target_tokens"], row["position"], row["replicate"]): row
        for row in cases
    }
    reconstructed = 0
    joins = 0
    for row in rows:
        key = (row["model"], row["target_tokens"], row["position"], row["replicate"])
        case = case_by_key.get(key)
        if case is None:
            continue
        joins += 1
        prompt = generator(int(case["filler_count"]), str(case["position"]), str(case["code"]))
        reconstructed += canonical_json_sha256(prompt) == case["prompt_sha256"] == row["prompt_sha256"]
    metrics["prompt_hash_reconstruction_rate"] = reconstructed / len(rows) if rows else 0.0
    analysis["case_joins"] = joins
    analysis["prompt_hashes_reconstructed"] = reconstructed
    return metrics, analysis


def configure() -> None:
    base.context_metrics = context_metrics
    base.__file__ = __file__


def execute(task_id: str, outdir: pathlib.Path) -> dict[str, Any]:
    configure()
    return base.execute(task_id, outdir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", choices=sorted(base.CONFIGS), required=True)
    parser.add_argument("--outdir", type=pathlib.Path, required=True)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        assert canonical_json_sha256("x") != __import__("hashlib").sha256(b"x").hexdigest()
        return 0
    receipt = execute(args.task_id, args.outdir.resolve())
    print(json.dumps(receipt["gates"], separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
