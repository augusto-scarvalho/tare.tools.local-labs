#!/usr/bin/env python3
"""Correct versioned rebind runner for retained context successor packets."""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import canonical_json_sha256
from tools.research import run_retained_fleet_rebind as base

_r1_context_metrics = base.context_metrics


def context_metrics(
    rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    registry: dict[str, Any],
    artifacts: dict[str, Any],
    generator: Callable[[int, str, str], str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics, analysis = _r1_context_metrics(rows, cases, registry, artifacts, generator)
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


def successor_configs() -> dict[str, dict[str, Any]]:
    envelope = copy.deepcopy(base.CONFIGS["BACKLOG-FLEET-CONTEXT-ENVELOPE-04"])
    envelope.update({
        "claim_pass": "QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_REBOUND_R5",
        "claim_fail": "QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_NOT_CONFIRMED_R5",
    })
    envelope["sources"].update({
        "config/research_backlog_admissions/BACKLOG-FLEET-CONTEXT-ENVELOPE-05.json": "a7c18505b6a37ee9cb5ec5d41034555487c9dadb1cf9d97b4d0bb180c66a1c39",
        "runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-04/raw/receipt.json": "6559d46b6e02a935db4269eca50d1bad20532183efa789d7a2f7796c9d57c50f",
    })

    interference = copy.deepcopy(base.CONFIGS["BACKLOG-FLEET-CONTEXT-INTERFERENCE-02"])
    interference.update({
        "claim_pass": "QUALIFIED_TEXT_FLEET_CONTEXT_INTERFERENCE_REBOUND_R3",
        "claim_fail": "QUALIFIED_TEXT_FLEET_CONTEXT_INTERFERENCE_NOT_CONFIRMED_R3",
    })
    interference["sources"].update({
        "config/research_backlog_admissions/BACKLOG-FLEET-CONTEXT-INTERFERENCE-03.json": "d3a76aa42247106a0eb9b7b6c786645af52cb321147bb73dc630db87b51c2c66",
        "runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-02/raw/run.terminal.json": "5f423740344e35dc9105893331cea302ffa8282af858995b51e78a0ed9a25778",
        "runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-02/runner.stderr.log": "8da89e242c912622680e297a7ee4922683f47b7eda4e4fe6b6474c52a4608e42",
    })
    return {
        "BACKLOG-FLEET-CONTEXT-ENVELOPE-05": envelope,
        "BACKLOG-FLEET-CONTEXT-INTERFERENCE-03": interference,
    }


def configure() -> None:
    base.CONFIGS = successor_configs()
    base.context_metrics = context_metrics
    base.__file__ = __file__


def execute(task_id: str, outdir: pathlib.Path) -> dict[str, Any]:
    configure()
    return base.execute(task_id, outdir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", choices=sorted(successor_configs()), required=True)
    parser.add_argument("--outdir", type=pathlib.Path, required=True)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        assert set(successor_configs()) == {
            "BACKLOG-FLEET-CONTEXT-ENVELOPE-05",
            "BACKLOG-FLEET-CONTEXT-INTERFERENCE-03",
        }
        return 0
    receipt = execute(args.task_id, args.outdir.resolve())
    print(json.dumps(receipt["gates"], separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
