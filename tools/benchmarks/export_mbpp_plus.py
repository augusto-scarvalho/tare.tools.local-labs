#!/usr/bin/env python3
"""Export the official EvalPlus MBPP+ release into a stable local prompt manifest.

Run with the Python environment that owns evalplus.  Canonical solutions and tests are
intentionally not exported, preventing accidental prompt leakage; the official release
hash is stored in a sidecar and scoring reloads the dataset inside EvalPlus.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    dataset = get_mbpp_plus()
    rows = [{"task_id": item["task_id"], "prompt": item["prompt"],
             "entry_point": item["entry_point"]} for item in dataset.values()]
    rows.sort(key=lambda row: int(row["task_id"].split("/")[-1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {"benchmark": "MBPP+", "evalplus_dataset": "mbpp",
                "evalplus_release_hash": get_mbpp_plus_hash(), "n_problems": len(rows),
                "fields_exported": ["task_id", "prompt", "entry_point"],
                "canonical_solutions_exported": False}
    args.output.with_suffix(".identity.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
