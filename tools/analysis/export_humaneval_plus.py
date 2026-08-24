#!/usr/bin/env python3
"""Export EvalPlus HumanEval+ prompts for the Windows-side lifecycle harness."""
from __future__ import annotations

import json
import pathlib

from evalplus.data import get_human_eval_plus


destination = pathlib.Path(__file__).parents[1] / "benchmarks" / "workloads" / "humaneval_plus.jsonl"
destination.parent.mkdir(parents=True, exist_ok=True)
problems = get_human_eval_plus()
with destination.open("w", encoding="utf-8", newline="\n") as stream:
    for task_id in sorted(problems):
        item = problems[task_id]
        stream.write(json.dumps({"task_id": task_id, "prompt": item["prompt"]}) + "\n")
print(f"exported {len(problems)} problems -> {destination}")
