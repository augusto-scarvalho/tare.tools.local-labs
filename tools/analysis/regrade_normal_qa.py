#!/usr/bin/env python3
"""Create a derivative grading receipt without modifying raw generations."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "benchmarks"))

from normal_qa_ab import grade, load_tasks, summarize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=pathlib.Path, required=True)
    parser.add_argument("--responses", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    records = json.loads(args.responses.read_text(encoding="utf-8"))
    by_id = {task["id"]: task for task in tasks}
    if len(records) != len(tasks) or {row["id"] for row in records} != set(by_id):
        raise SystemExit("response/task identity mismatch")

    derived = []
    for row in records:
        task = by_id[row["id"]]
        if row["prompt"] != task["prompt"]:
            raise SystemExit(f"prompt mismatch: {row['id']}")
        passed, detail = grade(task, row["answer"])
        derived.append({**row, "pass": passed, "grade_detail": detail})

    receipt = {
        "source_response_path": str(args.responses),
        "source_response_sha256": hashlib.sha256(args.responses.read_bytes()).hexdigest(),
        "tasks_path": str(args.tasks),
        "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "summary": summarize(derived),
        "records": derived,
    }
    args.out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(receipt["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
