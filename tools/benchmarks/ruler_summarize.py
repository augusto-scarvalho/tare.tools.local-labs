#!/usr/bin/env python3
"""Summarize RULER endpoint receipts and hash the generated dataset manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import Counter, defaultdict


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=pathlib.Path, required=True)
    parser.add_argument("--data-root", type=pathlib.Path, action="append", default=[])
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.receipts.read_text(encoding="utf-8").splitlines() if line]
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["target_length"], row["task"])].append(row)

    lengths = sorted({row["target_length"] for row in rows})
    by_length = {}
    for length in lengths:
        task_groups = {task: values for (item_length, task), values in groups.items() if item_length == length}
        pilot_scores = [min(values, key=lambda item: item["timestamp"])["score"] for values in task_groups.values()]
        task_scores = {task: sum(item["score"] for item in values) / len(values)
                       for task, values in sorted(task_groups.items())}
        length_rows = [row for row in rows if row["target_length"] == length]
        by_length[str(length)] = {
            "pilot_n_tasks": len(pilot_scores),
            "pilot_macro_accuracy": sum(pilot_scores) / len(pilot_scores),
            "bounded_panel_macro_accuracy": sum(task_scores.values()) / len(task_scores),
            "task_accuracy": task_scores,
            "task_n": {task: len(values) for task, values in sorted(task_groups.items())},
            "receipt_n": len(length_rows),
            "prompt_tokens_min": min(row["actual_prompt_tokens"] for row in length_rows),
            "prompt_tokens_max": max(row["actual_prompt_tokens"] for row in length_rows),
            "wall_seconds": sum(row["wall_s"] for row in length_rows),
            "finish_reason_counts": dict(sorted(Counter(row["finish_reason"] for row in length_rows).items())),
        }

    manifests = []
    for root in args.data_root:
        for path in sorted(root.rglob("test.jsonl")):
            manifests.append({
                "root": str(root),
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "lines": sum(1 for line in path.open(encoding="utf-8") if line.strip()),
                "sha256": sha256(path),
            })

    output = {
        "schema": "tare.ruler.summary.v1",
        "receipts": {"path": str(args.receipts), "rows": len(rows), "sha256": sha256(args.receipts)},
        "by_length": by_length,
        "dataset_manifests": manifests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["by_length"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
