#!/usr/bin/env python3
"""Score RepoBench-P predictions with the revision-pinned official LongBench metric."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import statistics


def load_metric(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("longbench_metrics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.code_sim_score


def first_code_line(prediction: str) -> str:
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            return line
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=pathlib.Path, required=True)
    parser.add_argument("--official-metrics", type=pathlib.Path, required=True)
    parser.add_argument("--dataset", type=pathlib.Path)
    parser.add_argument("--receipts", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    metric = load_metric(args.official_metrics)
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line]
    scores = [max(metric(row["pred"], answer, all_classes=row["all_classes"])
                  for answer in row["answers"]) for row in rows]
    strata = {"0-4k": [], "4-8k": [], "8k+": []}
    exact = 0
    language_scores = {}
    dataset_rows = {}
    if args.dataset is not None:
        dataset_rows = {row["_id"]: row for row in
                        (json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line)}
    for row, score in zip(rows, scores):
        key = "0-4k" if row["length"] < 4000 else "4-8k" if row["length"] < 8000 else "8k+"
        strata[key].append(score)
        exact += int(first_code_line(row["pred"]).strip() == row["answers"][0].strip())
        if dataset_rows:
            language_scores.setdefault(dataset_rows[row["_id"]]["language"], []).append(score)
    digest = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    result = {
        "schema": "tare.longbench.repobench-p.v1", "n": len(rows), "unique_ids": len({r["_id"] for r in rows}),
        "official_code_similarity": round(100 * statistics.mean(scores), 2),
        "exact_first_line": exact, "exact_first_line_rate": exact / len(rows),
        "by_reported_length": {key: {"n": len(values), "score": round(100 * statistics.mean(values), 2)}
                               for key, values in strata.items() if values},
        "by_language": {key: {"n": len(values), "score": round(100 * statistics.mean(values), 2)}
                        for key, values in sorted(language_scores.items())},
        "predictions_sha256": digest,
    }
    if args.dataset is not None:
        result["dataset_sha256"] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if args.receipts is not None:
        receipts = [json.loads(line) for line in args.receipts.read_text(encoding="utf-8").splitlines() if line]
        result["operational"] = {
            "receipts": len(receipts), "unique_ids": len({row["_id"] for row in receipts}),
            "nonempty": sum(bool(row["nonempty"]) for row in receipts),
            "stopped_eos": sum(row["stopped_eos"] is True for row in receipts),
            "stopped_limit": sum(row["stopped_limit"] is True for row in receipts),
            "prompt_tokens_min": min(row["prompt_tokens"] for row in receipts),
            "prompt_tokens_median": statistics.median(row["prompt_tokens"] for row in receipts),
            "prompt_tokens_max": max(row["prompt_tokens"] for row in receipts),
            "wall_seconds": sum(row["wall_s"] for row in receipts),
            "receipts_sha256": hashlib.sha256(args.receipts.read_bytes()).hexdigest(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
