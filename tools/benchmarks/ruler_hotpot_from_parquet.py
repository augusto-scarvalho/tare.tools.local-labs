#!/usr/bin/env python3
"""Convert the revision-pinned Hugging Face HotpotQA parquet to RULER's source shape."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = pq.read_table(args.input).to_pylist()
    converted = []
    for row in rows:
        context = row["context"]
        converted.append(
            {
                "_id": row["id"],
                "answer": row["answer"],
                "question": row["question"],
                "supporting_facts": list(
                    zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"])
                ),
                "context": list(zip(context["title"], context["sentences"])),
                "type": row["type"],
                "level": row["level"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, ensure_ascii=False), encoding="utf-8")
    print(f"converted {len(converted)} HotpotQA rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
