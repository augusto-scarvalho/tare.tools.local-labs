#!/usr/bin/env python3
"""Identity and infrastructure gate for the LAB-CODE-003 SWE-bench Verified pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from datetime import datetime, timezone


DATASET = "SWE-bench/SWE-bench_Verified"
REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"
EXPECTED_ROWS = 500


def canonical_sha256(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def spread_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0 or count > length:
        raise ValueError("require 0 < count <= length")
    if count == 1:
        return [0]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def selfcheck() -> None:
    rows_a = [{"b": 2, "a": 1}, {"id": "x"}]
    rows_b = [{"a": 1, "b": 2}, {"id": "x"}]
    assert canonical_sha256(rows_a) == canonical_sha256(rows_b)
    assert spread_indices(10, 4) == [0, 3, 6, 9]
    assert spread_indices(1, 1) == [0]
    print("SWE-bench Verified pilot self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--gold-predictions", type=pathlib.Path)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    if not args.manifest or not args.gold_predictions:
        parser.error("--manifest and --gold-predictions are required")

    from datasets import load_dataset

    dataset = load_dataset(DATASET, split="test", revision=REVISION)
    rows = sorted((dict(row) for row in dataset), key=lambda row: row["instance_id"])
    ids = [row["instance_id"] for row in rows]
    if len(rows) != EXPECTED_ROWS or len(set(ids)) != EXPECTED_ROWS:
        raise RuntimeError(f"dataset identity gate failed: rows={len(rows)} unique={len(set(ids))}")
    pilot = [ids[index] for index in spread_indices(len(rows), 10)]
    first = rows[0]
    manifest = {
        "campaign": "LAB-CODE-003",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "revision": REVISION,
        "split": "test",
        "rows": len(rows),
        "unique_instance_ids": len(set(ids)),
        "dataset_fingerprint": dataset._fingerprint,
        "canonical_content_sha256": canonical_sha256(rows),
        "columns": list(dataset.column_names),
        "gold_probe_instance_id": first["instance_id"],
        "future_pilot_instance_ids": pilot,
    }
    prediction = {
        first["instance_id"]: {
            "model_name_or_path": "gold",
            "instance_id": first["instance_id"],
            "model_patch": first["patch"],
        }
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    args.gold_predictions.write_text(json.dumps(prediction, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

