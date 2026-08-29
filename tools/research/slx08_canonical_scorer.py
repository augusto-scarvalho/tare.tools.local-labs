#!/usr/bin/env python3
"""Canonical fixed-order scorer for retained SLX-08 context vectors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import statistics

ARMS = ("dense", "corrected", "legacy")


def canonical_cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("cosine inputs must have equal nonzero length")
    dot = math.fsum(a * b for a, b in zip(left, right))
    left_norm = math.fsum(a * a for a in left)
    right_norm = math.fsum(b * b for b in right)
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("cosine inputs must have positive norms")
    return dot / math.sqrt(left_norm * right_norm)


def expected_keys(samples: list[dict]) -> set[str]:
    return {f"{row['cell']}_{arm}" for row in samples for arm in ARMS}


def summarize(rows: list[dict], tensor_count: int) -> dict:
    return {
        "retained_context_tensors": tensor_count,
        "retained_context_cells": len(rows),
        "tensor_hash_match_rate": sum(row["tensor_hash_match"] for row in rows) / len(rows) if rows else 0.0,
        "nonfinite_values": sum(row["nonfinite_values"] for row in rows),
        "canonical_median_selected_block_context_cosine": statistics.median(
            row["selected_block_context_cosine"] for row in rows
        ),
        "canonical_median_legacy_first_half_context_cosine": statistics.median(
            row["legacy_first_half_context_cosine"] for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--samples", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    import torch
    from safetensors import safe_open

    samples = [json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    with safe_open(str(args.bundle), framework="pt", device="cpu") as stream:
        keys = set(stream.keys())
        expected = expected_keys(samples)
        if keys != expected:
            raise ValueError(f"bundle key mismatch: missing={sorted(expected-keys)} extra={sorted(keys-expected)}")
        for source in samples:
            tensors = {arm: stream.get_tensor(f"{source['cell']}_{arm}") for arm in ARMS}
            if any(list(value.shape) != [8, 1, 256] for value in tensors.values()):
                raise ValueError(f"invalid shape: {source['cell']}")
            if any(value.dtype != torch.float32 for value in tensors.values()):
                raise ValueError(f"invalid dtype: {source['cell']}")
            flat = {arm: value.reshape(-1).tolist() for arm, value in tensors.items()}
            nonfinite = sum(sum(not math.isfinite(item) for item in values) for values in flat.values())
            hashes = {
                arm: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
                for arm, value in tensors.items()
            }
            rows.append(
                {
                    "cell": source["cell"],
                    "context": source["context"],
                    "layer": source["layer"],
                    "shape": [8, 1, 256],
                    "dtype": "float32",
                    "sha256": hashes,
                    "tensor_hash_match": hashes == source["context_vector_sha256"],
                    "nonfinite_values": nonfinite,
                    "selected_block_context_cosine": canonical_cosine(flat["dense"], flat["corrected"]),
                    "legacy_first_half_context_cosine": canonical_cosine(flat["dense"], flat["legacy"]),
                }
            )
    result = {
        "schema": "slx08-canonical-context-score-r3",
        "method": "row-major Python float plus math.fsum float64 products and squared norms",
        "rows": rows,
        "summary": summarize(rows, len(keys)),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
