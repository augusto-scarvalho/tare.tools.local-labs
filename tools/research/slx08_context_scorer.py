#!/usr/bin/env python3
"""Independently reopen and score retained SLX-08 context vectors."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics

ARMS = ("dense", "corrected", "legacy")


def expected_keys(samples: list[dict]) -> set[str]:
    return {f"{row['cell']}_{arm}" for row in samples for arm in ARMS}


def validate_keyset(actual: set[str], samples: list[dict]) -> None:
    expected = expected_keys(samples)
    if actual != expected:
        raise ValueError(f"bundle key mismatch: missing={sorted(expected-actual)} extra={sorted(actual-expected)}")


def projection_match(stored: float, recomputed: float, tolerance: float = 1e-9) -> bool:
    return abs(float(stored) - float(recomputed)) <= tolerance


def summarize(evaluations: list[dict]) -> dict:
    return {
        "retained_context_cells": len(evaluations),
        "recomputed_projection_match_rate": (
            sum(bool(row["projection_match"]) for row in evaluations) / len(evaluations)
            if evaluations else 0.0
        ),
        "recomputed_median_selected_block_context_cosine": statistics.median(
            row["selected_block_context_cosine"] for row in evaluations
        ),
        "recomputed_median_legacy_first_half_context_cosine": statistics.median(
            row["legacy_first_half_context_cosine"] for row in evaluations
        ),
        "nonfinite_values": sum(row["nonfinite_values"] for row in evaluations),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--worker", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    import torch
    from safetensors import safe_open

    payload = json.loads(args.worker.read_text(encoding="utf-8"))
    samples = payload["samples"]
    evaluations = []
    with safe_open(str(args.bundle), framework="pt", device="cpu") as stream:
        validate_keyset(set(stream.keys()), samples)
        for row in samples:
            vectors = {arm: stream.get_tensor(f"{row['cell']}_{arm}") for arm in ARMS}
            nonfinite = sum(int((~torch.isfinite(value)).sum().item()) for value in vectors.values())
            if any(list(value.shape) != [8, 1, 256] for value in vectors.values()):
                raise ValueError(f"invalid context-vector shape: {row['cell']}")
            hashes = {
                arm: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
                for arm, value in vectors.items()
            }
            selected = float(
                torch.nn.functional.cosine_similarity(
                    vectors["dense"].flatten(), vectors["corrected"].flatten(), dim=0
                ).item()
            )
            legacy = float(
                torch.nn.functional.cosine_similarity(
                    vectors["dense"].flatten(), vectors["legacy"].flatten(), dim=0
                ).item()
            )
            match = (
                hashes == row["context_vector_sha256"]
                and projection_match(row["selected_block_context_cosine"], selected)
                and projection_match(row["legacy_first_half_context_cosine"], legacy)
                and nonfinite == 0
            )
            evaluations.append(
                {
                    "cell": row["cell"],
                    "shape": list(vectors["dense"].shape),
                    "dtype": str(vectors["dense"].dtype),
                    "sha256": hashes,
                    "selected_block_context_cosine": selected,
                    "legacy_first_half_context_cosine": legacy,
                    "nonfinite_values": nonfinite,
                    "projection_match": match,
                }
            )
    result = {"schema": "slx08-context-evaluation-r2", "evaluations": evaluations, "summary": summarize(evaluations)}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
