#!/usr/bin/env python3
"""Independently rescore retained SLX11 logits tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib


def tensor_sha256(tensor) -> str:
    raw = tensor.detach().float().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def projection_matches(sample: dict, projection: dict) -> bool:
    return bool(
        sample.get("logits_key") == projection.get("logits_key")
        and sample.get("logits_shape") == projection.get("shape")
        and sample.get("argmax_token") == projection.get("argmax_token")
        and sample.get("logits_sha256") == projection.get("logits_sha256")
    )


def run(metadata_path: pathlib.Path, bundle_path: pathlib.Path) -> dict:
    import torch
    from safetensors.torch import load_file

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    samples = metadata["samples"]
    tensors = load_file(str(bundle_path), device="cpu")
    expected_keys = [row["logits_key"] for row in samples]
    if len(expected_keys) != len(set(expected_keys)) or set(tensors) != set(expected_keys):
        raise ValueError("logits key coverage mismatch")
    projections = []
    for sample in samples:
        tensor = tensors[sample["logits_key"]]
        finite = torch.isfinite(tensor)
        projection = {
            "task_id": sample["task_id"],
            "logits_key": sample["logits_key"],
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "nonfinite_count": int((~finite).sum().item()),
            "minimum": float(tensor.float().min().item()),
            "maximum": float(tensor.float().max().item()),
            "argmax_token": int(tensor.argmax(dim=-1).item()),
            "logits_sha256": tensor_sha256(tensor),
        }
        projection["worker_projection_match"] = projection_matches(sample, projection)
        projections.append(projection)
    return {
        "schema": "slx11-logits-independent-evaluation-v1",
        "bundle": str(bundle_path),
        "retained_logits_tensors": len(projections),
        "recomputed_finite_output_rate": sum(row["nonfinite_count"] == 0 for row in projections) / len(projections),
        "recomputed_projection_match_rate": sum(row["worker_projection_match"] for row in projections) / len(projections),
        "projections": projections,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=pathlib.Path, required=True)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = run(args.metadata, args.bundle)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("retained_logits_tensors", "recomputed_finite_output_rate", "recomputed_projection_match_rate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
