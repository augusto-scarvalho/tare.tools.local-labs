#!/usr/bin/env python3
"""Independently recompute GDN02 R2 metrics from retained vectors."""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Any, Mapping

import numpy as np


def score_payload(worker: Mapping[str, Any], tensors: Mapping[str, np.ndarray]) -> dict[str, Any]:
    rows = []
    matches = []
    for cell in worker["cells"]:
        layer = int(cell["layer"])
        b = np.asarray(tensors[f"layer_{layer}.baseline"], dtype=np.float64)
        c = np.asarray(tensors[f"layer_{layer}.treatment"], dtype=np.float64)
        o = np.asarray(tensors[f"layer_{layer}.oracle"], dtype=np.float64)
        if b.ndim != 1 or b.shape != c.shape or b.shape != o.shape:
            raise ValueError(f"invalid retained target vectors at layer {layer}")
        d_co = float(np.linalg.norm(c - o))
        d_cb = float(np.linalg.norm(c - b))
        d_bo = float(np.linalg.norm(b - o))
        leakage = 100.0 * d_co / max(d_co + d_cb, 1e-12)
        fidelity = 100.0 * max(0.0, 1.0 - d_co / d_bo)
        cosines = [float(value) for value in cell["collateral_cosines"]]
        if len(cosines) != 49 or not np.isfinite(cosines).all():
            raise ValueError(f"invalid collateral cosine set at layer {layer}")
        collateral = statistics.fmean(cosines) * 100.0
        comparisons = {
            "baseline_oracle_distance": d_bo,
            "correction_oracle_distance": d_co,
            "correction_baseline_distance": d_cb,
            "old_fact_leakage_pct": leakage,
            "updated_fact_fidelity_pct": fidelity,
            "collateral_retention_pct": collateral,
        }
        cell_matches = {
            key: abs(float(cell[key]) - value) <= max(1e-5, abs(value) * 1e-5)
            for key, value in comparisons.items()
        }
        matches.extend(cell_matches.values())
        rows.append({"layer": layer, **comparisons, "matches": cell_matches})
    metrics = {
        "learned_gdn_layer_cells": len(rows),
        "retained_decisive_layer_cells": len(rows),
        "retained_collateral_cosines": sum(len(cell["collateral_cosines"]) for cell in worker["cells"]),
        "recomputed_metric_match_rate": sum(matches) / len(matches),
        "median_old_fact_leakage_pct": statistics.median(row["old_fact_leakage_pct"] for row in rows),
        "median_collateral_retention_pct": statistics.median(row["collateral_retention_pct"] for row in rows),
        "median_updated_fact_fidelity_pct": statistics.median(row["updated_fact_fidelity_pct"] for row in rows),
        "distinct_recurrent_state_conditions": min(cell["distinct_recurrent_state_conditions"] for cell in worker["cells"]),
    }
    return {"schema": "gdn02-retained-score-v1", "rows": rows, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=pathlib.Path, required=True)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    from safetensors.numpy import load_file

    worker = json.loads(args.worker.read_text(encoding="utf-8"))
    result = score_payload(worker, load_file(str(args.bundle)))
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
