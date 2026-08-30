import numpy as np

from tools.research.gdn02_retained_scorer import score_payload


def test_retained_scorer_recomputes_three_cells_and_147_cosines():
    tensors = {}
    cells = []
    for layer in range(3):
        baseline = np.array([0.0, 0.0], dtype=np.float32)
        treatment = np.array([0.9, 0.0], dtype=np.float32)
        oracle = np.array([1.0, 0.0], dtype=np.float32)
        tensors[f"layer_{layer}.baseline"] = baseline
        tensors[f"layer_{layer}.treatment"] = treatment
        tensors[f"layer_{layer}.oracle"] = oracle
        cells.append({
            "layer": layer,
            "baseline_oracle_distance": 1.0,
            "correction_oracle_distance": 0.1,
            "correction_baseline_distance": 0.9,
            "old_fact_leakage_pct": 10.0,
            "updated_fact_fidelity_pct": 90.0,
            "collateral_retention_pct": 100.0,
            "collateral_cosines": [1.0] * 49,
            "distinct_recurrent_state_conditions": 3,
        })
    result = score_payload({"cells": cells}, tensors)
    assert result["metrics"]["retained_decisive_layer_cells"] == 3
    assert result["metrics"]["retained_collateral_cosines"] == 147
    assert result["metrics"]["recomputed_metric_match_rate"] == 1.0


def test_retained_scorer_rejects_incomplete_collateral_fixture():
    tensors = {
        "layer_0.baseline": np.array([0.0]),
        "layer_0.treatment": np.array([0.5]),
        "layer_0.oracle": np.array([1.0]),
    }
    worker = {"cells": [{
        "layer": 0, "collateral_cosines": [1.0] * 48,
        "distinct_recurrent_state_conditions": 3,
    }]}
    try:
        score_payload(worker, tensors)
    except ValueError as error:
        assert "collateral cosine" in str(error)
    else:
        raise AssertionError("incomplete collateral evidence was accepted")
