from tools.research.run_gdn02_learned_state_r2 import gate_rows


def test_gate_fixture_separates_evidence_completeness_from_science():
    metrics = {
        "learned_gdn_layer_cells": 3,
        "retained_decisive_layer_cells": 3,
        "retained_collateral_cosines": 147,
        "recomputed_metric_match_rate": 1.0,
        "median_old_fact_leakage_pct": 30.0,
        "median_collateral_retention_pct": 95.0,
        "median_updated_fact_fidelity_pct": 70.0,
        "distinct_recurrent_state_conditions": 3,
    }
    result = gate_rows(metrics)
    assert result["retained_cells"]["pass"] is True
    assert result["retained_collateral"]["pass"] is True
    assert result["independent_recompute"]["pass"] is True
    assert result["target_leakage"]["pass"] is False
    assert result["update_fidelity"]["pass"] is False
