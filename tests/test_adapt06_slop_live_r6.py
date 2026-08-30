from tools.research.run_adapt06_slop_live_r6 import derive_counterfactuals, gates


def test_counterfactual_fixture_requires_route_correct_cells():
    live = {
        "baselines": [
            {"route": "base", "index": 0, "content": "a"},
            {"route": "mlp", "index": 0, "content": "b"},
        ],
        "routed": [
            {"route": "base", "index": 0, "content": "a"},
            {"route": "mlp", "index": 0, "content": "wrong"},
        ],
        "cache": [{"route": "base"}],
        "schedule": {"alternating": [1, 2], "grouped": [1, 2]},
    }
    counterfactuals, bound = derive_counterfactuals(live)
    assert counterfactuals["route_correct_counterfactual_match_rate"] == 0.5
    assert counterfactuals["match_count"] == 1
    assert bound["cache_sequences"] == 1


def test_r6_gate_set_rejects_unbound_rows():
    metrics = {
        "converted_adapters": 2, "loaded_adapters": 2,
        "prompts_with_distinct_route_outputs": 4, "digest_bound_live_rows": False,
        "route_correct_counterfactual_match_rate": 1.0, "routed_exact_match_rate": 1.0,
        "cross_route_contamination_count": 0, "requested_route_switch_reduction": 0.9,
        "schedule_semantic_parity": 1.0, "original_service_restored": 1,
        "embedding_health": 200,
    }
    result = gates(metrics)
    assert result["bound_live_rows"]["pass"] is False
    assert all(row["pass"] for name, row in result.items() if name != "bound_live_rows")
