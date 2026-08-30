from tools.research.run_negative_kv_real_screen_r3 import evaluate_gates


def good_metrics():
    return {
        "blocked_r2_review_verified": True,
        "actual_model_activation_cells": 18,
        "actual_model_weight_matrices": 12,
        "candidate_hypotheses_evaluated": 5,
        "retained_candidate_cells": 78,
        "retained_decisive_tensor_files": 39,
        "retained_tensor_hashes_verified": True,
        "service_and_embedding_restored": True,
    }


def test_r3_retention_contract_passes_exact_complete_bundle():
    assert all(row["pass"] for row in evaluate_gates(good_metrics()).values())


def test_r3_retention_contract_rejects_each_missing_fixture():
    fixtures = {
        "activation_coverage": ("actual_model_activation_cells", 17),
        "weight_coverage": ("actual_model_weight_matrices", 11),
        "candidate_coverage": ("candidate_hypotheses_evaluated", 4),
        "sample_retention": ("retained_candidate_cells", 77),
        "tensor_retention": ("retained_decisive_tensor_files", 38),
        "tensor_hashes": ("retained_tensor_hashes_verified", False),
        "service_recovery": ("service_and_embedding_restored", False),
    }
    for gate, (metric, value) in fixtures.items():
        metrics = good_metrics()
        metrics[metric] = value
        assert evaluate_gates(metrics)[gate]["pass"] is False


def test_r3_binds_review_and_disallows_scalar_only_replay():
    metrics = good_metrics()
    metrics["blocked_r2_review_verified"] = False
    assert evaluate_gates(metrics)["r2_audit_bound"]["pass"] is False
