from __future__ import annotations

from tools.research import run_slx11_official_hybrid_r2 as runner
from tools.research import slx11_logits_scorer as scorer
from tools.research import slx11_official_hybrid_worker_r2 as worker


def topology_fixture() -> list[dict]:
    return (
        [{"match": True, "actual": "linear_attention"} for _ in range(18)]
        + [{"match": True, "actual": "full_attention"} for _ in range(6)]
    )


def test_layer_classifier_is_exclusive():
    assert worker.classify_layer(True, False) == "linear_attention"
    assert worker.classify_layer(False, True) == "full_attention"
    assert worker.classify_layer(True, True) == "ambiguous"
    assert worker.classify_layer(False, False) == "ambiguous"


def test_projection_match_requires_shape_argmax_and_hash():
    sample = {"logits_key": "k", "logits_shape": [1, 3], "argmax_token": 2, "logits_sha256": "a" * 64}
    projection = {"logits_key": "k", "shape": [1, 3], "argmax_token": 2, "logits_sha256": "a" * 64}
    assert scorer.projection_matches(sample, projection) is True
    for key, replacement in (("shape", [3]), ("argmax_token", 1), ("logits_sha256", "b" * 64)):
        mutated = dict(projection)
        mutated[key] = replacement
        assert scorer.projection_matches(sample, mutated) is False


def test_aggregate_uses_retained_scorer_metrics():
    metrics = runner.aggregate(
        {"model_type": "qwen3_5", "text_model_type": "qwen3_5_text", "physical_layers": topology_fixture(), "samples": [{}] * 24},
        {"retained_logits_tensors": 24, "recomputed_finite_output_rate": 1.0, "recomputed_projection_match_rate": 1.0},
        1,
    )
    assert metrics == {
        "official_checkpoint_identified": 1,
        "hybrid_layer_types_verified": 24,
        "physical_recurrent_layers": 18,
        "physical_full_attention_layers": 6,
        "successful_live_forwards": 24,
        "retained_logits_tensors": 24,
        "recomputed_finite_output_rate": 1.0,
        "recomputed_projection_match_rate": 1.0,
        "serving_process_unchanged": 1,
    }


def test_aggregate_fails_closed_on_scorer_or_service_failure():
    metrics = runner.aggregate(
        {"model_type": "qwen3_5", "text_model_type": "qwen3_5_text", "physical_layers": topology_fixture(), "samples": [{}] * 24},
        {"retained_logits_tensors": 23, "recomputed_finite_output_rate": 23 / 24, "recomputed_projection_match_rate": 23 / 24},
        0,
    )
    assert metrics["retained_logits_tensors"] == 23
    assert metrics["recomputed_finite_output_rate"] < 1.0
    assert metrics["recomputed_projection_match_rate"] < 1.0
    assert metrics["serving_process_unchanged"] == 0
