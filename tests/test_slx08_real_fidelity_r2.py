import math
from pathlib import Path

import pytest

from tools.research.slx08_context_scorer import (
    expected_keys,
    projection_match,
    summarize,
    validate_keyset,
)


@pytest.fixture
def retained_samples():
    return [
        {
            "cell": f"context_{context}_layer_{layer}",
            "selected_block_context_cosine": 0.96,
            "legacy_first_half_context_cosine": 0.95,
        }
        for context in range(2)
        for layer in (3, 7, 11, 15, 19, 23)
    ]


@pytest.fixture
def matching_evaluations(retained_samples):
    return [
        {
            "cell": row["cell"],
            "selected_block_context_cosine": 0.96 + index / 1000,
            "legacy_first_half_context_cosine": 0.94 + index / 1000,
            "nonfinite_values": 0,
            "projection_match": True,
        }
        for index, row in enumerate(retained_samples)
    ]


def test_expected_bundle_has_exactly_three_arms_per_cell(retained_samples):
    keys = expected_keys(retained_samples)
    assert len(keys) == 36
    assert "context_0_layer_3_dense" in keys
    assert "context_1_layer_23_corrected" in keys
    assert "context_1_layer_23_legacy" in keys


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_bundle_key_contract_fails_closed(retained_samples, mutation):
    keys = expected_keys(retained_samples)
    if mutation == "missing":
        keys.remove(next(iter(keys)))
    else:
        keys.add("context_9_layer_99_dense")
    with pytest.raises(ValueError, match="bundle key mismatch"):
        validate_keyset(keys, retained_samples)


def test_bundle_key_contract_accepts_exact_fixture(retained_samples):
    validate_keyset(expected_keys(retained_samples), retained_samples)


def test_projection_match_rejects_tampered_cosine():
    assert projection_match(0.995, 0.995 + 5e-10)
    assert not projection_match(0.995, 0.995 + 5e-6)


def test_summary_recomputes_coverage_medians_and_match_rate(matching_evaluations):
    matching_evaluations[-1]["projection_match"] = False
    result = summarize(matching_evaluations)
    assert result["retained_context_cells"] == 12
    assert result["recomputed_projection_match_rate"] == 11 / 12
    assert result["recomputed_median_selected_block_context_cosine"] > 0.96
    assert result["nonfinite_values"] == 0


def test_summary_surfaces_nonfinite_fixture(matching_evaluations):
    matching_evaluations[4]["nonfinite_values"] = 1
    result = summarize(matching_evaluations)
    assert result["nonfinite_values"] == 1


def test_worker_retains_vectors_and_runner_binds_bundle():
    worker = Path("tools/research/slx08_real_fidelity_worker_r2.py").read_text(encoding="utf-8")
    runner = Path("tools/research/run_slx08_real_fidelity_r2.py").read_text(encoding="utf-8")
    assert 'save_file(bundle, str(args.bundle)' in worker
    assert '"context_vectors.safetensors"' in runner
    assert '"context_evaluation.json"' in runner
    assert runner.index('raw / "context_vectors.safetensors"') < runner.index('write_json(raw / "receipt.json", receipt)')
