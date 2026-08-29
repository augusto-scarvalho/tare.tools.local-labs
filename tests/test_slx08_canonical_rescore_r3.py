import math
from pathlib import Path

import pytest

from tools.research.slx08_canonical_scorer import (
    canonical_cosine,
    expected_keys,
    summarize,
)


@pytest.fixture
def source_samples():
    return [
        {"cell": f"context_{context}_layer_{layer}"}
        for context in range(2)
        for layer in (3, 7, 11, 15, 19, 23)
    ]


@pytest.fixture
def canonical_rows(source_samples):
    return [
        {
            "cell": row["cell"],
            "tensor_hash_match": True,
            "nonfinite_values": 0,
            "selected_block_context_cosine": 0.96 + index / 1000,
            "legacy_first_half_context_cosine": 0.94 + index / 1000,
        }
        for index, row in enumerate(source_samples)
    ]


def test_canonical_cosine_known_vectors():
    assert canonical_cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert canonical_cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert canonical_cosine([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.0)


@pytest.mark.parametrize("left,right", [([], []), ([1.0], [1.0, 2.0]), ([0.0], [0.0])])
def test_canonical_cosine_fails_closed_on_invalid_inputs(left, right):
    with pytest.raises(ValueError):
        canonical_cosine(left, right)


def test_expected_key_contract_is_36_exact_tensors(source_samples):
    keys = expected_keys(source_samples)
    assert len(keys) == 36
    assert "context_0_layer_3_dense" in keys
    assert "context_1_layer_23_legacy" in keys


def test_summary_exposes_hash_and_nonfinite_failures(canonical_rows):
    canonical_rows[0]["tensor_hash_match"] = False
    canonical_rows[1]["nonfinite_values"] = 2
    result = summarize(canonical_rows, 36)
    assert result["retained_context_tensors"] == 36
    assert result["retained_context_cells"] == 12
    assert result["tensor_hash_match_rate"] == 11 / 12
    assert result["nonfinite_values"] == 2


def test_summary_uses_full_frozen_population(canonical_rows):
    result = summarize(canonical_rows, 36)
    assert result["canonical_median_selected_block_context_cosine"] == pytest.approx(0.9655)
    assert result["canonical_median_legacy_first_half_context_cosine"] == pytest.approx(0.9455)


def test_runner_is_retained_only_and_materializes_result_before_receipt():
    source = Path("tools/research/run_slx08_canonical_rescore_r3.py").read_text(encoding="utf-8")
    assert "systemctl(" not in source
    assert "wait_for_health(" not in source
    assert '"inference_requests_issued": 0' in source
    assert source.index('(outdir / "RESULT.md").write_text') < source.index('write_json(raw / "receipt.json", receipt)')
