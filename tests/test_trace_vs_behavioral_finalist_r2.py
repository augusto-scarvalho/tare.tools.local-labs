import pytest

from tools.research import run_trace_vs_behavioral_finalist_r2 as subject


def test_frozen_selected_trace_and_third_panel_match():
    ids, trace, _ = subject.load_frozen_inputs()
    assert len(ids) == 256
    assert trace["seed"] == 20260832
    assert trace["arm"] == "full_trace"
    assert [sample["task_id"] for sample in trace["math_samples"]] == ids


def test_frozen_qa_comparison_is_exactly_tied():
    _, _, qa = subject.load_frozen_inputs()
    trace = qa["trace_correct"] / qa["trace_total"]
    behavior = sum(row["correct"] / row["total"] for row in qa["behavioral"]) / 2
    assert trace == pytest.approx(behavior, abs=1e-15)
    assert [(row["correct"], row["total"]) for row in qa["behavioral"]] == [(12, 48), (10, 48)]


def test_hierarchical_bootstrap_identical_vectors_is_zero():
    values = [index % 2 for index in range(256)]
    result = subject.hierarchical_bootstrap(values, [values, values], replicates=200)
    assert result["lower_95"] == 0.0
    assert result["upper_95"] == 0.0


def test_all_frozen_hashes_verify():
    ledger = subject.verify_sources()
    assert len(ledger) == len(subject.EXPECTED_HASHES)
