from __future__ import annotations

import pathlib

import pytest

from tools.research import run_slx08_relevance_prefill_r6 as runner


def _block_texts(target: int, key: str = "R6CASE007") -> list[str]:
    blocks = ["instruction"] + [f"decoy block {index}" for index in range(1, 15)] + [f"query {key}"]
    blocks[target] = f"authoritative {key} record"
    return blocks


@pytest.mark.parametrize("target", range(1, 15))
def test_selector_retains_every_rotating_target_and_endpoints(target: int):
    selected = runner.select_relevant_blocks(_block_texts(target), "R6CASE007")
    assert len(selected) == 8
    assert selected == sorted(set(selected))
    assert selected[0] == 0
    assert selected[-1] == 15
    assert target in selected


def test_selector_fails_closed_outside_frozen_shape():
    with pytest.raises(ValueError):
        runner.select_relevant_blocks(["too short"], "key")
    with pytest.raises(ValueError):
        runner.select_relevant_blocks(_block_texts(3), "R6CASE007", retain=7)


@pytest.mark.parametrize(
    ("content", "expected", "correct"),
    [("7007", 7007, True), ("Answer: 7007.", 7007, True), ("17007", 7007, False), ("7008", 7007, False), ("", 7007, False)],
)
def test_exact_four_digit_scorer(content: str, expected: int, correct: bool):
    assert runner.answer_correct(content, expected) is correct


def _row(correct: bool, ttft_ms: float, arm: str, evidence: bool = True) -> dict:
    return {
        "correct": correct,
        "ttft_ms": ttft_ms,
        "route_observed": True,
        "evidence_retained": evidence,
        "telemetry": {"retained_attention_fraction": 1.0 if arm == "dense" else 0.5},
    }


def test_score_separates_selector_value_from_speed_and_route():
    pairs = []
    for case_id in range(64):
        pairs.append({
            "dense": _row(True, 1000.0, "dense"),
            "naive": _row(case_id % 2 == 0, 500.0, "naive", evidence=case_id % 2 == 0),
            "relevance": _row(True, 500.0, "relevance"),
        })
    metrics = runner.score(pairs, restored=True, embedding_status=200)
    gates = runner.evaluate_gates(metrics)
    assert metrics["dense_accuracy"] == 1.0
    assert metrics["naive_accuracy"] == 0.5
    assert metrics["relevance_accuracy"] == 1.0
    assert metrics["relevance_vs_naive_accuracy_delta"] == 0.5
    assert metrics["relevance_vs_dense_p50_ttft_speedup"] == 2.0
    assert all(gate["pass"] for gate in gates.values())


def test_request_only_sends_explicit_indices_for_relevance():
    fixture = {"tokens": [1] * 4096, "relevance_blocks": [0, 1, 3, 5, 7, 9, 11, 15]}
    assert "slx08_selected_block_indices" not in runner.request_for(fixture, "dense")
    assert "slx08_selected_block_indices" not in runner.request_for(fixture, "naive")
    assert runner.request_for(fixture, "relevance")["slx08_selected_block_indices"] == fixture["relevance_blocks"]


def test_cpp_contract_exposes_mode_indices_and_fail_closed_validation():
    source = pathlib.Path(r"C:\projects\slop.cpp\tools\server\server-context.cpp").read_text(encoding="utf-8")
    response = pathlib.Path(r"C:\projects\slop.cpp\tools\server\server-task.cpp").read_text(encoding="utf-8")
    assert "slx08_selected_block_indices" in source
    assert "must retain exactly half" in source
    assert "must be strictly increasing" in source
    assert "must retain the first and final" in source
    assert '"selection_mode"' in response
    assert '"selected_block_indices"' in response
