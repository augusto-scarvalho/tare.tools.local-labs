from __future__ import annotations

import pytest

from tools.research import run_slx08_physical_prefill_r4 as runner


@pytest.fixture
def token_fixture():
    intro = [1, 2, 3]
    filler = [40, 41, 42, 43]
    question = [90, 91, 92]
    return runner.build_fixture(7, intro, filler, question, 6)


def test_fixture_has_exact_even_blocks_and_boundaries(token_fixture):
    assert len(token_fixture["tokens"]) == 4096
    assert token_fixture["tokens"][:3] == [1, 2, 3]
    assert token_fixture["tokens"][-3:] == [90, 91, 92]
    assert len(token_fixture["prompt_sha256"]) == 64


def test_pad_tokens_fails_when_fixture_overflows():
    with pytest.raises(ValueError, match="do not fit"):
        runner.pad_tokens([1, 2], [3], 3, [4, 5])


@pytest.mark.parametrize(
    ("content", "expected", "correct"),
    [(" 7", 7, True), ("answer 4.", 4, True), ("14", 4, False), ("none", 0, False)],
)
def test_answer_correct_is_single_digit_bounded(content, expected, correct):
    assert runner.answer_correct(content, expected) is correct


def _pair(case_id: int, off_ms: float = 200.0, on_ms: float = 100.0):
    return {
        "case_id": case_id,
        "off": {"correct": True, "ttft_ms": off_ms},
        "on": {
            "correct": True,
            "ttft_ms": on_ms,
            "route_observed": True,
            "telemetry": {"retained_attention_fraction": 0.5},
        },
    }


def test_score_uses_all_64_physical_pairs_and_exact_fraction():
    metrics = runner.score([_pair(index) for index in range(64)], True, 200)
    assert metrics["physical_dense_prefill_requests"] == 64
    assert metrics["physical_selected_block_prefill_requests"] == 64
    assert metrics["selected_block_route_observation_rate"] == 1.0
    assert metrics["median_retained_attention_fraction"] == 0.5
    assert metrics["paired_accuracy_delta_ci95_low"] == 0.0
    assert metrics["paired_p50_ttft_speedup"] == 2.0
    assert all(gate["pass"] for gate in runner.evaluate_gates(metrics).values())


def test_tail_regression_fails_only_tail_gate():
    pairs = [_pair(index) for index in range(60)] + [
        _pair(index, off_ms=200.0, on_ms=10000.0) for index in range(60, 64)
    ]
    gates = runner.evaluate_gates(runner.score(pairs, True, 200))
    assert gates["tail_safety"]["pass"] is False
    assert gates["ttft_gain"]["pass"] is True
