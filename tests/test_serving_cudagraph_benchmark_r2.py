from __future__ import annotations

from tools.research.run_serving_cudagraph_benchmark_r2 import (
    BLOCKS,
    calculate_metrics,
    parse_properties,
    percentile,
    prompt_for,
    semantic_projection,
)


def test_abba_layout_pairs_every_prompt_once_per_treatment():
    assert [block["treatment"] for block in BLOCKS] == ["off", "on", "on", "off"]
    matrix = {(prompt_id, block["treatment"]) for block in BLOCKS for prompt_id in block["prompt_ids"]}
    assert matrix == {(prompt_id, treatment) for prompt_id in range(1, 31) for treatment in ("off", "on")}


def test_calculate_metrics_uses_true_treatments_and_semantics():
    samples = []
    for prompt_id in range(1, 31):
        semantic = {"content": str(prompt_id), "reasoning_content": "r", "finish_reason": "length", "completion_tokens": 64}
        samples.append({**semantic, "prompt_id": prompt_id, "block_id": "off", "treatment": "off", "wall_ms": 120.0})
        samples.append({**semantic, "prompt_id": prompt_id, "block_id": "on", "treatment": "on", "wall_ms": 100.0})
    metrics, pairs = calculate_metrics(samples)
    assert len(pairs) == 30
    assert metrics["response_mismatch_rate"] == 0
    assert metrics["paired_wall_speedup_p50"] == 1.2
    assert metrics["on_vs_off_p95_regression"] < 0


def test_semantic_projection_excludes_timing():
    left = {"content": "x", "reasoning_content": "y", "finish_reason": "stop", "completion_tokens": 7, "wall_ms": 10}
    right = {**left, "wall_ms": 99}
    assert semantic_projection(left) == semantic_projection(right)


def test_percentile_and_properties_helpers():
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([1, 2, 3, 4, 5], 0.95) == 5
    assert parse_properties("MainPID=123\nActiveState=active\n") == {"MainPID": "123", "ActiveState": "active"}


def test_prompts_are_deterministic_and_distinct():
    assert prompt_for(1) == prompt_for(1)
    assert len({prompt_for(index) for index in range(1, 31)}) == 30
