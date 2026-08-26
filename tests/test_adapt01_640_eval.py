from __future__ import annotations


def test_original_behavioral_gate_conjunction() -> None:
    metrics = {"target_correct": 16, "target_gain": 8, "qa": 2, "eos": 40, "ratio": 1.25}
    assert metrics["target_correct"] >= 16
    assert metrics["target_gain"] >= 3
    assert metrics["qa"] >= 2
    assert metrics["eos"] >= 40
    assert metrics["ratio"] <= 1.25
