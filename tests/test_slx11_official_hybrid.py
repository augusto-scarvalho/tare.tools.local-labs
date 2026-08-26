from __future__ import annotations


def test_hybrid_topology_classifier_is_exclusive() -> None:
    declared = ["linear_attention"] * 3 + ["full_attention"]
    actual = ["linear_attention"] * 3 + ["full_attention"]
    assert len(declared) == len(actual)
    assert sum(left == right for left, right in zip(declared, actual, strict=True)) == 4
    assert sum(value == "linear_attention" for value in actual) == 3
    assert sum(value == "full_attention" for value in actual) == 1
