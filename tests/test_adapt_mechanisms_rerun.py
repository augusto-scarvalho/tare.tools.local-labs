from __future__ import annotations

from tools.research.run_adapt_mechanisms_rerun import summary_from_arm


def test_summary_projection_is_fail_closed() -> None:
    arm = {"summary": {"target_correct": 3, "target_n": 32, "protected_pass": 4, "protected_n": 16, "natural_eos": 48, "generation_n": 48, "ignored": 99}}
    assert summary_from_arm(arm) == {"target_correct": 3, "target_n": 32, "protected_pass": 4, "protected_n": 16, "natural_eos": 48, "generation_n": 48}
