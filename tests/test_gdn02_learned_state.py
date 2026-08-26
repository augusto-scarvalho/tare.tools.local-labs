from __future__ import annotations

from tools.research.gdn02_learned_state_worker import NEW_VALUE, OLD_VALUE, TARGET_INDEX, build_conditions


def test_conditions_are_paired_and_targeted() -> None:
    records = [{"key": f"k{index}", "value": str(index)} for index in range(50)]
    records[TARGET_INDEX]["value"] = OLD_VALUE
    conditions = build_conditions(records)
    assert len(conditions["baseline"]) == 50
    assert len(conditions["treatment"]) == 50
    assert len(conditions["oracle"]) == 1
    assert f"answer {OLD_VALUE}" in conditions["baseline"][TARGET_INDEX]
    assert f"answer {NEW_VALUE}" in conditions["treatment"][TARGET_INDEX]
    assert conditions["oracle"][0].count(f"answer {NEW_VALUE}") >= 2
