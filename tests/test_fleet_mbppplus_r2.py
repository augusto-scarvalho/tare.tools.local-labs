from __future__ import annotations

from tools.research import run_fleet_mbppplus_r2 as experiment


def test_second_panel_is_frozen_and_disjoint_from_r1():
    first = [row["task_id"] for row in experiment.r1.load_subset()]
    second = [row["task_id"] for row in experiment.second_subset()]
    assert len(first) == len(second) == 100
    assert set(first).isdisjoint(second)
    assert experiment.canonical_json_sha256(second) == experiment.SECOND_PANEL_HASH


def test_stratified_bootstrap_preserves_clear_direction():
    first = [row["task_id"] for row in experiment.r1.load_subset()]
    second = [row["task_id"] for row in experiment.second_subset()]
    all_fail_first = [{"task_id": task_id} for task_id in first]
    all_fail_second = [{"task_id": task_id} for task_id in second]
    result = experiment.stratified_bootstrap([
        (first, {"failures": []}, {"failures": all_fail_first}),
        (second, {"failures": []}, {"failures": all_fail_second}),
    ])
    assert result["point"] == 1.0
    assert result["lower_95"] == 1.0
    assert result["upper_95"] == 1.0
