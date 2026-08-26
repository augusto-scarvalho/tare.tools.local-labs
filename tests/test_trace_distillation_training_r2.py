from __future__ import annotations

from tools.research.run_trace_distillation_training_r2 import (
    ARM_ORDERS,
    SEEDS,
    build_training_manifest,
    normalize_exec_start,
)


def test_manifest_has_matched_distinct_targets_and_no_holdout_leakage():
    manifest = build_training_manifest()
    assert manifest["eligible_count"] == 168
    holdout = set(manifest["heldout_ids"])
    for seed in SEEDS:
        rows = manifest["seeds"][str(seed)]
        assert len(rows) == 128
        assert len({row["task_id"] for row in rows}) == 128
        assert not ({row["task_id"] for row in rows} & holdout)
        assert all(row["answer_only"] != row["full_trace"] for row in rows)


def test_seed_arm_order_is_balanced_and_complete():
    assert set(ARM_ORDERS) == set(SEEDS)
    assert ARM_ORDERS[20260824] == ["answer_only", "full_trace"]
    assert ARM_ORDERS[20260825] == ["full_trace", "answer_only"]
    assert ARM_ORDERS[20260826] == ["answer_only", "full_trace"]
    assert all(set(order) == {"answer_only", "full_trace"} for order in ARM_ORDERS.values())


def test_exec_start_normalization_removes_only_volatile_suffix():
    left = "{ path=/bin/x ; argv[]=/bin/x --a ; ignore_errors=no ; start_time=one }"
    right = "{ path=/bin/x ; argv[]=/bin/x --a ; ignore_errors=no ; start_time=two }"
    assert normalize_exec_start(left) == normalize_exec_start(right)
