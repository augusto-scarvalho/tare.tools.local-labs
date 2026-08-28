from __future__ import annotations

from tools.research.run_trace_distillation_confirmation_r5 import (
    MATH_IDS,
    QA_IDS,
    SEEDS,
    TRAINING_EXAMPLES,
    TRAINING_STEPS,
    _training_rows,
    build_training_manifest,
    exact_sign_flip_pvalue,
    hierarchical_bootstrap,
    verify_inputs,
)


def test_frozen_inputs_and_manifest_are_disjoint_and_complete():
    assert len(verify_inputs()) == 5
    manifest = build_training_manifest()
    training_ids = {row["task_id"] for row in manifest["pool"]}
    assert len(training_ids) == TRAINING_EXAMPLES == 168
    assert len(MATH_IDS) == 256
    assert len(QA_IDS) == 48
    assert training_ids.isdisjoint(MATH_IDS)
    for seed in SEEDS:
        rows = _training_rows(manifest, seed)
        assert len(rows) == TRAINING_STEPS == 504
        assert len({row["task_id"] for row in rows}) == TRAINING_EXAMPLES


def test_hierarchical_bootstrap_preserves_clear_paired_direction():
    differences = [[1] * len(MATH_IDS) for _ in SEEDS]
    interval = hierarchical_bootstrap(differences, replicates=100)
    assert interval["lower_95"] == 1.0
    assert interval["upper_95"] == 1.0


def test_sign_flip_is_exact_over_seven_seeds():
    assert exact_sign_flip_pvalue([1.0] * 7) == 1 / 128
