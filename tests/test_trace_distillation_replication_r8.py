from __future__ import annotations

from tools.research import run_trace_distillation_replication_r8 as r8


def test_second_panel_is_frozen_disjoint_and_source_bound():
    first, second = r8.panel_ids()
    static, checkpoints = r8.verify_sources()
    assert len(first) == len(second) == 256
    assert set(first).isdisjoint(second)
    assert r8.canonical_json_sha256(first) == r8.FIRST_PANEL_HASH
    assert r8.canonical_json_sha256(second) == r8.SECOND_PANEL_HASH
    assert len(static) == 10
    assert len(checkpoints) == 14


def test_hierarchical_bootstrap_preserves_clear_direction():
    positive = [[1] * 256 for _ in range(7)]
    interval = r8.hierarchical_bootstrap(positive, replicates=100)
    assert interval["lower_95"] == 1.0
    assert interval["upper_95"] == 1.0


def test_exact_seed_sign_flip_and_binomial_tail():
    assert r8.exact_sign_flip_pvalue([1.0] * 7) == 1 / 128
    assert r8.binomial_upper_tail(7, 7) == 1 / 128


def test_imported_r7_qa_recomputes_without_mismatch():
    regression, match, rows = r8.imported_qa_regression()
    assert match is True
    assert len(rows) == 7
    assert regression <= 0.05
