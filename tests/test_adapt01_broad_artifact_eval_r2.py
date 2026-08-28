from tools.research.run_adapt01_broad_artifact_eval_r2 import heldout_ids, paired_bootstrap, verify_sources


def test_sources_and_teacher_disjoint_panel_are_frozen():
    assert len(verify_sources()) == 8
    ids = heldout_ids()
    assert len(ids) == 256
    assert len(set(ids)) == 256


def test_paired_bootstrap_preserves_clear_effect():
    interval = paired_bootstrap([1] * 256, replicates=100)
    assert interval["lower_95"] == 1.0
    assert interval["upper_95"] == 1.0
