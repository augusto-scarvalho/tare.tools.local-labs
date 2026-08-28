import pytest

from tools.research import run_adapt01_broad_artifact_eval_r2 as r2
from tools.research import run_adapt01_broad_artifact_eval_r4 as r4
from tools.research import run_adapt01_broad_artifact_eval_r5 as r5


def test_sources_and_third_panel_are_frozen_and_pairwise_disjoint():
    assert len(r5.verify_sources()) == 12
    panels = [set(r2.heldout_ids()), set(r4.second_panel_ids()), set(r5.third_panel_ids())]
    assert all(len(panel) == 256 for panel in panels)
    assert panels[0].isdisjoint(panels[1])
    assert panels[0].isdisjoint(panels[2])
    assert panels[1].isdisjoint(panels[2])


def test_stratified_bootstrap_preserves_clear_effect():
    interval = r5.stratified_bootstrap([[1] * 256 for _ in range(3)], replicates=100)
    assert interval["lower_95"] == 1.0
    assert interval["upper_95"] == 1.0


def test_stratified_bootstrap_rejects_wrong_dimensions():
    with pytest.raises(ValueError, match="three 256-task panels"):
        r5.stratified_bootstrap([[1] * 256 for _ in range(2)], replicates=10)
