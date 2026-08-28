import json

from tools.research import run_adapt01_broad_artifact_eval_r2 as r2
from tools.research import run_adapt01_broad_artifact_eval_r4 as r4


def test_sources_and_second_panel_are_frozen_and_disjoint():
    assert len(r4.verify_sources()) == 10
    first = set(r2.heldout_ids())
    second = r4.second_panel_ids()
    assert len(second) == 256
    assert len(set(second)) == 256
    assert first.isdisjoint(second)


def test_r3_source_has_complete_ordered_qa_for_both_arms():
    source = json.loads(r4.SOURCE_WORKER.read_text(encoding="utf-8"))
    expected = r4.r3.actual_qa_ids()
    for arm in source["arms"]:
        assert [sample["task_id"] for sample in arm["qa_samples"]] == expected
