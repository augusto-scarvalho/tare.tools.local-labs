import copy

from tools.research import run_adapt01_broad_artifact_eval_r3 as r3


def test_sources_and_actual_qa_panel_are_frozen():
    assert len(r3.verify_sources()) == 10
    ids = r3.actual_qa_ids()
    assert len(ids) == 48
    assert len(set(ids)) == 48


def test_merge_preserves_imported_math_and_adds_exact_missing_qa():
    qa_ids = r3.actual_qa_ids()
    source = {
        "arms": [],
        "base_preexisting_peft_module_count": 0,
    }
    fresh = {"arms": []}
    for arm_name in ("base", "lokr_3ep_lr1e4"):
        source["arms"].append({
            "arm": arm_name,
            "math_samples": [{"task_id": f"m{index}"} for index in range(256)],
            "qa_samples": [
                {"task_id": task_id, "correct": False}
                for task_id in qa_ids[:10]
            ],
        })
        fresh["arms"].append({
            "arm": arm_name,
            "qa_samples": [
                {"task_id": task_id, "correct": True}
                for task_id in qa_ids[10:]
            ],
        })

    source_before = copy.deepcopy(source)
    merged = r3.merge_payload(source, fresh, qa_ids)

    assert source == source_before
    for arm in merged["arms"]:
        assert len(arm["math_samples"]) == 256
        assert [sample["task_id"] for sample in arm["qa_samples"]] == qa_ids
        assert arm["qa_correct"] == 38
