from __future__ import annotations

from tools.research import run_fleet_mbppplus as experiment


def test_frozen_subset_is_deterministic_complete_and_unique():
    subset = experiment.load_subset()
    ids = [row["task_id"] for row in subset]
    assert len(ids) == len(set(ids)) == 100
    assert experiment.canonical_json_sha256(ids) == experiment.SUBSET_HASH


def test_request_contract_is_greedy_bounded_and_route_explicit():
    problem = experiment.load_subset()[0]
    request = experiment.payload("hauhaucs", problem)
    assert request["model"] == "hauhaucs"
    assert request["temperature"] == 0.0
    assert request["top_k"] == 1
    assert request["max_tokens"] == 768
    assert request["chat_template_kwargs"] == {"enable_thinking": False}


def test_paired_bootstrap_preserves_clear_direction():
    ids = [row["task_id"] for row in experiment.load_subset()]
    all_fail = [{"task_id": task_id} for task_id in ids]
    interval = experiment.paired_bootstrap({"failures": []}, {"failures": all_fail})
    assert interval["point"] == 1.0
    assert interval["lower_95"] == 1.0
    assert interval["upper_95"] == 1.0
