from __future__ import annotations

from tools.research import run_fleet_humanevalplus as experiment


def test_full_panel_is_frozen_complete_and_unique():
    panel = experiment.load_panel()
    ids = [row["task_id"] for row in panel]
    assert len(ids) == len(set(ids)) == 164
    assert experiment.canonical_json_sha256(ids) == experiment.PANEL_HASH


def test_request_contract_is_route_explicit_and_greedy():
    request = experiment.payload("hauhaucs", experiment.load_panel()[0])
    assert request["model"] == "hauhaucs"
    assert request["temperature"] == 0.0
    assert request["top_k"] == 1
    assert request["max_tokens"] == 768
    assert request["chat_template_kwargs"] == {"enable_thinking": False}


def test_paired_bootstrap_preserves_clear_direction():
    ids = [row["task_id"] for row in experiment.load_panel()]
    result = experiment.paired_bootstrap(
        {task_id: True for task_id in ids},
        {task_id: False for task_id in ids},
    )
    assert result["point"] == 1.0
    assert result["lower_95"] == 1.0
    assert result["upper_95"] == 1.0
