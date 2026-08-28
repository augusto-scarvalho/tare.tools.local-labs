from tools.research import run_trace_distillation_deploy_finalist as target


def test_seed_selection_is_frozen_and_reproducible():
    selected, rows = target.select_seed()
    assert selected == 20260832
    assert [(row["seed"], row["combined_trace_correct"]) for row in rows] == [
        (20260831, 220), (20260832, 223), (20260834, 210)]


def test_third_panel_is_disjoint_and_frozen():
    first, second, third = target.third_panel_ids()
    assert len(first) == len(second) == len(third) == 256
    assert not set(first) & set(second)
    assert not set(first) & set(third)
    assert not set(second) & set(third)
    assert target.canonical_json_sha256(third) == target.THIRD_PANEL_HASH


def test_paired_score_requires_both_arms():
    try:
        target.paired_score([], [f"gsm8k/{index}" for index in range(256)])
    except ValueError as error:
        assert "checkpoint pair" in str(error)
    else:
        raise AssertionError("incomplete pair was accepted")
