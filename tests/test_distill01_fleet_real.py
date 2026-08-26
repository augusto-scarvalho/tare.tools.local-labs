import json

from tools.research.run_distill01_fleet_real import ARMS, SOURCE, score


def test_clean_real_routing_recomputes_expected_scores():
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    selected = [row for row in rows if row["arm"] in ARMS]
    rescored, scores = score(selected)
    assert len(rescored) == 144
    assert scores["fleet_math_correct"] == 10
    assert scores["fleet_qa_correct"] == 5
    assert scores["fleet_total"] == 15
    assert scores["monolith_total"] == 13
    assert scores["fleet_gain_over_monolith"] < 0.20


def test_stored_flags_match_independent_rescore():
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    selected = [row for row in rows if row["arm"] in ARMS]
    rescored, _ = score(selected)
    assert all(row["correct"] == row["recomputed_correct"] for row in rescored)
