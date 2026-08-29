from __future__ import annotations

from tools.research import run_fleet_regression_screen_r2 as regression
from tools.research import run_fleet_seeded_stability_r2 as seeded


def test_seeded_request_enrichment_preserves_contract():
    row = {"model": "qwen38", "repeat": 0, "suite": "math", "case_id": "gsm8k-000"}
    # Use a real frozen case id so the fixture exercises the exact panel join.
    suite, case_id, _ = seeded.base.cases()[0]
    row.update({"suite": suite, "case_id": case_id})
    enriched = seeded.enrich_requests([row])[0]["request"]
    assert enriched["temperature"] == 0.2
    assert enriched["top_p"] == 0.95
    assert enriched["seed"] == 20260826
    assert enriched["model"] == "qwen38"


def test_successor_gate_is_fail_closed():
    assert regression._gate(448, 448)["pass"] is True
    assert regression._gate(447, 448)["pass"] is False
    assert seeded._gate(True, True)["pass"] is True
