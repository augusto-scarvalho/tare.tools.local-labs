from __future__ import annotations

from tools.research.run_ctrl01_receipt_recovery import rescore


def test_rescore_reconstructs_and_recomputes() -> None:
    rows = []
    for index in range(24):
        rows.append({"kind": "real_model", "raw_content": "{}", "filtered": "{}", "token_pieces": ["{}"], "decisions": [{"piece": "{}", "accepted": True}], "us_per_token": 2.0})
    for index in range(12):
        rows.append({"kind": "valid_control", "raw_content": "[]", "filtered": "[]", "token_pieces": ["[]"], "decisions": [{"piece": "[]", "accepted": True}], "us_per_token": 3.0})
    rescored, metrics = rescore(rows, False)
    assert len(rescored) == 36
    assert metrics["sanitized_complete_valid_rate"] == 1.0
    assert metrics["valid_token_acceptance_rate"] == 1.0
    assert metrics["logit_mask_runtime_integrated"] is False
