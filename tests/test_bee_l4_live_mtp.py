from __future__ import annotations

from tools.research.run_bee_l4_live_mtp import WORDS, score


def test_score_detects_repeat_nonce_and_rejection() -> None:
    rows=[]
    for round_index in range(2):
        for slot,word in WORDS.items():
            rows.append({"round":round_index,"slot":slot,"word":word,"content":word,"draft_n":4,"draft_n_accepted":2,"wall_latency_ms":1.0})
    metrics=score(rows)
    assert metrics["exact_repeat_rate"]==1.0
    assert metrics["own_nonce_rate"]==1.0
    assert metrics["cross_slot_leakage_count"]==0
    assert metrics["requests_with_rejected_draft_tokens"]==8
