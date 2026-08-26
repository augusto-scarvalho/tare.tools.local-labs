from __future__ import annotations

import copy

from tools.research.run_adapter_requalification_r2 import (
    EXPECTED_WORKERS,
    SMOKE_ORDER_A,
    SMOKE_ORDER_B,
    compare_smoke,
    semantic_samples,
    verify_artifact_ledger,
    verify_frozen_inputs,
)


def _payload(arm: str) -> dict:
    return {
        "arm": arm,
        "samples": [
            {
                "panel": "math",
                "task_id": "gsm8k/1",
                "prompt": "one plus one",
                "gold": "2",
                "extracted": "2",
                "correct": True,
                "output_text": "#### 2",
                "new_tokens": 3,
                "natural_eos": True,
                "elapsed_s": 99.0,
            }
        ],
    }


def test_smoke_contract_and_worker_count():
    assert SMOKE_ORDER_B == list(reversed(SMOKE_ORDER_A))
    assert EXPECTED_WORKERS == len(SMOKE_ORDER_A) + len(SMOKE_ORDER_B) + 11


def test_semantic_projection_ignores_timing_and_detects_output_drift():
    order_a = {arm: _payload(arm) for arm in SMOKE_ORDER_A}
    order_b = copy.deepcopy(order_a)
    for payload in order_b.values():
        payload["samples"][0]["elapsed_s"] = 1.0
    assert compare_smoke(order_a, order_b)["order_invariant"] is True

    order_b["target_mlp_only"]["samples"][0]["output_text"] = "#### 3"
    assert compare_smoke(order_a, order_b)["order_invariant"] is False


def test_semantic_projection_contains_decisive_fields():
    projected = semantic_samples(_payload("base"))[0]
    assert "elapsed_s" not in projected
    assert projected["output_text"] == "#### 2"
    assert projected["correct"] is True


def test_frozen_repo_and_adapter_ledgers_match_current_files():
    assert len(verify_frozen_inputs()) == 6
    ledger, paths = verify_artifact_ledger()
    assert len(ledger) == 13
    assert len(paths) == 26
