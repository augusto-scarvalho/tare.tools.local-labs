from __future__ import annotations

from tools.research.run_trace_distillation_training_r3 import IMPORTED_LABELS, materiality, verify_sources
from tools.research.run_trace_distillation_training_r2 import build_training_manifest


def test_frozen_continuation_sources_match():
    ledger = verify_sources()
    assert len(IMPORTED_LABELS) == 4
    assert len(ledger) == 16


def test_materiality_requires_matched_order_and_distinct_targets():
    manifest = build_training_manifest()
    payloads = []
    for seed in (20260824, 20260825, 20260826):
        ids = [row["task_id"] for row in manifest["seeds"][str(seed)]]
        payloads.extend([
            {"seed": seed, "arm": "answer_only", "training_task_ids": ids, "training_target_sha256": f"a{seed}"},
            {"seed": seed, "arm": "full_trace", "training_task_ids": ids, "training_target_sha256": f"t{seed}"},
        ])
    verified, rows = materiality(payloads, manifest)
    assert verified
    assert len(rows) == 3
