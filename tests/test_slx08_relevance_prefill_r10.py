from __future__ import annotations

import hashlib

from tools.research import run_slx08_relevance_prefill_r10 as runner


def test_every_frozen_r10_source_digest_matches_disk():
    for relative, expected in runner.SOURCE_HASHES.items():
        assert hashlib.sha256((runner.ROOT / relative).read_bytes()).hexdigest() == expected


def test_r10_ledger_contains_delegated_r6_failure_keys():
    assert "runs/autonomous/SLX08-RELEVANCE-R6-2026-08-30/FINAL.json" in runner.SOURCE_HASHES
    assert "runs/autonomous/SLX08-RELEVANCE-R6-2026-08-30/WORKER_EXIT.json" in runner.SOURCE_HASHES
