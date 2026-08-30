from __future__ import annotations

import hashlib

from tools.research import run_slx08_relevance_prefill_r9 as runner


def test_every_frozen_r9_source_digest_matches_disk():
    for relative, expected in runner.SOURCE_HASHES.items():
        actual = hashlib.sha256((runner.ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected


def test_delegate_receives_r9_identity():
    runner.configure_delegate()
    assert runner.r8.TASK_ID == runner.TASK_ID
    assert runner.r8.PRE_REG_SHA256 == runner.PRE_REG_SHA256
    assert runner.r8.SOURCE_HASHES == runner.SOURCE_HASHES
