from __future__ import annotations

import hashlib
import math

from tools.research import run_slx08_relevance_prefill_r11 as runner


def test_exact_zero_failure_bounds_reproduce_audit_and_clear_at_126():
    assert math.isclose(runner.exact_upper_failure_bound(0, 64), 1 - 0.05 ** (1 / 64), rel_tol=1e-12)
    assert runner.exact_upper_failure_bound(0, 64) > 0.03
    assert runner.exact_upper_failure_bound(0, 126) < 0.03


def test_exact_bound_rejects_invalid_counts():
    for failures, opportunities in ((-1, 10), (11, 10), (0, 0)):
        try:
            runner.exact_upper_failure_bound(failures, opportunities)
        except ValueError:
            pass
        else:
            raise AssertionError((failures, opportunities))


def test_every_frozen_r11_source_digest_matches_disk():
    for relative, expected in runner.SOURCE_HASHES.items():
        assert hashlib.sha256((runner.ROOT / relative).read_bytes()).hexdigest() == expected


def test_r11_balance_constants_are_exact():
    ids = range(runner.CASE_OFFSET, runner.CASE_OFFSET + runner.PAIRS)
    assert all(sum(case_id % 14 == position for case_id in ids) == 9 for position in range(14))
    assert all(sum(case_id % 3 == period for case_id in ids) == 42 for period in range(3))
