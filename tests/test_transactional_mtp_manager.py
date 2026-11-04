from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "analysis"))

from transactional_mtp_manager import TransactionalSpeculativeManager, run_concurrency_stress


class TestTransactionalSpeculativeManager(unittest.TestCase):
    def test_single_slot_commit_and_rollback(self):
        mgr = TransactionalSpeculativeManager(num_slots=2)
        # Step 1: full commit
        mgr.begin_step(0)
        mgr.append_draft(0, [10, 20])
        res1 = mgr.complete_step(0, accepted_count=2, total_drafted=2)
        self.assertEqual(res1, [10, 20])

        # Step 2: full rollback
        mgr.begin_step(0)
        mgr.append_draft(0, [30, 40])
        res2 = mgr.complete_step(0, accepted_count=0, total_drafted=2)
        self.assertEqual(res2, [10, 20])

        # Step 3: partial commit
        mgr.begin_step(0)
        mgr.append_draft(0, [50, 60, 70])
        res3 = mgr.complete_step(0, accepted_count=1, total_drafted=3)
        self.assertEqual(res3, [10, 20, 50])

    def test_cross_slot_independence(self):
        mgr = TransactionalSpeculativeManager(num_slots=2)
        mgr.begin_step(0)
        mgr.begin_step(1)

        mgr.append_draft(0, [1, 2])
        mgr.append_draft(1, [99, 100])

        res0 = mgr.complete_step(0, accepted_count=2, total_drafted=2)
        res1 = mgr.complete_step(1, accepted_count=0, total_drafted=2)

        self.assertEqual(res0, [1, 2])
        self.assertEqual(res1, [])

    def test_concurrency_stress_clean(self):
        res = run_concurrency_stress(num_slots=4, cycles_per_slot=100)
        self.assertEqual(res["stats"]["cross_slot_corruptions"], 0)
        self.assertEqual(res["stats"]["total_transactions"], 400)


if __name__ == "__main__":
    unittest.main()
