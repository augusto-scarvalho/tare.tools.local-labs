from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "analysis"))

from multi_adapter_router import MultiAdapterFlightRouter, run_serving_simulation


class TestMultiAdapterFlightRouter(unittest.TestCase):
    def test_affinity_grouping(self):
        router = MultiAdapterFlightRouter(num_slots=4)
        router.assign_slot(0, 100, "adapter_A", [1, 2])
        router.assign_slot(1, 101, "adapter_B", [3, 4])
        router.assign_slot(2, 102, "adapter_A", [5, 6])
        router.assign_slot(3, 103, "adapter_B", [7, 8])

        res = router.dispatch_step_affinity()
        self.assertEqual(len(res["affinity_groups"]), 2)
        self.assertEqual(res["affinity_groups"]["adapter_A"], [0, 2])
        self.assertEqual(res["affinity_groups"]["adapter_B"], [1, 3])

    def test_simulation_zero_errors_and_high_reduction(self):
        res = run_serving_simulation(num_requests=50, seed=123)
        self.assertEqual(res["routing_errors"], 0)
        self.assertGreater(res["context_switch_reduction_pct"], 40.0)


if __name__ == "__main__":
    unittest.main()
