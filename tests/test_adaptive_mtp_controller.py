from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "analysis"))

from adaptive_mtp_controller import AdaptiveMTPController, simulate_engine


class TestAdaptiveMTPController(unittest.TestCase):
    def test_high_acceptance_expands_depth(self):
        controller = AdaptiveMTPController(window_size=8, gamma=0.15, max_k=4)
        for _ in range(10):
            controller.record_step(accepted_tokens=4, drafted_tokens=4)
        self.assertGreaterEqual(controller.get_current_acceptance_rate(), 0.8)
        self.assertEqual(controller.get_recommended_depth(), 4)

    def test_low_acceptance_throttles_depth(self):
        controller = AdaptiveMTPController(window_size=8, gamma=0.15, max_k=4)
        for _ in range(10):
            controller.record_step(accepted_tokens=0, drafted_tokens=4)
        self.assertLess(controller.get_current_acceptance_rate(), 0.3)
        self.assertEqual(controller.get_recommended_depth(), 0)

    def test_simulation_determinism(self):
        regimes = [(0.8, 50), (0.2, 50)]
        res1 = simulate_engine("ADAPTIVE", regimes, seed=123)
        res2 = simulate_engine("ADAPTIVE", regimes, seed=123)
        self.assertEqual(res1["effective_throughput"], res2["effective_throughput"])
        self.assertEqual(res1["total_tokens_emitted"], res2["total_tokens_emitted"])


if __name__ == "__main__":
    unittest.main()
