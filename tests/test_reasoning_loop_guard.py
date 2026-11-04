from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "analysis"))

from reasoning_loop_guard import ReasoningLoopGuard, evaluate_guard


class TestReasoningLoopGuard(unittest.TestCase):
    def test_single_reversal_allowed(self):
        guard = ReasoningLoopGuard(window_size=32, max_reversals=3)
        tokens = ["Let's ", "calculate. ", "Wait, ", "we ", "need ", "to ", "add ", "5. ", "Total ", "is ", "15."]
        for t in tokens:
            triggered, reason = guard.feed_token(t)
            self.assertFalse(triggered)

    def test_reversal_loop_triggers_cut(self):
        guard = ReasoningLoopGuard(window_size=32, max_reversals=3)
        tokens = ["Wait, ", "let ", "me ", "reconsider. ", "Wait, ", "let ", "me ", "check. ", "Wait, ", "hold ", "on. ", "Wait!"]
        triggered = False
        for t in tokens:
            cut, reason = guard.feed_token(t)
            if cut:
                triggered = True
                break
        self.assertTrue(triggered)

    def test_think_tag_closure_disables_guard(self):
        guard = ReasoningLoopGuard(window_size=32, max_reversals=2)
        guard.feed_token("</think>")
        for _ in range(10):
            cut, _ = guard.feed_token("Wait, ")
            self.assertFalse(cut)

    def test_evaluation_clean_pass(self):
        res = evaluate_guard()
        self.assertEqual(res["false_positives"], 0)
        self.assertEqual(res["true_positives"], 25)


if __name__ == "__main__":
    unittest.main()
