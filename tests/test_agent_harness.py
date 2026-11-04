"""Fail-closed tests for Track H harness primitives."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_lifecycle.agent_harness import (  # noqa: E402
    ContractDelta,
    TaskContract,
    build_evidence_pack,
    deterministic_maintainability_gate,
    test_baseline_non_weakening as baseline_non_weakening,
)


class ContractTests(unittest.TestCase):
    def test_delta_preserves_invariants_and_chains_digest(self):
        base = TaskContract("t1", 1, "fix cache", ("no delete",), ("pytest",))
        updated = base.apply(ContractDelta("t1", base.digest, ("cache.py:41",), "ACTIVE", "test"))
        self.assertEqual(updated.version, 2)
        self.assertEqual(updated.objective, base.objective)
        self.assertEqual(updated.constraints, base.constraints)
        self.assertEqual(updated.required_tests, base.required_tests)
        self.assertEqual(updated.parent_digest, base.digest)
        self.assertNotEqual(updated.digest, base.digest)

    def test_stale_or_cross_contract_delta_fails(self):
        base = TaskContract("t1", 1, "fix cache", (), ())
        with self.assertRaisesRegex(ValueError, "stale"):
            base.apply(ContractDelta("t1", "0" * 64))
        with self.assertRaisesRegex(ValueError, "different"):
            base.apply(ContractDelta("t2", base.digest))


class EvidenceTests(unittest.TestCase):
    def test_structural_pack_finds_expected_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            (root / "cache.py").write_text("def restore_slot():\n    return 'oracle cache restore'\n", encoding="utf-8")
            (root / "noise.md").write_text("# unrelated\nweather only\n", encoding="utf-8")
            pack = build_evidence_pack(root, "cache restore oracle")
            self.assertTrue(pack.chunks)
            self.assertEqual(pack.chunks[0].path, "cache.py")
            self.assertIn("restore_slot", pack.chunks[0].text)


class BaselineTests(unittest.TestCase):
    def test_regression_and_missing_are_fail_closed(self):
        result = baseline_non_weakening({"a": True, "b": False}, {"a": False})
        self.assertFalse(result["pass"])
        self.assertEqual(result["regressions"], ["a"])
        self.assertEqual(result["missing"], ["b"])

    def test_addition_without_regression_passes(self):
        result = baseline_non_weakening({"a": True}, {"a": True, "new": True})
        self.assertTrue(result["pass"])
        self.assertEqual(result["additions"], ["new"])


class MaintainabilityTests(unittest.TestCase):
    def test_clean_deterministic_code_passes(self):
        source = "def normalize(value):\n    if value is None:\n        return ''\n    return str(value).strip()\n"
        self.assertTrue(deterministic_maintainability_gate(source)["pass"])

    def test_broad_swallow_sleep_random_and_marker_fail(self):
        source = """import random
import time
def retry():
    # TODO remove fallback
    try:
        time.sleep(1)
        return random.random()
    except Exception:
        pass
"""
        result = deterministic_maintainability_gate(source)
        self.assertFalse(result["pass"])
        kinds = {row["kind"] for row in result["violations"]}
        self.assertTrue({
            "unseeded_random_dependency", "blocking_sleep", "broad_exception",
            "swallowed_exception", "unfinished_marker",
        }.issubset(kinds))


if __name__ == "__main__":
    unittest.main()
