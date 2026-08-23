import unittest
from model_lifecycle.agent_harness import (
    TaskContract,
    ContractDelta,
    test_baseline_non_weakening,
)


class TestContractLifecycle(unittest.TestCase):
    def setUp(self):
        self.base_contract = TaskContract(
            contract_id="task-123",
            version=1,
            objective="Implement login",
            constraints=("Use JWT", "No session cookies"),
            required_tests=("test_login_success", "test_login_fail"),
            status="OPEN",
            next_action="inspect",
        )

    def test_digest_determinism(self):
        """Verify that the digest is consistent for identical data."""
        d1 = self.base_contract.digest
        d2 = TaskContract(
            contract_id="task-123",
            version=1,
            objective="Implement login",
            constraints=("Use JWT", "No session cookies"),
            required_tests=("test_login_success", "test_login_fail"),
            status="OPEN",
            next_action="inspect",
        ).digest
        self.assertEqual(d1, d2)

    def test_valid_delta_version_increment(self):
        """Verify that applying a delta increments the version."""
        delta = ContractDelta(
            contract_id="task-123",
            base_digest=self.base_contract.digest,
            status="IN_PROGRESS",
        )
        new_contract = self.base_contract.apply(delta)
        self.assertEqual(new_contract.version, 2)
        self.assertEqual(new_contract.status, "IN_PROGRESS")

    def test_parent_digest_chaining(self):
        """Verify that the new contract's parent_digest is the old contract's digest."""
        delta = ContractDelta(
            contract_id="task-123",
            base_digest=self.base_contract.digest,
        )
        new_contract = self.base_contract.apply(delta)
        self.assertEqual(new_contract.parent_digest, self.base_contract.digest)

    def test_invariant_objective_constraints_required_tests(self):
        """Verify that objective, constraints, and required_tests remain unchanged by delta."""
        delta = ContractDelta(
            contract_id="task-123",
            base_digest=self.base_contract.digest,
            status="DONE",
        )
        new_contract = self.base_contract.apply(delta)
        self.assertEqual(new_contract.objective, self.base_contract.objective)
        self.assertEqual(new_contract.constraints, self.base_contract.constraints)
        self.assertEqual(new_contract.required_tests, self.base_contract.required_tests)

    def test_evidence_append_rather_than_replacement(self):
        """Verify that evidence is appended, not overwritten."""
        initial_evidence = ("ev1",)
        self.base_contract = TaskContract(
            contract_id="task-123",
            version=1,
            objective="X",
            constraints=(),
            required_tests=(),
            evidence=initial_evidence,
        )
        delta = ContractDelta(
            contract_id="task-123",
            base_digest=self.base_contract.digest,
            evidence_append=("ev2",),
        )
        new_contract = self.base_contract.apply(delta)
        self.assertEqual(new_contract.evidence, ("ev1", "ev2"))

    def test_stale_digest_rejection(self):
        """Verify that a delta with an incorrect base_digest raises a ValueError."""
        delta = ContractDelta(
            contract_id="task-123",
            base_digest="wrong_digest",
        )
        with self.assertRaisesRegex(ValueError, "stale delta"):
            self.base_contract.apply(delta)

    def test_cross_contract_rejection(self):
        """Verify that a delta targeting a different contract_id raises a ValueError."""
        delta = ContractDelta(
            contract_id="different-id",
            base_digest=self.base_contract.digest,
        )
        with self.assertRaisesRegex(ValueError, "delta targets a different contract"):
            self.base_contract.apply(delta)

    def test_status_next_action_preservation_and_update(self):
        """Verify status and next_action update when provided, and persist when not."""
        # Update both
        delta_both = ContractDelta(
            contract_id="task-123",
            base_digest=self.base_contract.digest,
            status="REVIEW",
            next_action="deploy",
        )
        new_contract = self.base_contract.apply(delta_both)
        self.assertEqual(new_contract.status, "REVIEW")
        self.assertEqual(new_contract.next_action, "deploy")

        # Preserve original (None in delta)
        delta_none = ContractDelta(
            contract_id="task-123",
            base_digest=new_contract.digest,
            status=None,
            next_action=None,
        )
        final_contract = new_contract.apply(delta_none)
        self.assertEqual(final_contract.status, "REVIEW")
        self.assertEqual(final_contract.next_action, "deploy")

    def test_missing_baseline_test_rejection(self):
        """Verify test_baseline_non_weakening fails if a required test is missing."""
        before = {"test_1": True, "test_2": True}
        after = {"test_1": True}  # test_2 is missing
        result = test_baseline_non_weakening(before, after, require_same_tests=True)
        self.assertFalse(result["pass"])
        self.assertIn("test_2", result["missing"])

    def test_passing_to_failing_regression_rejection(self):
        """Verify test_baseline_non_weakening fails if a test goes from True to False."""
        before = {"test_1": True}
        after = {"test_1": False}
        result = test_baseline_non_weakening(before, after)
        self.assertFalse(result["pass"])
        self.assertIn("test_1", result["regressions"])

    def test_harmless_test_additions(self):
        """Verify test_baseline_non_weakening passes when new tests are added."""
        before = {"test_1": True}
        after = {"test_1": True, "test_2": True}
        result = test_baseline_non_weakening(before, after, require_same_tests=True)
        self.assertTrue(result["pass"])
        self.assertEqual(result["additions"], ["test_2"])


if __name__ == "__main__":
    unittest.main()
