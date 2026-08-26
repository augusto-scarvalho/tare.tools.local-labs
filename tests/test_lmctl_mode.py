"""LAB-OPS-001 fail-closed SERVE/LAB state-lock qualification."""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "benchmarks"))
import lmctl  # noqa: E402


class ModeLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.temp.name) / "mode.json"

    def tearDown(self):
        self.temp.cleanup()

    def test_uninitialized_fails_closed(self):
        with self.assertRaisesRegex(lmctl.ModeLockError, "UNINITIALIZED"):
            lmctl._read_mode_state(self.path)

    def test_corrupt_state_fails_closed(self):
        self.path.write_text("not json", encoding="utf-8")
        with self.assertRaisesRegex(lmctl.ModeLockError, "corrupt"):
            lmctl._read_mode_state(self.path)

    def test_atomic_initialize_and_compare_and_set(self):
        first = lmctl._write_mode_state(
            "SERVE", owner="qa", reason="canonical baseline", expect="UNINITIALIZED",
            path=self.path,
        )
        self.assertEqual(first["mode"], "SERVE")
        second = lmctl._write_mode_state(
            "LAB", owner="qa", reason="isolated experiment", expect="SERVE", path=self.path,
        )
        self.assertEqual(second["previous_mode"], "SERVE")
        self.assertEqual(lmctl._read_mode_state(self.path)["mode"], "LAB")
        self.assertFalse(self.path.with_suffix(".json.lock").exists())

    def test_compare_and_set_mismatch_preserves_state(self):
        lmctl._write_mode_state("SERVE", owner="qa", reason="baseline", path=self.path)
        with self.assertRaisesRegex(lmctl.ModeLockError, "compare-and-set"):
            lmctl._write_mode_state(
                "LAB", owner="qa", reason="wrong expectation", expect="LAB", path=self.path,
            )
        self.assertEqual(lmctl._read_mode_state(self.path)["mode"], "SERVE")

    def test_missing_reason_is_rejected(self):
        with self.assertRaisesRegex(lmctl.ModeLockError, "reason"):
            lmctl._write_mode_state("SERVE", owner="qa", reason="", path=self.path)

    def test_transition_requires_no_server_for_lab(self):
        with self.assertRaisesRegex(lmctl.ModeLockError, "cannot enter LAB"):
            lmctl._validate_transition("LAB", [8080])
        lmctl._validate_transition("LAB", [])
        lmctl._validate_transition("LAB", [8081])

    def test_serve_transition_rejects_experimental_port(self):
        with self.assertRaisesRegex(lmctl.ModeLockError, "cannot enter SERVE"):
            lmctl._validate_transition("SERVE", [8092])
        lmctl._validate_transition("SERVE", [8080])

    def test_required_mode_maps_8080_to_serve_and_other_ports_to_lab(self):
        lmctl._write_mode_state("SERVE", owner="qa", reason="baseline", path=self.path)
        with mock.patch.object(lmctl, "_server_ports", return_value=[]):
            self.assertEqual(lmctl._require_mode_for_port(8080, self.path)["mode"], "SERVE")
            with self.assertRaisesRegex(lmctl.ModeLockError, "requires LAB"):
                lmctl._require_mode_for_port(8092, self.path)
        lmctl._write_mode_state("LAB", owner="qa", reason="campaign", path=self.path)
        with mock.patch.object(lmctl, "_server_ports", return_value=[]):
            self.assertEqual(lmctl._require_mode_for_port(8092, self.path)["mode"], "LAB")

    def test_mode_drift_is_fail_closed(self):
        self.assertTrue(lmctl._runtime_drift("SERVE", [8092]))
        self.assertTrue(lmctl._runtime_drift("LAB", [8080]))
        self.assertTrue(lmctl._runtime_drift("LAB", [8091, 8092]))
        self.assertFalse(lmctl._runtime_drift("SERVE", [8080]))
        self.assertFalse(lmctl._runtime_drift("SERVE", [8081, 8080]))
        self.assertFalse(lmctl._runtime_drift("LAB", [8092]))
        self.assertFalse(lmctl._runtime_drift("LAB", [8081, 8092]))

    def test_router_backend_normalizes_to_canonical_8080(self):
        completed = mock.Mock(
            stdout="10 llama-server --port 8081\n11 llama-server --port 18080\n"
        )
        with mock.patch.object(lmctl, "_wsl", return_value=completed), \
             mock.patch.object(lmctl, "_qualified_gateway_live", return_value=True):
            self.assertEqual(lmctl._server_ports(), [8081, 8080])

    def test_orphan_router_backend_remains_drift(self):
        completed = mock.Mock(stdout="11 llama-server --port 18080\n")
        with mock.patch.object(lmctl, "_wsl", return_value=completed), \
             mock.patch.object(lmctl, "_qualified_gateway_live", return_value=False):
            ports = lmctl._server_ports()
        self.assertEqual(ports, [18080])
        self.assertTrue(lmctl._runtime_drift("SERVE", ports))

    def test_existing_writer_lock_is_not_stolen(self):
        lock = self.path.with_suffix(".json.lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({"pid": 123}), encoding="utf-8")
        with self.assertRaisesRegex(lmctl.ModeLockError, "already locked"):
            lmctl._write_mode_state("SERVE", owner="qa", reason="baseline", path=self.path)
        self.assertTrue(lock.exists())


if __name__ == "__main__":
    unittest.main()
