from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_lifecycle.qualified_fleet import (  # noqa: E402
    FleetConfigError,
    build_backend_command,
    load_registry,
    recommend,
    resolve_model,
    validate_registry,
)


class QualifiedFleetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = load_registry()

    def test_only_role_qualified_models_are_routable(self) -> None:
        self.assertEqual(
            set(self.registry["models"]),
            {"qwen38", "qwen36-moe", "fable-tc", "hauhaucs", "gemma-vision", "muse-vision"},
        )
        self.assertTrue(all(
            card["qualification"] in {"promoted", "qualified_role"}
            for card in self.registry["models"].values()
        ))

    def test_aliases_and_recommendations_are_deterministic(self) -> None:
        self.assertEqual(resolve_model(self.registry, "coding")[0], "hauhaucs")
        self.assertEqual(resolve_model(self.registry, "vision-hard")[0], "muse-vision")
        self.assertEqual(resolve_model(self.registry, "throughput")[0], "qwen36-moe")
        self.assertEqual(recommend(self.registry, "math")[0], "fable-tc")
        self.assertEqual(recommend(self.registry, "agent-tools")[0], "qwen38")

    def test_one_resident_model_is_fail_closed(self) -> None:
        invalid = copy.deepcopy(self.registry)
        invalid["fleet"]["max_resident_models"] = 2
        with self.assertRaisesRegex(FleetConfigError, "max_resident_models=1"):
            validate_registry(invalid)

    def test_hold_model_cannot_be_added(self) -> None:
        invalid = copy.deepcopy(self.registry)
        invalid["models"]["qwen38"]["qualification"] = "hold"
        with self.assertRaisesRegex(FleetConfigError, "promoted/qualified_role"):
            validate_registry(invalid)

    def test_gateway_owns_identity_and_network_flags(self) -> None:
        card = self.registry["models"]["qwen38"]
        command = build_backend_command("qwen38", card, host="127.0.0.1", port=18080)
        self.assertEqual(command[0], card["runtime"]["binary"])
        self.assertEqual(command[command.index("--alias") + 1], "qwen38")
        self.assertEqual(command[command.index("--port") + 1], "18080")
        self.assertEqual(command[command.index("-m") + 1], card["artifact"]["path"])


if __name__ == "__main__":
    unittest.main()
