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
    host_mode,
    load_registry,
    ninfer_chat_request,
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
            {"qwen38", "qwen38-ninfer", "qwen38-gsq", "swift27b", "swift-next", "qwen36-moe", "fable-tc", "hauhaucs",
             "gemma-vision", "muse-vision"},
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
        self.assertEqual(recommend(self.registry, "agent-tools")[0], "qwen38-gsq")

    def test_one_resident_model_is_fail_closed(self) -> None:
        invalid = copy.deepcopy(self.registry)
        invalid["fleet"]["max_resident_models"] = 2
        with self.assertRaisesRegex(FleetConfigError, "max_resident_models=1"):
            validate_registry(invalid)

    def test_hold_model_cannot_be_added(self) -> None:
        invalid = copy.deepcopy(self.registry)
        invalid["models"]["hauhaucs"]["qualification"] = "hold"
        with self.assertRaisesRegex(FleetConfigError, "promoted/qualified_role"):
            validate_registry(invalid)

    def test_gateway_owns_identity_and_network_flags(self) -> None:
        card = self.registry["models"]["hauhaucs"]
        command = build_backend_command("hauhaucs", card, host="127.0.0.1", port=18080)
        self.assertEqual(command[0], card["runtime"]["binary"])
        self.assertEqual(command[command.index("--alias") + 1], "hauhaucs")
        self.assertEqual(command[command.index("--port") + 1], "18080")
        self.assertEqual(command[command.index("-m") + 1], card["artifact"]["path"])

    def _ninfer_card(self) -> dict:
        card = copy.deepcopy(self.registry["models"]["hauhaucs"])
        card["runtime"] = {"kind": "ninfer", "binary": "/home/augus/opt/ninfer/v0.6.1-rtx3090/ninfer-serve",
                           "environment": {}, "args": ["--max-context", "32768", "--spec", "mtp"]}
        card["artifact"]["path"] = "/home/augus/models/qwen38-27b/ninfer/qwen3_8_27b.ninfer"
        return card

    def test_ninfer_backend_takes_artifact_positionally_and_gateway_owns_its_identity(self) -> None:
        registry = copy.deepcopy(self.registry)
        registry["models"]["qwen38-ninfer"] = self._ninfer_card()
        validate_registry(registry)
        command = build_backend_command("qwen38-ninfer", registry["models"]["qwen38-ninfer"],
                                        host="127.0.0.1", port=18080)
        self.assertEqual(command[1], "/home/augus/models/qwen38-27b/ninfer/qwen3_8_27b.ninfer")
        self.assertEqual(command[command.index("--model-id") + 1], "qwen38-ninfer")
        self.assertNotIn("-m", command)
        wrong = copy.deepcopy(registry)
        wrong["models"]["qwen38-ninfer"]["runtime"]["binary"] = "/home/augus/opt/slop/bin/llama-server"
        with self.assertRaisesRegex(FleetConfigError, "ninfer-serve"):
            validate_registry(wrong)
        owned = copy.deepcopy(registry)
        owned["models"]["qwen38-ninfer"]["runtime"]["args"] += ["--model-id", "other"]
        with self.assertRaisesRegex(FleetConfigError, "gateway-owned"):
            validate_registry(owned)

    def test_strata_backend_is_started_from_its_config_on_the_gateway_port(self) -> None:
        registry = copy.deepcopy(self.registry)
        card = copy.deepcopy(self.registry["models"]["hauhaucs"])
        card["artifact"]["path"] = "/mnt/wsl/models/flash-next/swift-IQ3_XXS/shard-1.gguf"
        card["runtime"] = {"kind": "strata", "binary": "/mnt/wsl/models/strata/.venv/bin/python",
                           "server": "/mnt/wsl/models/strata/serve/server.py",
                           "config": "/mnt/wsl/models/strata/strata-swift-iq3_xxs.json",
                           "model_name": "swift-1.5-iq3_xxs", "environment": {}, "args": [],
                           "idle_unload_seconds": 900}
        registry["models"]["swift-next"] = card
        validate_registry(registry)
        command = build_backend_command("swift-next", card, host="127.0.0.1", port=18080)
        self.assertEqual(command[1:3], ["/mnt/wsl/models/strata/serve/server.py", "--engine"])
        self.assertEqual(command[command.index("--port") + 1], "18080")
        self.assertEqual(command[command.index("--config") + 1], card["runtime"]["config"])
        for broken in ({"model_name": ""}, {"idle_unload_seconds": 10}, {"args": ["--port", "1"]}):
            bad = copy.deepcopy(registry)
            bad["models"]["swift-next"]["runtime"].update(broken)
            with self.assertRaises(FleetConfigError):
                validate_registry(bad)

    def test_solo_model_only_when_its_host_ram_is_free_now(self) -> None:
        modes = self.registry["fleet"]["host_modes"]
        now = 1_000_000
        needed = modes["solo_host_ram_gb"] + modes["ram_margin_gb"]
        fits = {"ts": now - 60, "windows_free_gb": needed, "desktop_idle_seconds": 5}  # someone at the desk is fine
        self.assertEqual(host_mode(self.registry, fits, now)["model"], "swift-next")
        # A resident solo model already holds its RAM: it keeps fitting instead of flapping.
        held = {**fits, "windows_free_gb": needed - modes["solo_host_ram_gb"]}
        self.assertEqual(host_mode(self.registry, held, now, "swift-next")["mode"], "solo")
        cases = {
            "host_ram_short": {**fits, "windows_free_gb": needed - 0.5},
            "host_status_stale": {**fits, "ts": now - modes["max_status_age_seconds"] - 1},
            "host_status_missing": None,
        }
        for reason, status in cases.items():
            result = host_mode(self.registry, status, now)
            self.assertEqual((result["mode"], result["model"], result["reason"]), ("shared", "qwen38-gsq", reason))
        broken = copy.deepcopy(self.registry)
        broken["fleet"]["host_modes"]["solo"] = "unknown"
        with self.assertRaisesRegex(FleetConfigError, "host_modes"):
            validate_registry(broken)

    def test_ninfer_request_moves_thinking_controls_to_top_level(self) -> None:
        body = {"model": "m", "temperature": 1.0, "reasoning_effort": "high",
                "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low", "other": 1}}
        out = ninfer_chat_request(body)
        self.assertEqual(out["enable_thinking"], True)
        self.assertEqual(out["reasoning_effort"], "high")  # an explicit top-level value wins
        self.assertEqual(out["chat_template_kwargs"], {"other": 1})
        self.assertEqual(ninfer_chat_request({"chat_template_kwargs": {"enable_thinking": False}}),
                         {"enable_thinking": False})
        self.assertIn("chat_template_kwargs", body)  # input is not mutated


if __name__ == "__main__":
    unittest.main()
