#!/usr/bin/env python3
"""Command-only signature correction for the frozen duplicate-action guard."""
from __future__ import annotations

import json

import mini_swe_verified_duplicate_guard as base


class CommandOnlyDuplicateActionGuardAgent(base.DuplicateActionGuardAgent):
    """Ignore per-response tool-call IDs when identifying duplicate commands."""

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        commands = [action.get("command", "") for action in actions]
        signature = json.dumps(commands, separators=(",", ":")) if actions else None
        if signature is not None and signature == self._last_action_signature:
            self._identical_action_streak += 1
        else:
            self._last_action_signature = signature
            self._identical_action_streak = 1 if signature is not None else 0

        if actions and self._identical_action_streak >= 3:
            self._blocked_action_count += 1
            outputs = [
                {
                    "output": (
                        "LOOP_GUARD_BLOCKED: this exact command was already executed twice in succession. "
                        "Do not repeat it. Use a substantively different command, implement the best "
                        "evidence-backed change, or verify and submit the current patch."
                    ),
                    "returncode": 125,
                    "exception_info": "",
                }
                for _ in actions
            ]
            observations = self.model.format_observation_messages(
                message, outputs, self.get_template_vars(loop_guard_blocked=True)
            )
            return self.add_messages(*observations)
        return base.DefaultAgent.execute_actions(self, message)


base.DuplicateActionGuardAgent = CommandOnlyDuplicateActionGuardAgent


if __name__ == "__main__":
    raise SystemExit(base.main())
