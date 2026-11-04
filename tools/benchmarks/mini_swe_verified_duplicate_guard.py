#!/usr/bin/env python3
"""Run the pinned SWE-bench pilot with an exact duplicate-action execution guard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from minisweagent.agents.default import DefaultAgent


DATASET = "SWE-bench/SWE-bench_Verified"
REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"


class DuplicateActionGuardAgent(DefaultAgent):
    """Block a third consecutive identical action batch without spending shell execution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_action_signature: str | None = None
        self._identical_action_streak = 0
        self._blocked_action_count = 0

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        signature = json.dumps(actions, sort_keys=True, separators=(",", ":")) if actions else None
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
                        "LOOP_GUARD_BLOCKED: this exact action was already executed twice in succession. "
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
        return super().execute_actions(message)

    def serialize(self, *extra_dicts) -> dict:
        return super().serialize(
            {"info": {"loop_guard": {"blocked_action_count": self._blocked_action_count}}},
            *extra_dicts,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset
    from minisweagent.config import get_config_from_spec
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import get_sb_environment
    from minisweagent.utils.serialize import recursive_merge

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["revision"] != REVISION or manifest["dataset"] != DATASET:
        raise RuntimeError("manifest dataset identity does not match the frozen pilot")
    selected_ids = manifest["future_pilot_instance_ids"]
    if args.limit:
        selected_ids = selected_ids[: args.limit]

    dataset = load_dataset(DATASET, split="test", revision=REVISION)
    if dataset._fingerprint != manifest["dataset_fingerprint"]:
        raise RuntimeError(
            f"dataset fingerprint drift: {dataset._fingerprint} != {manifest['dataset_fingerprint']}"
        )
    by_id = {row["instance_id"]: dict(row) for row in dataset}
    if any(instance_id not in by_id for instance_id in selected_ids):
        raise RuntimeError("one or more frozen pilot IDs are absent from the pinned dataset")

    config = recursive_merge(
        get_config_from_spec(args.base_config),
        get_config_from_spec(args.model_config),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output / "preds.json"
    predictions = json.loads(predictions_path.read_text()) if predictions_path.exists() else {}

    for instance_id in selected_ids:
        if instance_id in predictions and not args.redo:
            print(f"SKIP {instance_id}: prediction already exists", flush=True)
            continue
        instance = by_id[instance_id]
        instance_dir = args.output / instance_id
        trajectory_path = instance_dir / f"{instance_id}.traj.json"
        environment = None
        agent = None
        info = {"exit_status": "NotStarted", "submission": ""}
        print(f"START {instance_id}", flush=True)
        try:
            environment = get_sb_environment(config, instance)
            model = get_model(config=config["model"])
            agent = DuplicateActionGuardAgent(model, environment, **config["agent"])
            info = agent.run(instance["problem_statement"])
        except Exception as exc:
            info = {
                "exit_status": type(exc).__name__,
                "submission": "",
                "exception_str": str(exc),
            }
            print(f"ERROR {instance_id}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if agent is not None:
                agent.save(trajectory_path, {"instance_id": instance_id, "info": info})
            if environment is not None:
                environment.cleanup()

        submission = info.get("submission") or ""
        predictions[instance_id] = {
            "model_name_or_path": config["model"]["model_name"],
            "instance_id": instance_id,
            "model_patch": submission,
        }
        predictions_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
        blocked = agent._blocked_action_count if agent is not None else 0
        print(
            f"DONE {instance_id}: exit={info.get('exit_status')} "
            f"patch_bytes={len(submission.encode())} guard_blocks={blocked}",
            flush=True,
        )
        if info.get("exit_status") != "Submitted" or not submission.strip():
            print("FAIL_CLOSED: protocol qualification did not produce a submitted patch", flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
