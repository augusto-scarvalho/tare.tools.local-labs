#!/usr/bin/env python3
"""Run a frozen sequence of watched experiments with fail-closed resume."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "tools/analysis/launch_watched_experiment.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import canonical_json_sha256


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def backlog_item(task_id: str) -> dict[str, Any] | None:
    manifest = read_json(ROOT / "config/research_backlog.json")
    items = manifest.get("items", [])
    return next(
        (item for item in items if isinstance(item, dict) and item.get("id") == task_id),
        None,
    )


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != "local-labs-watched-wave-v1":
        raise ValueError("unsupported wave schema")
    if not isinstance(manifest.get("wave_id"), str) or not manifest["wave_id"].strip():
        raise ValueError("wave_id must be a non-empty string")
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("wave items must be a non-empty list")
    task_ids = [item.get("task_id") for item in items if isinstance(item, dict)]
    if (
        len(task_ids) != len(items)
        or any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
        or len(set(task_ids)) != len(items)
    ):
        raise ValueError("wave task_id values must be unique strings")
    if type(manifest.get("poll_seconds", 300)) is not int or manifest.get("poll_seconds", 300) <= 0:
        raise ValueError("poll_seconds must be a positive integer")
    for item in items:
        if not isinstance(item.get("command"), list) or not item["command"]:
            raise ValueError(f"{item.get('task_id')}: command must be a non-empty argv list")
        if not all(isinstance(part, str) and part for part in item["command"]):
            raise ValueError(f"{item['task_id']}: command contains an invalid argv element")
        if type(item.get("max_runtime_seconds", 86_400)) is not int or item.get("max_runtime_seconds", 86_400) <= 0:
            raise ValueError(f"{item['task_id']}: max_runtime_seconds must be positive")
        terminal = bool(item.get("require_harness_terminal", False))
        progress = item.get("progress")
        if terminal and progress is not None:
            raise ValueError(f"{item['task_id']}: choose harness terminal or legacy progress")
        if not terminal:
            if not isinstance(progress, dict):
                raise ValueError(f"{item['task_id']}: legacy item requires progress")
            if progress.get("mode") not in {"files", "jsonl_lines"}:
                raise ValueError(f"{item['task_id']}: unsupported progress mode")
            if not isinstance(progress.get("glob"), str) or not progress["glob"]:
                raise ValueError(f"{item['task_id']}: progress glob is required")
            if type(progress.get("expected")) is not int or progress["expected"] <= 0:
                raise ValueError(f"{item['task_id']}: expected progress must be positive")


def packet_state(task_id: str) -> tuple[pathlib.Path, str, str]:
    item = backlog_item(task_id)
    if item is None or not isinstance(item.get("packet_dir"), str):
        raise ValueError(f"{task_id}: not registered in canonical backlog")
    packet_dir = (ROOT / item["packet_dir"]).resolve()
    packet = read_json(packet_dir / "PIPELINE.json")
    if packet.get("task_id") != task_id:
        raise ValueError(f"{task_id}: packet identity mismatch")
    return packet_dir, str(item.get("state")), str(packet.get("stage"))


def watcher_final(path: pathlib.Path, task_id: str) -> dict[str, Any]:
    final = read_json(path)
    if final.get("schema") != "local-labs-experiment-watch-final-v1":
        raise ValueError(f"{task_id}: watcher final schema mismatch")
    if final.get("status") != "complete":
        raise ValueError(f"{task_id}: watcher status is {final.get('status')!r}")
    if task_id not in final.get("audit_ready_ids", []):
        raise ValueError(f"{task_id}: watcher did not mark the packet audit-ready")
    return final


def launcher_argv(
    manifest: dict[str, Any], item: dict[str, Any], packet_dir: pathlib.Path, watch_dir: pathlib.Path
) -> list[str]:
    task_id = item["task_id"]
    argv = [
        sys.executable, str(LAUNCHER),
        "--task-id", task_id,
        "--packet-dir", str(packet_dir),
        "--watch-id", f"{manifest['wave_id']}-{task_id}",
        "--watch-outdir", str(watch_dir),
        "--poll-seconds", str(int(manifest.get("poll_seconds", 300))),
        "--max-runtime-seconds", str(int(item.get("max_runtime_seconds", 86_400))),
    ]
    if item.get("require_harness_terminal"):
        argv.append("--require-harness-terminal")
    else:
        progress = item["progress"]
        argv.extend([
            "--progress-glob", progress["glob"],
            "--progress-mode", progress["mode"],
            "--expected-progress", str(progress["expected"]),
        ])
    return [*argv, "--", *item["command"]]


def run_launcher(argv: list[str], stdout_path: pathlib.Path, stderr_path: pathlib.Path) -> int:
    with stdout_path.open("a", encoding="utf-8", buffering=1) as stdout, stderr_path.open(
        "a", encoding="utf-8", buffering=1
    ) as stderr:
        completed = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    return completed.returncode


def initial_state(manifest: dict[str, Any], digest: str) -> dict[str, Any]:
    return {
        "schema": "local-labs-watched-wave-state-v1",
        "wave_id": manifest["wave_id"],
        "manifest_sha256": digest,
        "status": "ready",
        "started_at": utc_now(),
        "current_task": None,
        "completed": [],
    }


def execute(manifest_path: pathlib.Path, outdir: pathlib.Path) -> int:
    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    digest = canonical_json_sha256(manifest)
    outdir.mkdir(parents=True, exist_ok=True)
    state_path = outdir / "STATE.json"
    if state_path.is_file():
        state = read_json(state_path)
        if state.get("schema") != "local-labs-watched-wave-state-v1":
            raise ValueError("wave state schema mismatch")
        if state.get("wave_id") != manifest["wave_id"] or state.get("manifest_sha256") != digest:
            raise ValueError("wave state is bound to a different manifest")
    else:
        state = initial_state(manifest, digest)
        write_json(state_path, state)

    completed_ids = {
        row["task_id"] for row in state.get("completed", []) if isinstance(row, dict)
    }
    try:
        for index, item in enumerate(manifest["items"]):
            task_id = item["task_id"]
            watch_dir = outdir / "watchers" / f"{index:03d}-{task_id}"
            final_path = watch_dir / "FINAL.json"
            packet_dir, manifest_stage, packet_stage_value = packet_state(task_id)
            if task_id in completed_ids:
                if manifest_stage != "EXECUTED" or packet_stage_value != "EXECUTED":
                    raise RuntimeError(f"{task_id}: completed wave item is no longer EXECUTED")
                watcher_final(final_path, task_id)
                continue
            if manifest_stage == "EXECUTED" and packet_stage_value == "EXECUTED":
                # Recover the narrow crash window after the watcher committed
                # EXECUTED but before this supervisor persisted completion.
                watcher_final(final_path, task_id)
                state["completed"].append({
                    "task_id": task_id,
                    "finished_at": utc_now(),
                    "final": str(final_path),
                    "recovered": True,
                })
                completed_ids.add(task_id)
                write_json(state_path, state)
                continue
            if manifest_stage != "IMPLEMENTED" or packet_stage_value != "IMPLEMENTED":
                raise RuntimeError(
                    f"{task_id}: wave requires IMPLEMENTED, found {manifest_stage}/{packet_stage_value}"
                )
            state.update({"status": "running", "current_task": task_id, "current_index": index})
            write_json(state_path, state)
            watch_dir.mkdir(parents=True, exist_ok=True)
            argv = launcher_argv(manifest, item, packet_dir, watch_dir)
            returncode = run_launcher(argv, outdir / f"{index:03d}.stdout.log", outdir / f"{index:03d}.stderr.log")
            if returncode != 0:
                raise RuntimeError(f"{task_id}: watched launcher exited {returncode}")
            watcher_final(final_path, task_id)
            _, manifest_stage, packet_stage_value = packet_state(task_id)
            if manifest_stage != "EXECUTED" or packet_stage_value != "EXECUTED":
                raise RuntimeError(f"{task_id}: watcher completed without EXECUTED state")
            state["completed"].append({
                "task_id": task_id,
                "finished_at": utc_now(),
                "final": str(final_path),
            })
            completed_ids.add(task_id)
            write_json(state_path, state)
        state.update({"status": "complete", "current_task": None, "finished_at": utc_now()})
        write_json(state_path, state)
        write_json(outdir / "FINAL.json", {
            "schema": "local-labs-watched-wave-final-v1",
            "wave_id": manifest["wave_id"],
            "status": "complete",
            "finished_at": state["finished_at"],
            "completed_ids": [row["task_id"] for row in state["completed"]],
        })
        print(json.dumps({"event": "wave_completed", "wave_id": manifest["wave_id"], "completed": len(completed_ids)}, separators=(",", ":")), flush=True)
        return 0
    except Exception as error:  # noqa: BLE001 - durable stop record is the contract
        state.update({
            "status": "stopped_fail_closed",
            "error": f"{type(error).__name__}: {error}",
            "finished_at": utc_now(),
        })
        write_json(state_path, state)
        print(state["error"], file=sys.stderr, flush=True)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--outdir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    return execute(args.manifest.resolve(), args.outdir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
