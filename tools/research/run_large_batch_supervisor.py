#!/usr/bin/env python3
"""Persistent fail-closed supervisor for an authorized local experiment batch."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "config/autonomous_experiment_batch_2026-08-26.json"


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}"],
        capture_output=True, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return result.returncode == 0


def http_ok(url: str) -> bool:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "LocalLabs-Batch-Supervisor/1.0"})
        with urllib.request.urlopen(request, timeout=15.0) as response:
            body = json.loads(response.read().decode("utf-8", "replace"))
            return response.status == 200 and body.get("status") == "ok"
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False


def packet_stage(task_id: str) -> str:
    packet = ROOT / "runs/research" / task_id / "PIPELINE.json"
    return str(load_json(packet).get("stage")) if packet.is_file() else "MISSING"


def task_state(task_id: str) -> dict[str, Any]:
    path = ROOT / "runs/research" / task_id / "raw/runner_state.json"
    return load_json(path) if path.is_file() else {}


def restored(state: dict[str, Any]) -> bool:
    restoration = state.get("restoration") or {}
    if "error" in restoration:
        return False
    if "status" in restoration:
        return restoration["status"].get("backend_healthy") is True and restoration.get("embedding_health") == 200
    if "initial_model_restored" in restoration:
        return restoration.get("initial_model_restored") is True and restoration.get("embedding", {}).get("http_status") == 200
    return False


def wait_monitor(item: dict[str, Any], poll_seconds: int, state_path: pathlib.Path, batch_state: dict[str, Any]) -> None:
    task_id = item["task_id"]
    pid = int(item["pid"])
    while process_alive(pid):
        state = task_state(task_id)
        batch_state.update({"status": "monitoring", "current_task": task_id, "current_pid": pid, "task_progress": state})
        write_json(state_path, batch_state)
        print(f"monitor {task_id}: {state.get('completed_requests', state.get('completed_observations', 0))}", flush=True)
        time.sleep(poll_seconds)


def validate_completed(task_id: str) -> dict[str, Any]:
    state = task_state(task_id)
    checks = {
        "runner_completed": state.get("status") == "completed",
        "packet_executed": packet_stage(task_id) == "EXECUTED",
        "restored": restored(state),
        "gateway_health": http_ok("http://127.0.0.1:8080/health"),
        "embedding_health": http_ok("http://127.0.0.1:8081/health"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"post-task stop condition for {task_id}: {checks}; state={state}")
    return checks


def run_item(item: dict[str, Any], batch_dir: pathlib.Path, state_path: pathlib.Path, batch_state: dict[str, Any]) -> int:
    task_id = item["task_id"]
    logs = batch_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{task_id}.stdout.log"
    stderr_path = logs / f"{task_id}.stderr.log"
    argv = [sys.executable, *item["command"]]
    with stdout_path.open("a", encoding="utf-8", buffering=1) as stdout, stderr_path.open("a", encoding="utf-8", buffering=1) as stderr:
        process = subprocess.Popen(
            argv, cwd=ROOT, stdout=stdout, stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        batch_state.update({"status": "running", "current_task": task_id, "current_pid": process.pid, "command": argv})
        write_json(state_path, batch_state)
        print(f"started {task_id} pid={process.pid}", flush=True)
        return process.wait()


def execute(manifest_path: pathlib.Path, batch_dir: pathlib.Path) -> int:
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "local-labs-autonomous-batch-v1":
        raise ValueError("unsupported batch manifest")
    poll_seconds = int(manifest.get("poll_seconds", 10))
    state_path = batch_dir / "state.json"
    batch_state: dict[str, Any] = {
        "batch_id": manifest["batch_id"], "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"), "completed_tasks": [],
    }
    write_json(state_path, batch_state)
    try:
        for item in manifest["queue"]:
            task_id = item["task_id"]
            if item["mode"] == "monitor":
                wait_monitor(item, poll_seconds, state_path, batch_state)
            elif item["mode"] == "run":
                returncode = run_item(item, batch_dir, state_path, batch_state)
                if returncode != 0:
                    raise RuntimeError(f"runner exited nonzero for {task_id}: {returncode}")
            else:
                raise ValueError(f"unsupported queue mode: {item['mode']}")
            checks = validate_completed(task_id)
            batch_state["completed_tasks"].append({"task_id": task_id, "checks": checks})
            write_json(state_path, batch_state)
            print(f"completed and validated {task_id}", flush=True)
        batch_state.update({"status": "completed", "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "current_task": None, "current_pid": None})
        write_json(state_path, batch_state)
        return 0
    except Exception as exc:
        batch_state.update({"status": "stopped_fail_closed", "error": f"{type(exc).__name__}: {exc}", "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        write_json(state_path, batch_state)
        print(batch_state["error"], file=sys.stderr, flush=True)
        return 1


def selfcheck() -> None:
    manifest = load_json(DEFAULT_MANIFEST)
    assert manifest["schema"] == "local-labs-autonomous-batch-v1"
    assert len(manifest["queue"]) >= 2
    assert manifest["queue"][0]["mode"] == "monitor"
    assert manifest["queue"][1]["mode"] == "run"
    print("large batch supervisor self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--batch-dir", type=pathlib.Path, default=ROOT / "runs/autonomous/LARGE-BATCH-2026-08-26")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    return execute(args.manifest.resolve(), args.batch_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
