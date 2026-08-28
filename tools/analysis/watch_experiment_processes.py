#!/usr/bin/env python3
"""Persistently watch experiment PIDs and fail-closed finalize valid receipts."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "tools/analysis/backlog_pipeline.py"
STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_event(path: pathlib.Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def http_status(url: str) -> int | None:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status
    except Exception:
        return None


def gpu_state() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        return {"error": completed.stderr.strip(), "returncode": completed.returncode}
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(values) != 5:
        return {"raw": completed.stdout.strip()}
    return {
        "memory_used_mib": int(values[0]),
        "memory_free_mib": int(values[1]),
        "utilization_percent": int(values[2]),
        "temperature_c": int(values[3]),
        "power_w": float(values[4]),
    }


def pipeline_stage(packet_dir: pathlib.Path) -> str | None:
    path = packet_dir / "PIPELINE.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("stage")


def run_pipeline(*arguments: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(PIPELINE), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def parse_pipeline_json(result: dict[str, Any]) -> Any:
    if result["returncode"] != 0:
        return None
    try:
        return json.loads(result["stdout"])
    except (TypeError, json.JSONDecodeError):
        return None


def backlog_queue_snapshot() -> dict[str, Any]:
    status_result = run_pipeline("status", "--json")
    next_result = run_pipeline("next", "--json")
    items = parse_pipeline_json(status_result)
    next_candidate = parse_pipeline_json(next_result)
    counts = Counter(
        item.get("state")
        for item in items
        if isinstance(item, dict) and item.get("state")
    ) if isinstance(items, list) else Counter()
    status_valid = status_result["returncode"] == 0 and isinstance(items, list)
    next_valid = next_result["returncode"] == 0 and (
        isinstance(next_candidate, dict) or next_result["stdout"].strip() == "null"
    )
    return {
        "refreshed_at": utc_now(),
        "status_returncode": status_result["returncode"],
        "next_returncode": next_result["returncode"],
        "valid": status_valid and next_valid,
        "state_counts": dict(sorted(counts.items())),
        "next_candidate": next_candidate,
    }


def completion_action(experiment_mode: bool, final_status: str, next_candidate: Any) -> str:
    if final_status not in {"complete", "complete_with_alert"}:
        return "stop_fail_closed"
    if final_status == "complete_with_alert":
        return "inspect_alert_before_dispatch"
    if not experiment_mode:
        return "notify_completion"
    if not isinstance(next_candidate, dict):
        return "notify_queue_empty"
    return "dispatch_next_candidate"


def finalize_experiment(item: dict[str, Any], actor: str) -> dict[str, Any]:
    packet_dir = ROOT / item["packet_dir"]
    receipt = packet_dir / "raw/receipt.json"
    result = packet_dir / "RESULT.md"
    stage_before = pipeline_stage(packet_dir)
    outcome: dict[str, Any] = {
        "at": utc_now(),
        "receipt_exists": receipt.is_file(),
        "result_exists": result.is_file(),
        "stage_before": stage_before,
        "progress": len(list(packet_dir.glob(item["progress_glob"]))),
        "expected_progress": item["expected_progress"],
    }
    if not receipt.is_file():
        outcome["status"] = "failed_no_receipt"
        return outcome
    if outcome["progress"] < outcome["expected_progress"]:
        outcome["status"] = "failed_incomplete_progress"
        return outcome
    if stage_before == "IMPLEMENTED":
        outcome["advance"] = run_pipeline(
            "advance", item["task_id"], "--to", "EXECUTED", "--actor", actor
        )
    elif stage_before == "EXECUTED":
        outcome["advance"] = {"returncode": 0, "stdout": "already EXECUTED", "stderr": ""}
    else:
        outcome["advance"] = {
            "returncode": 2,
            "stdout": "",
            "stderr": f"unexpected pre-finalization stage {stage_before!r}",
        }
    outcome["stage_after"] = pipeline_stage(packet_dir)
    outcome["gate"] = run_pipeline("gate")
    outcome["status"] = (
        "executed_valid"
        if outcome["advance"]["returncode"] == 0
        and outcome["stage_after"] == "EXECUTED"
        and outcome["gate"]["returncode"] == 0
        else "failed_validation"
    )
    return outcome


def snapshot(config: dict[str, Any], states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    experiments = []
    for item in config["experiments"]:
        packet_dir = ROOT / item["packet_dir"]
        progress = len(list(packet_dir.glob(item["progress_glob"])))
        experiments.append({
            "task_id": item["task_id"],
            "pid": item["pid"],
            "process_alive": process_alive(int(item["pid"])),
            "progress": progress,
            "expected_progress": item["expected_progress"],
            "pipeline_stage": pipeline_stage(packet_dir),
            "watch_status": states[item["task_id"]]["status"],
        })
    return {
        "schema": "local-labs-experiment-watch-status-v1",
        "watch_id": config["watch_id"],
        "updated_at": utc_now(),
        "experiments": experiments,
        "gpu": gpu_state(),
        "health": {url: http_status(url) for url in config["final_health_urls"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=pathlib.Path)
    parser.add_argument("--outdir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    status_path = outdir / "WATCH_STATUS.json"
    event_path = outdir / "events.jsonl"
    final_path = outdir / "FINAL.json"
    states = {
        item["task_id"]: {"status": "watching", "was_seen_alive": process_alive(int(item["pid"]))}
        for item in config["experiments"]
    }
    append_event(event_path, {"at": utc_now(), "event": "watcher_started", "config": config, "states": states})
    previous_signature = None
    poll_seconds = max(5, int(config.get("poll_seconds", 15)))
    while any(state["status"] == "watching" for state in states.values()):
        current = snapshot(config, states)
        signature = [(row["task_id"], row["process_alive"], row["progress"], row["pipeline_stage"]) for row in current["experiments"]]
        if signature != previous_signature:
            append_event(event_path, {"at": utc_now(), "event": "progress", "experiments": current["experiments"], "gpu": current["gpu"]})
            previous_signature = signature
        for item, row in zip(config["experiments"], current["experiments"]):
            state = states[item["task_id"]]
            if state["status"] != "watching":
                continue
            state["was_seen_alive"] = state["was_seen_alive"] or row["process_alive"]
            if not row["process_alive"]:
                if not state["was_seen_alive"]:
                    state["status"] = "failed_pid_never_observed"
                    state["finalization"] = {"at": utc_now(), "status": state["status"]}
                else:
                    state["finalization"] = finalize_experiment(item, config["actor"])
                    state["status"] = state["finalization"]["status"]
                append_event(event_path, {
                    "at": utc_now(), "event": "experiment_finished",
                    "task_id": item["task_id"], "state": state,
                })
        write_json(status_path, snapshot(config, states) | {"states": states})
        if any(state["status"] == "watching" for state in states.values()):
            time.sleep(poll_seconds)

    settle_deadline = time.monotonic() + max(0, int(config.get("service_settle_seconds", 180)))
    health = {url: http_status(url) for url in config["final_health_urls"]}
    while any(value != 200 for value in health.values()) and time.monotonic() < settle_deadline:
        time.sleep(5)
        health = {url: http_status(url) for url in config["final_health_urls"]}
    gate = run_pipeline("gate")
    all_valid = all(state["status"] == "executed_valid" for state in states.values())
    backlog_queue = backlog_queue_snapshot()
    experiment_mode = bool(config.get("experiment_mode", False))
    final_status = (
        "complete"
        if all_valid
        and gate["returncode"] == 0
        and all(value == 200 for value in health.values())
        and (not experiment_mode or backlog_queue["valid"])
        else "complete_with_alert"
    )
    final = {
        "schema": "local-labs-experiment-watch-final-v1",
        "watch_id": config["watch_id"],
        "finished_at": utc_now(),
        "status": final_status,
        "experiment_mode": experiment_mode,
        "states": states,
        "pipeline_gate": gate,
        "backlog_queue": backlog_queue,
        "completion_action": completion_action(
            experiment_mode,
            final_status,
            backlog_queue["next_candidate"],
        ),
        "final_health": health,
        "final_gpu": gpu_state(),
    }
    write_json(final_path, final)
    write_json(status_path, snapshot(config, states) | {"states": states, "final": final})
    append_event(event_path, {"at": utc_now(), "event": "watcher_finished", "final": final})
    return 0 if final["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
