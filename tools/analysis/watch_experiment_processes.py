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

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_lifecycle.experiment_harness import TERMINAL_NAME, verify_run


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
        stat_path = pathlib.Path(f"/proc/{pid}/stat")
        try:
            stat = stat_path.read_text(encoding="utf-8")
            closing = stat.rfind(")")
            if closing >= 0 and stat[closing + 2:closing + 3] == "Z":
                return False
        except (OSError, UnicodeError):
            pass
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


def pipeline_packet(packet_dir: pathlib.Path) -> dict[str, Any] | None:
    path = packet_dir / "PIPELINE.json"
    if not path.is_file():
        return None
    try:
        packet = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return packet if isinstance(packet, dict) else None


def pipeline_stage(packet_dir: pathlib.Path) -> str | None:
    packet = pipeline_packet(packet_dir)
    return packet.get("stage") if packet is not None else None


def backlog_item(task_id: str) -> dict[str, Any] | None:
    path = ROOT / "config/research_backlog.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    items = manifest.get("items") if isinstance(manifest, dict) else None
    if not isinstance(items, list):
        return None
    return next(
        (
            item for item in items
            if isinstance(item, dict) and item.get("id") == task_id
        ),
        None,
    )


def harness_terminal(packet_dir: pathlib.Path) -> dict[str, Any]:
    """Return a compact verification result; legacy packets have ``present=False``."""
    raw_dir = packet_dir / "raw"
    terminal_path = raw_dir / TERMINAL_NAME
    if not terminal_path.is_file():
        return {"present": False, "valid": None, "status": None, "errors": []}
    try:
        report = verify_run(raw_dir)
    except Exception as error:  # noqa: BLE001 - verification must fail closed
        return {
            "present": True,
            "valid": False,
            "status": None,
            "task_id": None,
            "sample_count": None,
            "errors": [f"terminal verification crashed: {type(error).__name__}: {error}"],
        }
    return {
        "present": True,
        "valid": report["valid"],
        "status": report["status"],
        "task_id": report.get("task_id"),
        "sample_count": report.get("sample_count"),
        "errors": report["errors"],
    }


def worker_exit_status(item: dict[str, Any]) -> dict[str, Any]:
    configured = item.get("worker_exit_path")
    if not configured:
        return {"present": False, "valid": None, "returncode": None, "errors": []}
    path = pathlib.Path(configured)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        return {"present": False, "valid": None, "returncode": None, "errors": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "present": True,
            "valid": False,
            "returncode": None,
            "errors": [f"worker exit receipt unavailable: {error}"],
        }
    errors = []
    if not isinstance(payload, dict):
        errors.append("worker exit receipt is not an object")
        payload = {}
    if payload.get("schema") != "local-labs-worker-exit-v1":
        errors.append("worker exit receipt schema mismatch")
    if payload.get("task_id") != item.get("task_id"):
        errors.append("worker exit receipt task_id mismatch")
    if payload.get("pid") != item.get("pid"):
        errors.append("worker exit receipt pid mismatch")
    if payload.get("run_id") != item.get("run_id"):
        errors.append("worker exit receipt run_id mismatch")
    if type(payload.get("returncode")) is not int:
        errors.append("worker exit receipt lacks an integer returncode")
    return {
        "present": True,
        "valid": not errors,
        "returncode": payload.get("returncode"),
        "timed_out": bool(payload.get("timed_out", False)),
        "errors": errors,
    }


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


def backlog_queue_snapshot(*, rebalance: bool = False, actor: str = "watcher") -> dict[str, Any]:
    rebalance_result = None
    rebalance_report = None
    rebalance_valid = True
    if rebalance:
        rebalance_result = run_pipeline(
            "rebalance", "--apply", "--actor", actor, "--json"
        )
        parsed = parse_pipeline_json(rebalance_result)
        rebalance_valid = (
            rebalance_result["returncode"] == 0
            and isinstance(parsed, dict)
            and parsed.get("schema") == "local-labs-backlog-priority-report-v1"
        )
        if isinstance(parsed, dict):
            rebalance_report = {
                "mode": parsed.get("mode"),
                "policy_sha256": parsed.get("policy_sha256"),
                "assessed_count": parsed.get("assessed_count"),
                "change_count": parsed.get("change_count"),
                "applied_ids": parsed.get("applied_ids", []),
            }
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
        "valid": rebalance_valid and status_valid and next_valid,
        "priority_rebalance": {
            "requested": rebalance,
            "returncode": rebalance_result["returncode"] if rebalance_result else None,
            "valid": rebalance_valid if rebalance else None,
            "report": rebalance_report,
        },
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


def progress_value(packet_dir: pathlib.Path, item: dict[str, Any]) -> int:
    """Measure progress using an explicit contract instead of guessing from a path."""
    mode = item.get("progress_mode", "files")
    matches = sorted(packet_dir.glob(item["progress_glob"]))
    if mode == "files":
        return len(matches)
    if mode == "jsonl_lines":
        return sum(
            1
            for path in matches
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    raise ValueError(f"unsupported progress_mode: {mode!r}")


def finalize_experiment(item: dict[str, Any], actor: str) -> dict[str, Any]:
    packet_dir = (ROOT / item["packet_dir"]).resolve()
    receipt = packet_dir / "raw/receipt.json"
    result = packet_dir / "RESULT.md"
    packet = pipeline_packet(packet_dir)
    stage_before = packet.get("stage") if packet is not None else None
    terminal = harness_terminal(packet_dir)
    worker_exit = worker_exit_status(item)
    outcome: dict[str, Any] = {
        "at": utc_now(),
        "receipt_exists": receipt.is_file(),
        "result_exists": result.is_file(),
        "stage_before": stage_before,
        "progress": progress_value(packet_dir, item),
        "progress_mode": item.get("progress_mode", "files"),
        "expected_progress": item["expected_progress"],
        "harness_terminal": terminal,
        "worker_exit": worker_exit,
        "managed_backlog": bool(item.get("managed_backlog", True)),
    }
    managed_backlog = outcome["managed_backlog"]
    if managed_backlog:
        canonical = backlog_item(item["task_id"])
        if canonical is None or not isinstance(canonical.get("packet_dir"), str):
            outcome["status"] = "failed_noncanonical_packet"
            return outcome
        canonical_packet_dir = (ROOT / canonical["packet_dir"]).resolve()
        if packet_dir != canonical_packet_dir or canonical.get("state") != stage_before:
            outcome["status"] = "failed_noncanonical_packet"
            return outcome
    if packet is None or packet.get("task_id") != item["task_id"]:
        outcome["status"] = "failed_identity_mismatch"
        return outcome
    if item.get("require_worker_exit") and not worker_exit["present"]:
        outcome["status"] = "failed_no_worker_exit"
        return outcome
    if worker_exit["present"] and not worker_exit["valid"]:
        outcome["status"] = "failed_invalid_worker_exit"
        return outcome
    if worker_exit["present"] and worker_exit.get("timed_out"):
        outcome["status"] = "failed_worker_timeout"
        return outcome
    if worker_exit["present"] and worker_exit["returncode"] != 0:
        outcome["status"] = "failed_worker_exit"
        return outcome
    if item.get("require_harness_terminal") and not terminal["present"]:
        outcome["status"] = "failed_no_harness_terminal"
        return outcome
    if terminal["present"] and not terminal["valid"]:
        outcome["status"] = "failed_invalid_harness_terminal"
        return outcome
    if terminal["present"] and terminal["status"] == "ABORTED":
        outcome["status"] = "failed_harness_aborted"
        return outcome
    if terminal["present"] and terminal["status"] != "SEALED":
        outcome["status"] = "failed_invalid_harness_terminal"
        return outcome
    if terminal["present"] and terminal.get("task_id") != item["task_id"]:
        outcome["status"] = "failed_identity_mismatch"
        return outcome
    if not receipt.is_file():
        outcome["status"] = "failed_no_receipt"
        return outcome
    if not result.is_file():
        outcome["status"] = "failed_no_result"
        return outcome
    if not terminal["present"] and outcome["progress"] < outcome["expected_progress"]:
        outcome["status"] = "failed_incomplete_progress"
        return outcome
    if not managed_backlog:
        outcome["advance"] = {
            "returncode": 0 if stage_before == "EXECUTED" else 2,
            "stdout": "unmanaged canary; no backlog transition",
            "stderr": "" if stage_before == "EXECUTED" else "unmanaged canary must be pre-staged EXECUTED",
        }
    elif stage_before == "IMPLEMENTED":
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
        ("executed_valid" if managed_backlog else "completed_unmanaged")
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
        progress = progress_value(packet_dir, item)
        terminal = harness_terminal(packet_dir)
        worker_exit = worker_exit_status(item)
        experiments.append({
            "task_id": item["task_id"],
            "pid": item["pid"],
            "process_alive": process_alive(int(item["pid"])),
            "progress": progress,
            "progress_mode": item.get("progress_mode", "files"),
            "expected_progress": item["expected_progress"],
            "pipeline_stage": pipeline_stage(packet_dir),
            "watch_status": states[item["task_id"]]["status"],
            "harness_terminal": terminal,
            "worker_exit": worker_exit,
            "managed_backlog": bool(item.get("managed_backlog", True)),
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
        item["task_id"]: {
            "status": "watching",
            "was_seen_alive": process_alive(int(item["pid"])),
        }
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
            deadline = float(item.get("deadline_epoch", float("inf")))
            if row["worker_exit"]["present"]:
                state["finalization"] = finalize_experiment(item, config["actor"])
                state["status"] = state["finalization"]["status"]
            elif time.time() >= deadline:
                state["status"] = "failed_worker_timeout"
                state["finalization"] = {"at": utc_now(), "status": state["status"]}
            elif not row["process_alive"] and not item.get("require_worker_exit"):
                if not state["was_seen_alive"] and not row["harness_terminal"]["present"]:
                    state["status"] = "failed_pid_never_observed"
                    state["finalization"] = {"at": utc_now(), "status": state["status"]}
                else:
                    state["finalization"] = finalize_experiment(item, config["actor"])
                    state["status"] = state["finalization"]["status"]
            if state["status"] != "watching":
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
    all_valid = all(
        state["status"] in {"executed_valid", "completed_unmanaged"}
        for state in states.values()
    )
    experiment_mode = bool(config.get("experiment_mode", False))
    backlog_queue = backlog_queue_snapshot(
        rebalance=experiment_mode and all_valid,
        actor=config.get("actor", "watcher"),
    )
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
        "audit_ready_ids": sorted(
            task_id
            for task_id, state in states.items()
            if state["status"] == "executed_valid"
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
