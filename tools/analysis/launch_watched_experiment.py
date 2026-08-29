#!/usr/bin/env python3
"""Launch an experiment and keep the controlling session bound to its watcher."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[2]
WATCHER = ROOT / "tools/analysis/watch_experiment_processes.py"
_SYSTEM_POPEN = subprocess.Popen


def write_json(path: pathlib.Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def record_launch_failure(watch_outdir: pathlib.Path, phase: str, error: Exception) -> None:
    write_json(
        watch_outdir / "LAUNCH_FAILED.json",
        {
            "schema": "local-labs-watched-launch-failure-v1",
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": phase,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )


def terminate_if_running(
    process: subprocess.Popen[bytes] | subprocess.Popen[str], *, tree: bool = False
) -> None:
    if process.poll() is not None:
        return
    group_terminated = False
    if tree and os.name == "nt":
        tree_killer = _SYSTEM_POPEN(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            group_terminated = tree_killer.wait(timeout=10) == 0
        except subprocess.TimeoutExpired:
            tree_killer.kill()
            tree_killer.wait(timeout=10)
    elif tree:
        try:
            process_group = os.getpgid(process.pid)
            if process_group != os.getpgrp():
                os.killpg(process_group, signal.SIGTERM)
                group_terminated = True
        except (ProcessLookupError, PermissionError, OSError):
            pass
    if not group_terminated:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if tree and os.name != "nt" and group_terminated:
            try:
                os.killpg(process_group, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        else:
            process.kill()
        process.wait(timeout=10)


def validate_packet_target(task_id: str, packet_dir: pathlib.Path, unmanaged: bool) -> None:
    if unmanaged:
        return
    manifest_path = ROOT / "config/research_backlog.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items") if isinstance(manifest, dict) else None
    task = next(
        (
            item for item in items or []
            if isinstance(item, dict) and item.get("id") == task_id
        ),
        None,
    )
    if task is None or not isinstance(task.get("packet_dir"), str):
        raise ValueError(f"{task_id}: task is not registered in the canonical backlog")
    if packet_dir != (ROOT / task["packet_dir"]).resolve():
        raise ValueError(f"{task_id}: packet_dir is not the canonical backlog packet")
    pipeline = json.loads((packet_dir / "PIPELINE.json").read_text(encoding="utf-8"))
    if not isinstance(pipeline, dict) or pipeline.get("task_id") != task_id:
        raise ValueError(f"{task_id}: PIPELINE.json identity mismatch")
    if task.get("state") != "IMPLEMENTED" or pipeline.get("stage") != "IMPLEMENTED":
        raise ValueError(f"{task_id}: launcher requires manifest and packet stage IMPLEMENTED")


def supervise_worker(
    experiment: subprocess.Popen[bytes] | subprocess.Popen[str],
    watcher: subprocess.Popen[bytes] | subprocess.Popen[str],
    max_runtime_seconds: int,
) -> dict[str, object]:
    """Reap the worker while ensuring its watcher remains alive."""
    deadline = time.monotonic() + max_runtime_seconds
    while True:
        watcher_returncode = watcher.poll()
        experiment_returncode = experiment.poll()
        if watcher_returncode is not None and experiment_returncode is None:
            terminate_if_running(experiment, tree=True)
            experiment_returncode = experiment.poll()
            return {
                "returncode": experiment_returncode if type(experiment_returncode) is int else 1,
                "timed_out": False,
                "watcher_failed_early": True,
                "watcher_returncode": watcher_returncode,
            }
        if experiment_returncode is not None:
            experiment.wait()
            return {
                "returncode": experiment_returncode,
                "timed_out": False,
                "watcher_failed_early": False,
                "watcher_returncode": watcher_returncode,
            }
        if time.monotonic() >= deadline:
            terminate_if_running(experiment, tree=True)
            experiment_returncode = experiment.poll()
            return {
                "returncode": experiment_returncode if type(experiment_returncode) is int else 124,
                "timed_out": True,
                "watcher_failed_early": False,
                "watcher_returncode": watcher_returncode,
            }
        time.sleep(0.1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--progress-glob")
    parser.add_argument(
        "--progress-mode",
        choices=("files", "jsonl_lines"),
        default="files",
        help="interpret progress as matched files or non-empty JSONL records",
    )
    parser.add_argument("--expected-progress", type=int)
    parser.add_argument(
        "--require-harness-terminal",
        action="store_true",
        help="require a verified raw/run.terminal.json instead of accepting a legacy packet",
    )
    parser.add_argument(
        "--unmanaged-canary",
        action="store_true",
        help="run an explicit temporary canary without backlog transition or audit handoff",
    )
    parser.add_argument("--watch-id", required=True)
    parser.add_argument("--watch-outdir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=300,
        help="watcher poll cadence; defaults to five minutes to keep monitoring quiet",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=86_400,
        help="hard worker deadline; defaults to 24 hours",
    )
    parser.add_argument(
        "--verbose-controller-output",
        action="store_true",
        help="print full launch metadata instead of compact controller notifications",
    )
    parser.add_argument(
        "--experiment-mode",
        action="store_true",
        help=(
            "after completion, refresh the backlog and instruct the controlling "
            "agent to dispatch the next dependency-ready candidate"
        ),
    )
    parser.add_argument(
        "--detach-watcher",
        action="store_true",
        help=(
            "return immediately after launch; this preserves on-disk monitoring "
            "but cannot notify the controlling session when the watcher finishes"
        ),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.max_runtime_seconds <= 0:
        parser.error("--max-runtime-seconds must be positive")
    if args.expected_progress is not None and args.expected_progress <= 0:
        parser.error("--expected-progress must be positive")
    if args.detach_watcher and args.require_harness_terminal:
        parser.error("harness-terminal runs require foreground control to preserve worker exit status")
    if not args.require_harness_terminal and (
        args.progress_glob is None or args.expected_progress is None
    ):
        parser.error(
            "legacy runs require --progress-glob and --expected-progress; "
            "new runners may use --require-harness-terminal instead"
        )
    progress_glob = args.progress_glob or "raw/run.terminal.json"
    expected_progress = args.expected_progress if args.expected_progress is not None else 1
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("an experiment command is required after --")

    watch_outdir = (ROOT / args.watch_outdir).resolve() if not args.watch_outdir.is_absolute() else args.watch_outdir.resolve()
    watch_outdir.mkdir(parents=True, exist_ok=True)
    packet_dir = (ROOT / args.packet_dir).resolve()
    try:
        validate_packet_target(args.task_id, packet_dir, args.unmanaged_canary)
    except Exception as error:
        record_launch_failure(watch_outdir, "packet_validation", error)
        print(f"packet validation failed: {error}", file=sys.stderr, flush=True)
        return 2
    if args.unmanaged_canary:
        packet_dir.mkdir(parents=True, exist_ok=True)
    elif not packet_dir.is_dir():
        error = FileNotFoundError("canonical packet directory does not exist")
        record_launch_failure(watch_outdir, "packet_validation", error)
        print(f"packet validation failed: {error}", file=sys.stderr, flush=True)
        return 2
    stdout_path = packet_dir / "runner.stdout.log"
    stderr_path = packet_dir / "runner.stderr.log"
    if stdout_path.exists() or stderr_path.exists():
        error = FileExistsError("runner logs already exist; use a fresh experiment attempt")
        record_launch_failure(watch_outdir, "log_initialization", error)
        print(f"log initialization failed: {error}", file=sys.stderr, flush=True)
        return 2
    try:
        experiment_stdout = stdout_path.open("x", encoding="utf-8")
        experiment_stderr = stderr_path.open("x", encoding="utf-8")
    except Exception as error:
        if "experiment_stdout" in locals():
            experiment_stdout.close()
            try:
                stdout_path.unlink()
            except FileNotFoundError:
                pass
        record_launch_failure(watch_outdir, "log_initialization", error)
        print(f"log initialization failed: {error}", file=sys.stderr, flush=True)
        return 2
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    try:
        experiment = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=experiment_stdout,
            stderr=experiment_stderr,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
    except Exception as error:
        record_launch_failure(watch_outdir, "experiment_spawn", error)
        print(f"experiment launch failed: {error}", file=sys.stderr, flush=True)
        return 2
    finally:
        experiment_stdout.close()
        experiment_stderr.close()

    run_id = uuid.uuid4().hex
    worker_exit_path = watch_outdir / "WORKER_EXIT.json"

    config = {
        "schema": "local-labs-experiment-watch-v1",
        "watch_id": args.watch_id,
        "experiment_mode": args.experiment_mode,
        "poll_seconds": max(5, args.poll_seconds),
        "service_settle_seconds": 180,
        "experiments": [{
            "task_id": args.task_id,
            "pid": experiment.pid,
            "packet_dir": pathlib.Path(args.packet_dir).as_posix(),
            "progress_glob": progress_glob,
            "progress_mode": args.progress_mode,
            "expected_progress": expected_progress,
            "require_harness_terminal": args.require_harness_terminal,
            "managed_backlog": not args.unmanaged_canary,
            "require_worker_exit": not args.detach_watcher,
            "worker_exit_path": str(worker_exit_path),
            "run_id": run_id,
            "deadline_epoch": time.time() + args.max_runtime_seconds,
        }],
        "final_health_urls": [
            "http://127.0.0.1:8080/health",
            "http://127.0.0.1:8081/health",
        ],
        "actor": "Codex executor watcher",
    }
    config_path = watch_outdir / "config.json"
    write_json(config_path, config)
    watcher_stdout = (watch_outdir / "watcher.stdout.log").open("w", encoding="utf-8")
    watcher_stderr = (watch_outdir / "watcher.stderr.log").open("w", encoding="utf-8")
    try:
        watcher = subprocess.Popen(
            [sys.executable, str(WATCHER), str(config_path), "--outdir", str(watch_outdir)],
            cwd=ROOT,
            stdout=watcher_stdout,
            stderr=watcher_stderr,
            creationflags=creationflags,
        )
    except Exception as error:
        terminate_if_running(experiment, tree=True)
        record_launch_failure(watch_outdir, "watcher_spawn", error)
        print(f"watcher launch failed: {error}", file=sys.stderr, flush=True)
        return 2
    finally:
        watcher_stdout.close()
        watcher_stderr.close()
    launch = {
        "schema": "local-labs-watched-launch-v1",
        "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task_id": args.task_id,
        "experiment_pid": experiment.pid,
        "watcher_pid": watcher.pid,
        "command": command,
        "config": str(config_path),
        "status": str(watch_outdir / "WATCH_STATUS.json"),
        "events": str(watch_outdir / "events.jsonl"),
        "final": str(watch_outdir / "FINAL.json"),
        "controller_binding": (
            "detached_no_completion_delivery"
            if args.detach_watcher
            else "foreground_until_watcher_completion"
        ),
        "experiment_mode": args.experiment_mode,
    }
    write_json(watch_outdir / "LAUNCH.json", launch)
    controller_launch = (
        launch
        if args.verbose_controller_output
        else {
            "event": "watcher_started",
            "task_id": args.task_id,
            "watch_id": args.watch_id,
            "poll_seconds": config["poll_seconds"],
        }
    )
    print(json.dumps(controller_launch, separators=(",", ":")), flush=True)
    if args.detach_watcher:
        print(
            "WARNING: watcher detached; completion will only be persisted on disk "
            "and will not wake the controlling session.",
            file=sys.stderr,
            flush=True,
        )
        return 0

    supervision = supervise_worker(experiment, watcher, args.max_runtime_seconds)
    write_json(
        worker_exit_path,
        {
            "schema": "local-labs-worker-exit-v1",
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "task_id": args.task_id,
            "pid": experiment.pid,
            "run_id": run_id,
            "returncode": supervision["returncode"],
            "timed_out": supervision["timed_out"],
            "watcher_failed_early": supervision["watcher_failed_early"],
        },
    )
    watcher_timed_out = False
    if supervision["watcher_failed_early"]:
        watcher_returncode = watcher.wait()
    else:
        watcher_grace_seconds = 180 + config["poll_seconds"] + 30
        try:
            watcher_returncode = watcher.wait(timeout=watcher_grace_seconds)
        except subprocess.TimeoutExpired:
            watcher_timed_out = True
            terminate_if_running(watcher)
            polled = watcher.poll()
            watcher_returncode = polled if type(polled) is int else 124
    final_path = watch_outdir / "FINAL.json"
    final = None
    final_error = None
    if supervision["watcher_failed_early"]:
        final_error = "watcher exited before worker completion"
    elif watcher_timed_out:
        final_error = "watcher did not finish within its bounded completion window"
    elif final_path.is_file():
        try:
            final = json.loads(final_path.read_text(encoding="utf-8"))
            if not isinstance(final, dict):
                raise TypeError("FINAL.json is not an object")
            if final.get("schema") != "local-labs-experiment-watch-final-v1":
                raise ValueError("FINAL.json schema mismatch")
            if final.get("watch_id") != args.watch_id:
                raise ValueError("FINAL.json watch_id mismatch")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            final_error = f"invalid FINAL.json: {error}"
    else:
        final_error = "watcher exited without FINAL.json"
    next_candidate = (
        final.get("backlog_queue", {}).get("next_candidate")
        if isinstance(final, dict)
        else None
    )
    controller_final = {
        "event": "watcher_completed",
        "task_id": args.task_id,
        "status": final.get("status") if isinstance(final, dict) else None,
        "action": final.get("completion_action") if isinstance(final, dict) else None,
        "next_id": next_candidate.get("id") if isinstance(next_candidate, dict) else None,
        "audit_ready_ids": final.get("audit_ready_ids", []) if isinstance(final, dict) else [],
    }
    if args.verbose_controller_output:
        controller_final |= {
            "watch_id": args.watch_id,
            "watcher_returncode": watcher_returncode,
            "final_path": str(final_path),
            "experiment_mode": (
                final.get("experiment_mode")
                if isinstance(final, dict)
                else args.experiment_mode
            ),
            "next_candidate": next_candidate,
        }
    print(json.dumps(controller_final, separators=(",", ":")), flush=True)
    if final_error:
        print(final_error, file=sys.stderr, flush=True)
        return watcher_returncode if watcher_returncode != 0 else 3
    return watcher_returncode


if __name__ == "__main__":
    raise SystemExit(main())
