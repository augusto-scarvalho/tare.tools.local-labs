#!/usr/bin/env python3
"""Launch an experiment and keep the controlling session bound to its watcher."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
WATCHER = ROOT / "tools/analysis/watch_experiment_processes.py"


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def terminate_if_running(process: subprocess.Popen[bytes] | subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--progress-glob", required=True)
    parser.add_argument("--expected-progress", required=True, type=int)
    parser.add_argument("--watch-id", required=True)
    parser.add_argument("--watch-outdir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=300,
        help="watcher poll cadence; defaults to five minutes to keep monitoring quiet",
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
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("an experiment command is required after --")

    watch_outdir = (ROOT / args.watch_outdir).resolve() if not args.watch_outdir.is_absolute() else args.watch_outdir.resolve()
    watch_outdir.mkdir(parents=True, exist_ok=True)
    packet_dir = (ROOT / args.packet_dir).resolve()
    packet_dir.mkdir(parents=True, exist_ok=True)
    experiment_stdout = (packet_dir / "runner.stdout.log").open("w", encoding="utf-8")
    experiment_stderr = (packet_dir / "runner.stderr.log").open("w", encoding="utf-8")
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
        )
    except Exception as error:
        record_launch_failure(watch_outdir, "experiment_spawn", error)
        print(f"experiment launch failed: {error}", file=sys.stderr, flush=True)
        return 2
    finally:
        experiment_stdout.close()
        experiment_stderr.close()

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
            "progress_glob": args.progress_glob,
            "expected_progress": args.expected_progress,
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
        terminate_if_running(experiment)
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

    watcher_returncode = watcher.wait()
    final_path = watch_outdir / "FINAL.json"
    final = None
    final_error = None
    if final_path.is_file():
        try:
            final = json.loads(final_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
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
