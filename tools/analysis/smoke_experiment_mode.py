#!/usr/bin/env python3
"""Run a live, non-mutating canary through the experiment-mode watcher path."""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import time


ROOT = pathlib.Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "tools/analysis/launch_watched_experiment.py"
BACKLOG = ROOT / "config/research_backlog.json"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    backlog_before = sha256(BACKLOG)
    with tempfile.TemporaryDirectory(prefix="local-labs-experiment-mode-") as temporary:
        root = pathlib.Path(temporary)
        packet = root / "packet"
        watch = root / "watch"
        write_json(packet / "PIPELINE.json", {"stage": "EXECUTED"})
        write_json(packet / "raw/receipt.json", {"schema": "canary"})
        (packet / "RESULT.md").write_text("# Canary result\n", encoding="utf-8")

        worker = (
            "import pathlib,sys,time; "
            "p=pathlib.Path(sys.argv[1])/'raw/finalized'; "
            "p.mkdir(parents=True,exist_ok=True); "
            "(p/'done.json').write_text('{}',encoding='utf-8'); "
            "time.sleep(3)"
        )
        command = [
            sys.executable,
            str(LAUNCHER),
            "--task-id", "EXPERIMENT-MODE-CANARY",
            "--packet-dir", str(packet),
            "--progress-glob", "raw/finalized/*.json",
            "--expected-progress", "1",
            "--watch-id", "EXPERIMENT-MODE-CANARY",
            "--watch-outdir", str(watch),
            "--poll-seconds", "5",
            "--experiment-mode",
            "--",
            sys.executable,
            "-c",
            worker,
            str(packet),
        ]
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
            check=False,
        )
        elapsed = time.monotonic() - started
        final_path = watch / "FINAL.json"
        if completed.returncode != 0:
            raise RuntimeError(
                f"launcher returned {completed.returncode}: {completed.stderr}\n{completed.stdout}"
            )
        if not final_path.is_file():
            raise RuntimeError("watcher did not produce FINAL.json")
        final = json.loads(final_path.read_text(encoding="utf-8"))
        controller_events = []
        for line in completed.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                controller_events.append(event)
        next_candidate = final["backlog_queue"]["next_candidate"]
        expected_action = (
            "dispatch_next_candidate" if isinstance(next_candidate, dict)
            else "notify_queue_empty"
        )
        checks = {
            "foreground_wait_observed": elapsed >= 3,
            "completion_delivered": any(
                event.get("event") == "watcher_completed"
                for event in controller_events
            ),
            "final_complete": final.get("status") == "complete",
            "experiment_mode_preserved": final.get("experiment_mode") is True,
            "queue_refreshed": final.get("backlog_queue", {}).get("status_returncode") == 0,
            "next_candidate_selected": final.get("completion_action") == expected_action,
            "backlog_unchanged": sha256(BACKLOG) == backlog_before,
        }
        report = {
            "status": "pass" if all(checks.values()) else "fail",
            "elapsed_seconds": round(elapsed, 3),
            "checks": checks,
            "completion_action": final.get("completion_action"),
            "next_candidate": next_candidate,
            "state_counts": final["backlog_queue"]["state_counts"],
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
