#!/usr/bin/env python3
"""Run the frozen, seeded mutation gate for the experiment harness and watcher."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = pathlib.Path(__file__).resolve().parents[2]
H = "src/model_lifecycle/experiment_harness.py"
W = "tools/analysis/watch_experiment_processes.py"
L = "tools/analysis/launch_watched_experiment.py"

# Each operator deletes or reverses one fail-closed decision. Exact one-match
# replacement makes source drift visible as INVALID instead of silently skipping it.
MUTANTS = [
    ("receipt_schema", H, 'if payload.get("schema") != RECEIPT_SCHEMA:', "if False:"),
    ("receipt_task", H, 'if payload.get("task_id") != task_id:', "if False:"),
    ("receipt_fingerprint", H, "if supplied is not None and supplied != computed:", "if False:"),
    ("nested_reserved_files", H, "if not path.is_file() or path in {\n            raw_dir / LOCK_NAME,\n            raw_dir / TERMINAL_NAME,\n        }:", 'if not path.is_file() or path.name in {LOCK_NAME, TERMINAL_NAME}:'),
    ("temporary_files", H, 'if reject_temporary and path.name.endswith(".tmp"):', "if False:"),
    ("fresh_attempt", H, "if any(path.exists() for path in (", "if False and any(path.exists() for path in ("),
    ("exclusive_lock", H, "os.O_CREAT | os.O_EXCL | os.O_WRONLY", "os.O_CREAT | os.O_WRONLY"),
    ("preflight_transport", H, "if not result.ok:", "if False:"),
    ("preflight_answered", H, "if contract.require_answered and not result.answered:", "if False:"),
    ("preflight_text", H, "if contract.require_text and not result.text.strip():", "if False:"),
    ("preflight_tokens", H, "if contract.require_generated_tokens and result.completion_tokens <= 0:", "if False:"),
    ("preflight_timings", H, "if contract.require_server_timings and not (", "if False and ("),
    ("sample_counter", H, "self._sample_count += 1", "self._sample_count += 0"),
    ("restoration_result", H, "self._restored = ok", "self._restored = True"),
    ("seal_restoration", H, "if self.requires_restoration and not self._restored:", "if False:"),
    ("seal_status", H, 'self._write_terminal("SEALED")', 'self._write_terminal("ABORTED")'),
    ("journal_object", H, "if not isinstance(event, dict):", "if False:"),
    ("journal_schema", H, 'if event.get("schema") != JOURNAL_SCHEMA:', "if False:"),
    ("journal_sequence", H, 'if event.get("seq") != index:', "if False:"),
    ("journal_task", H, 'if event.get("task_id") != task_id:', "if False:"),
    ("journal_chain", H, 'if event.get("prev_sha256") != previous:', "if False:"),
    ("journal_digest", H, 'if event.get("event_sha256") != _event_digest(event):', "if False:"),
    ("terminal_object", H, "if not isinstance(terminal, dict):", "if False:"),
    ("terminal_schema", H, 'if terminal.get("schema") != TERMINAL_SCHEMA:', "if False:"),
    ("terminal_fingerprint", H, "if supplied_terminal_fingerprint != _sha256_value(terminal_content):", "if False:"),
    ("terminal_lock", H, "if (raw / LOCK_NAME).exists():", "if False:"),
    ("manifest_missing", H, "if missing:", "if False:"),
    ("manifest_extra", H, "if extra:", "if False:"),
    ("manifest_changed", H, "if changed:", "if False:"),
    ("journal_tail", H, 'if events and events[-1].get("event_sha256") != terminal.get("last_event_sha256"):', "if False:"),
    ("terminal_status", H, "if expected_last_type is None:", "if False:"),
    ("terminal_journal_agreement", H, 'elif events and events[-1].get("type") != expected_last_type:', "elif False:"),
    ("terminal_restoration", H, 'terminal.get("status") == "SEALED"\n        and terminal.get("requires_restoration")', 'False\n        and terminal.get("requires_restoration")'),
    ("verify_receipt_fingerprint", H, "if supplied != _sha256_value(normalized):", "if False:"),
    ("verify_receipt_task", H, 'if receipt.get("task_id") != task_id:', "if False:"),
    ("verify_receipt_schema", H, 'if receipt.get("schema") != RECEIPT_SCHEMA:', "if False:"),
    ("verify_sample_count", H, 'if len(samples) != terminal.get("sample_count"):', "if False:"),
    ("journal_known_events", H, "if unknown_types:", "if False:"),
    ("journal_single_started", H, 'if event_types.count("STARTED") != 1:', "if False:"),
    ("journal_terminal_position", H, "if terminal_positions != [len(events) - 1]:", "if False:"),
    ("journal_failed_transition", H, 'if event_types[failed_at + 1:] != ["ABORTED"]:', "if False:"),
    ("started_inputs", H, "if not (\n                isinstance(inputs_sha256, str)", "if False and (\n                isinstance(inputs_sha256, str)"),
    ("started_contract", H, 'if started_payload.get("requires_restoration") != terminal.get("requires_restoration"):', "if False:"),
    ("final_event_sample_count", H, 'elif final_payload.get("sample_count") != len(samples):', "elif False:"),
    ("restoration_event", H, 'not restoration_events or restoration_events[-1].get("type") != "RESTORED"', "False"),
    ("sealed_receipt_sha", H, 'if final_payload.get("receipt_sha256") != _sha256_file(receipt_path):', "if False:"),
    ("sealed_receipt_fingerprint", H, 'if final_payload.get("receipt_fingerprint") != receipt.get("receipt_fingerprint"):', "if False:"),
    ("verify_replay", H, "if not replay_matches:", "if False:"),
    ("zombie_detection", W, 'if closing >= 0 and stat[closing + 2:closing + 3] == "Z":', "if False:"),
    ("strict_exit_type", W, 'if type(payload.get("returncode")) is not int:', 'if not isinstance(payload.get("returncode"), int):'),
    ("canonical_packet", W, 'if packet_dir != canonical_packet_dir or canonical.get("state") != stage_before:', "if False:"),
    ("packet_identity", W, 'if packet is None or packet.get("task_id") != item["task_id"]:', "if False:"),
    ("worker_exit_required", W, 'if item.get("require_worker_exit") and not worker_exit["present"]:', "if False:"),
    ("worker_exit_valid", W, 'if worker_exit["present"] and not worker_exit["valid"]:', "if False:"),
    ("worker_timeout", W, 'if worker_exit["present"] and worker_exit.get("timed_out"):', "if False:"),
    ("worker_returncode", W, 'if worker_exit["present"] and worker_exit["returncode"] != 0:', "if False:"),
    ("harness_required", W, 'if item.get("require_harness_terminal") and not terminal["present"]:', "if False:"),
    ("harness_valid", W, 'if terminal["present"] and not terminal["valid"]:', "if False:"),
    ("harness_aborted", W, 'if terminal["present"] and terminal["status"] == "ABORTED":', "if False:"),
    ("harness_status", W, 'if terminal["present"] and terminal["status"] != "SEALED":', "if False:"),
    ("harness_identity", W, 'if terminal["present"] and terminal.get("task_id") != item["task_id"]:', "if False:"),
    ("result_required", W, "if not result.is_file():", "if False:"),
    ("sealed_progress_bypass", W, 'if not terminal["present"] and outcome["progress"] < outcome["expected_progress"]:', 'if outcome["progress"] < outcome["expected_progress"]:'),
    ("watch_deadline", W, 'elif time.time() >= deadline:', "elif False:"),
    ("audit_ready", W, 'if state["status"] == "executed_valid"', 'if state["status"] != "executed_valid"'),
    ("legacy_contract", L, "if not args.require_harness_terminal and (", "if False and ("),
    ("detach_harness", L, "if args.detach_watcher and args.require_harness_terminal:", "if False:"),
    ("unmanaged_canary_forwarding", L, '"managed_backlog": not args.unmanaged_canary,', '"managed_backlog": True,'),
    ("watcher_early_exit", L, "if watcher_returncode is not None and experiment_returncode is None:", "if False:"),
    (
        "watcher_early_tree_termination",
        L,
        "if watcher_returncode is not None and experiment_returncode is None:\n            terminate_if_running(experiment, tree=True)",
        "if watcher_returncode is not None and experiment_returncode is None:\n            terminate_if_running(experiment)",
    ),
    ("preserve_existing_runner_logs", L, "if stdout_path.exists() or stderr_path.exists():", "if False:"),
    (
        "launcher_stage_authority",
        L,
        'if task.get("state") != "IMPLEMENTED" or pipeline.get("stage") != "IMPLEMENTED":',
        "if False:",
    ),
    ("terminal_forwarding", L, '"require_harness_terminal": args.require_harness_terminal,', '"require_harness_terminal": False,'),
    ("exit_forwarding", L, '"require_worker_exit": not args.detach_watcher,', '"require_worker_exit": False,'),
    ("exit_returncode", L, '"returncode": experiment_returncode,', '"returncode": 0,'),
    ("audit_handoff", L, '"audit_ready_ids": final.get("audit_ready_ids", []) if isinstance(final, dict) else [],', '"audit_ready_ids": [],'),
]


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_tree(destination: pathlib.Path) -> None:
    shutil.copytree(ROOT / "src", destination / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "tools").mkdir()
    shutil.copytree(ROOT / "tools/analysis", destination / "tools/analysis", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "tests").mkdir()
    for name in ("test_experiment_harness.py", "test_watch_experiment_processes.py"):
        shutil.copy2(ROOT / "tests" / name, destination / "tests" / name)


def run_tests(tree: pathlib.Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tree)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_experiment_harness.py", "tests/test_watch_experiment_processes.py", "-q", "--tb=short"],
        cwd=tree,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=15,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "runs/benchmarks/HARNESS-MUTATION-2026-08-28/report.json",
    )
    args = parser.parse_args()
    results = []
    with tempfile.TemporaryDirectory(prefix="local-labs-mutants-") as temporary:
        base = pathlib.Path(temporary) / "base"
        copy_tree(base)
        baseline = run_tests(base)
        if baseline.returncode:
            print(baseline.stdout, file=sys.stderr)
            return 2
        for index, (name, relative, old, new) in enumerate(MUTANTS, 1):
            tree = pathlib.Path(temporary) / f"mutant-{index:03d}"
            shutil.copytree(base, tree, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            target = tree / relative
            source = target.read_text(encoding="utf-8")
            occurrences = source.count(old)
            line = source[:source.find(old)].count("\n") + 1 if occurrences else None
            if occurrences != 1:
                result = {"name": name, "file": relative, "line": line, "status": "INVALID", "occurrences": occurrences}
            else:
                target.write_text(source.replace(old, new, 1), encoding="utf-8")
                try:
                    completed = run_tests(tree)
                    result = {
                        "name": name,
                        "file": relative,
                        "line": line,
                        "status": "KILLED" if completed.returncode else "SURVIVED",
                        "returncode": completed.returncode,
                    }
                except subprocess.TimeoutExpired:
                    result = {
                        "name": name,
                        "file": relative,
                        "line": line,
                        "status": "KILLED",
                        "returncode": None,
                        "timeout": True,
                    }
            results.append(result)
            print(f"{index:02d}/{len(MUTANTS)} {name}: {result['status']}", flush=True)
    counts = {
        status.lower(): sum(result["status"] == status for result in results)
        for status in ("KILLED", "SURVIVED", "INVALID")
    }
    report = {
        "schema": "local-labs-seeded-mutation-report-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": "python tools/analysis/mutation_test_experiment_harness.py",
        "test_files": ["tests/test_experiment_harness.py", "tests/test_watch_experiment_processes.py"],
        "baseline": "PASS",
        "python": sys.version.split()[0],
        "source_sha256": {
            relative: sha256(ROOT / relative)
            for relative in (
                H,
                W,
                L,
                "tests/test_experiment_harness.py",
                "tests/test_watch_experiment_processes.py",
                "tools/analysis/mutation_test_experiment_harness.py",
            )
        },
        "counts": counts | {"total": len(results)},
        "mutants": results,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("MUTATION_SUMMARY=" + json.dumps(report["counts"], sort_keys=True))
    print(f"MUTATION_REPORT={output}")
    return 0 if counts["survived"] == 0 and counts["invalid"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
