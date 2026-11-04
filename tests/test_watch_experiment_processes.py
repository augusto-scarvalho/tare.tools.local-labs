import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

import tools.analysis.launch_watched_experiment as launcher
import tools.analysis.watch_experiment_processes as watcher
from src.model_lifecycle.experiment_harness import ExperimentRun


@pytest.fixture
def watcher_repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "config").mkdir()
    (root / "config/research_backlog.json").write_text(
        json.dumps({"items": []}), encoding="utf-8"
    )
    monkeypatch.setattr(watcher, "ROOT", root)
    monkeypatch.setattr(watcher, "PIPELINE", root / "tools/analysis/backlog_pipeline.py")
    monkeypatch.setattr(watcher, "gpu_state", lambda: {"utilization_percent": 0})
    monkeypatch.setattr(watcher.time, "sleep", lambda _seconds: None)
    return root


@pytest.fixture
def packet_factory(watcher_repo):
    def create(
        task_id="BACKLOG-TEST-01",
        *,
        stage="IMPLEMENTED",
        receipt=True,
        result=True,
        progress=1,
    ):
        packet = watcher_repo / "runs/research" / task_id
        (packet / "raw/finalized").mkdir(parents=True)
        (packet / "PIPELINE.json").write_text(
            json.dumps({"task_id": task_id, "stage": stage}) + "\n", encoding="utf-8"
        )
        backlog_path = watcher_repo / "config/research_backlog.json"
        backlog = json.loads(backlog_path.read_text(encoding="utf-8"))
        backlog["items"] = [item for item in backlog["items"] if item.get("id") != task_id]
        backlog["items"].append(
            {
                "id": task_id,
                "state": stage,
                "packet_dir": f"runs/research/{task_id}",
            }
        )
        backlog_path.write_text(json.dumps(backlog), encoding="utf-8")
        if receipt:
            (packet / "raw/receipt.json").write_text("{}\n", encoding="utf-8")
        if result:
            (packet / "RESULT.md").write_text("# Result\n", encoding="utf-8")
        for index in range(progress):
            (packet / f"raw/finalized/{index}.json").write_text("{}\n", encoding="utf-8")
        return {
            "task_id": task_id,
            "pid": abs(hash(task_id)) % 100_000 + 100,
            "packet_dir": f"runs/research/{task_id}",
            "progress_glob": "raw/finalized/*.json",
            "expected_progress": 1,
        }

    return create


@pytest.fixture
def worker_exit_factory(watcher_repo):
    def create(item, *, returncode=0, timed_out=False, **overrides):
        path = watcher_repo / f"worker-exit-{item['task_id']}.json"
        item.update(
            {
                "require_worker_exit": True,
                "worker_exit_path": str(path),
                "run_id": f"run-{item['task_id']}",
                "deadline_epoch": 4_102_444_800,
            }
        )
        payload = {
            "schema": "local-labs-worker-exit-v1",
            "task_id": item["task_id"],
            "pid": item["pid"],
            "run_id": item["run_id"],
            "returncode": returncode,
            "timed_out": timed_out,
        }
        payload.update(overrides)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return item

    return create


@pytest.fixture
def harness_packet(packet_factory, watcher_repo):
    def create(task_id="BACKLOG-HARNESS-WATCH", *, status="SEALED", mutate=False):
        item = packet_factory(task_id, receipt=False, progress=0)
        raw = watcher_repo / item["packet_dir"] / "raw"
        receipt = {
            "schema": "local-labs-backlog-receipt-v1",
            "task_id": task_id,
            "provenance": {"fixture": True},
            "provenance_complete": True,
            "gates": {},
            "evidence": {"raw_samples": "raw/samples.jsonl"},
        }
        if status == "SEALED":
            with ExperimentRun(raw, task_id, {"fixture": True}) as run:
                run.record({"fixture": 1})
                run.seal(receipt)
        else:
            with pytest.raises(RuntimeError):
                with ExperimentRun(raw, task_id, {"fixture": True}) as run:
                    run.record({"partial": True})
                    raise RuntimeError("fixture abort")
        if mutate:
            (raw / "samples.jsonl").write_text('{"fixture":2}\n', encoding="utf-8")
        item["require_harness_terminal"] = True
        return item

    return create


@pytest.fixture
def pipeline_stub(watcher_repo, monkeypatch):
    state = SimpleNamespace(
        calls=[],
        advance_returncode=0,
        gate_returncode=0,
        status_returncode=0,
        next_returncode=0,
        rebalance_returncode=0,
        rebalance_change_count=0,
        rebalance_applied_ids=[],
        status_items=[{"id": "DONE", "state": "EXECUTED"}],
        next_candidate=None,
    )

    def run_pipeline(*arguments):
        state.calls.append(arguments)
        command = arguments[0]
        if command == "advance":
            if state.advance_returncode == 0:
                packet = watcher_repo / "runs/research" / arguments[1] / "PIPELINE.json"
                payload = json.loads(packet.read_text(encoding="utf-8"))
                payload["stage"] = arguments[3]
                packet.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            return {
                "returncode": state.advance_returncode,
                "stdout": "advanced" if state.advance_returncode == 0 else "",
                "stderr": "" if state.advance_returncode == 0 else "advance failed",
            }
        if command == "gate":
            return {
                "returncode": state.gate_returncode,
                "stdout": "BACKLOG PIPELINE: PASS" if state.gate_returncode == 0 else "",
                "stderr": "" if state.gate_returncode == 0 else "gate failed",
            }
        if command == "status":
            return {
                "returncode": state.status_returncode,
                "stdout": json.dumps(state.status_items),
                "stderr": "" if state.status_returncode == 0 else "status failed",
            }
        if command == "next":
            return {
                "returncode": state.next_returncode,
                "stdout": json.dumps(state.next_candidate),
                "stderr": "" if state.next_returncode == 0 else "next failed",
            }
        if command == "rebalance":
            payload = {
                "schema": "local-labs-backlog-priority-report-v1",
                "mode": "apply",
                "policy_sha256": "a" * 64,
                "assessed_count": 4,
                "change_count": state.rebalance_change_count,
                "applied_ids": state.rebalance_applied_ids,
                "items": [],
            }
            return {
                "returncode": state.rebalance_returncode,
                "stdout": json.dumps(payload) if state.rebalance_returncode == 0 else "",
                "stderr": "" if state.rebalance_returncode == 0 else "rebalance failed",
            }
        raise AssertionError(arguments)

    monkeypatch.setattr(watcher, "run_pipeline", run_pipeline)
    return state


@pytest.fixture
def run_watcher(watcher_repo, pipeline_stub, monkeypatch):
    sequence_number = 0

    def run(
        items,
        *,
        experiment_mode=True,
        alive="observed_then_finished",
        health=None,
        service_settle_seconds=0,
    ):
        nonlocal sequence_number
        sequence_number += 1
        call_counts = {}

        def process_alive(pid):
            call_counts[pid] = call_counts.get(pid, 0) + 1
            if alive == "never_observed":
                return False
            if isinstance(alive, dict):
                return call_counts[pid] <= alive.get(pid, 1)
            return call_counts[pid] == 1

        health_calls = {}

        def http_status(url):
            health_calls[url] = health_calls.get(url, 0) + 1
            value = health.get(url, 200) if isinstance(health, dict) else (health or 200)
            if isinstance(value, list):
                return value[min(health_calls[url] - 1, len(value) - 1)]
            return value

        monkeypatch.setattr(watcher, "process_alive", process_alive)
        monkeypatch.setattr(watcher, "http_status", http_status)
        config = {
            "schema": "local-labs-experiment-watch-v1",
            "watch_id": f"WATCH-FIXTURE-{sequence_number}",
            "experiment_mode": experiment_mode,
            "poll_seconds": 5,
            "service_settle_seconds": service_settle_seconds,
            "experiments": items,
            "final_health_urls": ["http://fixture/8080", "http://fixture/8081"],
            "actor": "pytest fixture",
        }
        config_path = watcher_repo / f"config-{sequence_number}.json"
        outdir = watcher_repo / f"watch-{sequence_number}"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        monkeypatch.setattr(
            sys,
            "argv",
            ["watch_experiment_processes.py", str(config_path), "--outdir", str(outdir)],
        )
        returncode = watcher.main()
        final = json.loads((outdir / "FINAL.json").read_text(encoding="utf-8"))
        events = [
            json.loads(line)
            for line in (outdir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        return SimpleNamespace(
            returncode=returncode,
            final=final,
            events=events,
            outdir=outdir,
            process_calls=call_counts,
            health_calls=health_calls,
        )

    return run


class FakeProcess:
    def __init__(self, pid, returncode=0, on_wait=None, finish_on_poll=False):
        self.pid = pid
        self.returncode = returncode
        self.on_wait = on_wait
        self.wait_calls = 0
        self.terminated = False
        self.killed = False
        self.running = True
        self.finish_on_poll = finish_on_poll

    def poll(self):
        if self.running and self.finish_on_poll:
            self.running = False
        return None if self.running else self.returncode

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.on_wait:
            self.on_wait()
        self.running = False
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.running = False

    def kill(self):
        self.killed = True
        self.running = False


@pytest.fixture
def launcher_harness(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(launcher, "ROOT", root)
    monkeypatch.setattr(launcher, "WATCHER", root / "tools/analysis/watch_experiment_processes.py")

    def run(
        *,
        detach=False,
        experiment_mode=True,
        require_terminal=False,
        omit_progress=False,
        spawn_error_at=None,
        final_payload="default",
        watcher_returncode=0,
        worker_returncode=0,
        watcher_exits_early=False,
        unmanaged_canary=False,
        existing_logs=False,
        task_state="IMPLEMENTED",
        progress_mode="files",
        expected_progress=1,
    ):
        packet = root / "packet"
        watch_outdir = root / "watch"
        packet.mkdir(exist_ok=True)
        (packet / "PIPELINE.json").write_text(
            json.dumps({"task_id": "BACKLOG-TEST", "stage": task_state}),
            encoding="utf-8",
        )
        (root / "config").mkdir(exist_ok=True)
        (root / "config/research_backlog.json").write_text(
            json.dumps(
                {
                    "items": [
                        {
                            "id": "BACKLOG-TEST",
                            "state": task_state,
                            "packet_dir": "packet",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        if existing_logs:
            (packet / "runner.stdout.log").write_text("preserved stdout", encoding="utf-8")
            (packet / "runner.stderr.log").write_text("preserved stderr", encoding="utf-8")
        processes = []
        popen_calls = []

        def write_final():
            if final_payload is None:
                return
            watch_outdir.mkdir(exist_ok=True)
            path = watch_outdir / "FINAL.json"
            if final_payload == "invalid":
                path.write_text("{invalid", encoding="utf-8")
                return
            payload = {
                "schema": "local-labs-experiment-watch-final-v1",
                "watch_id": "WATCH-TEST",
                "status": "complete",
                "experiment_mode": experiment_mode,
                "completion_action": "dispatch_next_candidate",
                "backlog_queue": {"next_candidate": {"id": "BACKLOG-NEXT"}},
                "audit_ready_ids": ["BACKLOG-TEST"],
            } if final_payload == "default" else final_payload
            path.write_text(json.dumps(payload), encoding="utf-8")

        def popen(*args, **kwargs):
            index = len(popen_calls) + 1
            popen_calls.append((args, kwargs))
            if spawn_error_at == index:
                raise OSError(f"spawn {index} failed")
            process = FakeProcess(
                1000 + index,
                returncode=watcher_returncode if index == 2 else worker_returncode,
                on_wait=write_final if index == 2 and not detach else None,
                finish_on_poll=(
                    index == 1
                    and spawn_error_at != 2
                    and not watcher_exits_early
                ),
            )
            if index == 2 and watcher_exits_early:
                process.running = False
            processes.append(process)
            return process

        monkeypatch.setattr(launcher.subprocess, "Popen", popen)
        argv = [
            "launch_watched_experiment.py",
            "--task-id", "BACKLOG-TEST",
            "--packet-dir", "packet",
            "--watch-id", "WATCH-TEST",
            "--watch-outdir", "watch",
        ]
        if not omit_progress:
            argv.extend([
                "--progress-glob", "raw/*.json",
                "--progress-mode", progress_mode,
                "--expected-progress", str(expected_progress),
            ])
        if experiment_mode:
            argv.append("--experiment-mode")
        if require_terminal:
            argv.append("--require-harness-terminal")
        if unmanaged_canary:
            argv.append("--unmanaged-canary")
        if detach:
            argv.append("--detach-watcher")
        argv.extend(["--", sys.executable, "-c", "pass"])
        monkeypatch.setattr(sys, "argv", argv)
        returncode = launcher.main()
        launch_path = watch_outdir / "LAUNCH.json"
        failure_path = watch_outdir / "LAUNCH_FAILED.json"
        return SimpleNamespace(
            returncode=returncode,
            packet=packet,
            processes=processes,
            popen_calls=popen_calls,
            launch=json.loads(launch_path.read_text(encoding="utf-8")) if launch_path.exists() else None,
            failure=json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else None,
            config=json.loads((watch_outdir / "config.json").read_text(encoding="utf-8")) if (watch_outdir / "config.json").exists() else None,
        )

    return run


def test_process_alive_detects_current_process():
    assert watcher.process_alive(os.getpid())


def test_process_alive_rejects_impossible_pid():
    assert not watcher.process_alive(2_000_000_000)


def test_process_alive_treats_posix_zombie_as_finished(monkeypatch):
    monkeypatch.setattr(watcher.os, "name", "posix")
    monkeypatch.setattr(watcher.os, "kill", lambda _pid, _signal: None)

    class ZombieStat:
        def read_text(self, **_kwargs):
            return "123 (fixture worker) Z 1 2 3"

    monkeypatch.setattr(watcher.pathlib, "Path", lambda _path: ZombieStat())
    assert watcher.process_alive(123) is False


def test_http_status_returns_response_status(monkeypatch):
    class Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(watcher.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    assert watcher.http_status("http://fixture") == 204


def test_http_status_returns_none_on_transport_error(monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(watcher.urllib.request, "urlopen", fail)
    assert watcher.http_status("http://fixture") is None


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [
        (1, "", {"error": "gpu failed", "returncode": 1}),
        (0, "bad,data", {"raw": "bad,data"}),
        (
            0,
            "100, 200, 30, 40, 50.5",
            {
                "memory_used_mib": 100,
                "memory_free_mib": 200,
                "utilization_percent": 30,
                "temperature_c": 40,
                "power_w": 50.5,
            },
        ),
    ],
)
def test_gpu_state_parsing(monkeypatch, returncode, stdout, expected):
    completed = SimpleNamespace(returncode=returncode, stdout=stdout, stderr="gpu failed")
    monkeypatch.setattr(watcher.subprocess, "run", lambda *_args, **_kwargs: completed)
    assert watcher.gpu_state() == expected


def test_pipeline_stage_handles_missing_and_existing_packet(tmp_path):
    assert watcher.pipeline_stage(tmp_path) is None
    (tmp_path / "PIPELINE.json").write_text('{"stage":"EXECUTED"}', encoding="utf-8")
    assert watcher.pipeline_stage(tmp_path) == "EXECUTED"


def test_run_pipeline_captures_command_result(monkeypatch, tmp_path):
    completed = SimpleNamespace(returncode=7, stdout=" out \n", stderr=" err \n")
    monkeypatch.setattr(watcher, "ROOT", tmp_path)
    monkeypatch.setattr(watcher, "PIPELINE", tmp_path / "pipeline.py")
    monkeypatch.setattr(watcher.subprocess, "run", lambda *_args, **_kwargs: completed)
    assert watcher.run_pipeline("gate") == {
        "returncode": 7,
        "stdout": "out",
        "stderr": "err",
    }


@pytest.mark.parametrize(
    ("experiment_mode", "status", "candidate", "expected"),
    [
        (True, "complete", {"id": "BACKLOG-NEXT"}, "dispatch_next_candidate"),
        (True, "complete", None, "notify_queue_empty"),
        (True, "complete_with_alert", {"id": "BACKLOG-NEXT"}, "inspect_alert_before_dispatch"),
        (False, "complete", {"id": "BACKLOG-NEXT"}, "notify_completion"),
        (True, "crashed", {"id": "BACKLOG-NEXT"}, "stop_fail_closed"),
    ],
)
def test_completion_action_matrix(experiment_mode, status, candidate, expected):
    assert watcher.completion_action(experiment_mode, status, candidate) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("IMPLEMENTED", "dispatch_implemented_candidate"),
        ("EXECUTED", "dispatch_audit_candidate"),
        ("BLOCKED", "resolve_blocked_candidate"),
    ],
)
def test_completion_action_surfaces_unresolved_leaf(state, expected):
    candidate = {"id": "BACKLOG-LEAF", "state": state}
    assert watcher.completion_action(True, "complete", None, candidate) == expected


def test_unresolved_leaf_candidate_ignores_superseded_predecessors():
    items = [
        {"id": "OLD", "state": "BLOCKED", "priority": 0, "priority_score": 99},
        {"id": "DONE", "state": "PROMOTED", "priority": 1, "supersedes": "OLD"},
        {"id": "LIVE", "state": "BLOCKED", "priority": 1, "priority_score": 50},
    ]
    assert watcher.unresolved_leaf_candidate(items)["id"] == "LIVE"


def test_clean_completion_advances_packet_and_dispatches_candidate(
    packet_factory, pipeline_stub, run_watcher, watcher_repo
):
    item = packet_factory()
    pipeline_stub.next_candidate = {"id": "BACKLOG-NEXT", "state": "PROPOSED"}
    result = run_watcher([item])
    packet = watcher_repo / item["packet_dir"] / "PIPELINE.json"
    assert result.returncode == 0
    assert result.final["status"] == "complete"
    assert result.final["completion_action"] == "dispatch_next_candidate"
    assert json.loads(packet.read_text(encoding="utf-8"))["stage"] == "EXECUTED"
    assert result.final["audit_ready_ids"] == ["BACKLOG-TEST-01"]
    assert result.final["backlog_queue"]["priority_rebalance"]["requested"] is True
    assert any(call[0] == "rebalance" for call in pipeline_stub.calls)
    assert any(event["event"] == "watcher_finished" for event in result.events)


def test_clean_completion_reports_empty_queue(
    packet_factory, pipeline_stub, run_watcher
):
    pipeline_stub.status_items = []
    result = run_watcher([packet_factory()])
    assert result.returncode == 0
    assert result.final["completion_action"] == "notify_queue_empty"


def test_clean_completion_surfaces_blocked_leaf(
    packet_factory, pipeline_stub, run_watcher
):
    pipeline_stub.status_items = [{
        "id": "BACKLOG-BLOCKED",
        "state": "BLOCKED",
        "priority": 0,
        "priority_score": 80,
        "title": "Physical implementation",
        "next_action": "Implement and rerun",
    }]
    result = run_watcher([packet_factory()])
    assert result.returncode == 0
    assert result.final["completion_action"] == "resolve_blocked_candidate"
    assert result.final["backlog_queue"]["continuation_candidate"]["id"] == "BACKLOG-BLOCKED"


def test_non_experiment_mode_only_notifies(packet_factory, pipeline_stub, run_watcher):
    pipeline_stub.next_candidate = {"id": "BACKLOG-NEXT"}
    result = run_watcher([packet_factory()], experiment_mode=False)
    assert result.final["completion_action"] == "notify_completion"
    assert result.final["backlog_queue"]["priority_rebalance"]["requested"] is False
    assert not any(call[0] == "rebalance" for call in pipeline_stub.calls)


def test_missing_receipt_alerts_and_does_not_dispatch(packet_factory, run_watcher):
    result = run_watcher([packet_factory(receipt=False)])
    state = result.final["states"]["BACKLOG-TEST-01"]
    assert result.returncode == 2
    assert state["status"] == "failed_no_receipt"
    assert result.final["completion_action"] == "inspect_alert_before_dispatch"
    assert result.final["backlog_queue"]["priority_rebalance"]["requested"] is False


def test_missing_result_alerts_and_does_not_advance(
    packet_factory, pipeline_stub, run_watcher
):
    item = packet_factory(result=False)
    result = run_watcher([item])
    state = result.final["states"][item["task_id"]]
    assert state["status"] == "failed_no_result"
    assert not any(call[0] == "advance" for call in pipeline_stub.calls)


def test_packet_task_identity_mismatch_never_reuses_other_evidence(
    packet_factory, pipeline_stub, run_watcher
):
    item = packet_factory("BACKLOG-TASK-B", stage="EXECUTED")
    item["task_id"] = "BACKLOG-TASK-A"
    result = run_watcher([item])
    state = result.final["states"]["BACKLOG-TASK-A"]
    assert state["status"] == "failed_noncanonical_packet"
    assert result.final["audit_ready_ids"] == []
    assert not any(call[0] == "advance" for call in pipeline_stub.calls)


def test_pipeline_internal_task_id_must_match_canonical_task(
    packet_factory, watcher_repo, pipeline_stub, run_watcher
):
    item = packet_factory("BACKLOG-CANONICAL-TASK")
    pipeline = watcher_repo / item["packet_dir"] / "PIPELINE.json"
    pipeline.write_text(
        json.dumps({"task_id": "BACKLOG-OTHER-TASK", "stage": "IMPLEMENTED"}),
        encoding="utf-8",
    )
    result = run_watcher([item])
    state = result.final["states"][item["task_id"]]
    assert state["status"] == "failed_identity_mismatch"
    assert result.final["audit_ready_ids"] == []
    assert not any(call[0] == "advance" for call in pipeline_stub.calls)


def test_terminal_task_identity_mismatch_never_reuses_other_evidence(
    harness_packet, watcher_repo, pipeline_stub, run_watcher
):
    item = harness_packet("BACKLOG-TASK-B")
    item["task_id"] = "BACKLOG-TASK-A"
    pipeline = watcher_repo / item["packet_dir"] / "PIPELINE.json"
    pipeline.write_text(
        json.dumps({"task_id": "BACKLOG-TASK-A", "stage": "IMPLEMENTED"}),
        encoding="utf-8",
    )
    backlog_path = watcher_repo / "config/research_backlog.json"
    backlog = json.loads(backlog_path.read_text(encoding="utf-8"))
    backlog["items"] = [
        {
            "id": "BACKLOG-TASK-A",
            "state": "IMPLEMENTED",
            "packet_dir": item["packet_dir"],
        }
    ]
    backlog_path.write_text(json.dumps(backlog), encoding="utf-8")
    result = run_watcher([item], alive="never_observed")
    state = result.final["states"]["BACKLOG-TASK-A"]
    assert state["status"] == "failed_identity_mismatch"
    assert result.final["audit_ready_ids"] == []
    assert not any(call[0] == "advance" for call in pipeline_stub.calls)


def test_shadow_packet_outside_canonical_backlog_path_is_rejected(
    packet_factory, watcher_repo, pipeline_stub, run_watcher
):
    item = packet_factory("BACKLOG-SHADOW-TARGET", stage="EXECUTED")
    canonical = watcher_repo / item["packet_dir"]
    shadow = watcher_repo / "shadow/BACKLOG-SHADOW-TARGET"
    shadow.parent.mkdir()
    shutil.copytree(canonical, shadow)
    item["packet_dir"] = "shadow/BACKLOG-SHADOW-TARGET"
    result = run_watcher([item])
    state = result.final["states"][item["task_id"]]
    assert state["status"] == "failed_noncanonical_packet"
    assert result.final["audit_ready_ids"] == []
    assert not any(call[0] == "advance" for call in pipeline_stub.calls)


def test_explicit_unmanaged_canary_completes_without_audit_or_transition(
    packet_factory, pipeline_stub, run_watcher
):
    item = packet_factory("UNMANAGED-CANARY", stage="EXECUTED")
    item["managed_backlog"] = False
    result = run_watcher([item])
    state = result.final["states"][item["task_id"]]
    assert result.returncode == 0
    assert state["status"] == "completed_unmanaged"
    assert result.final["audit_ready_ids"] == []
    assert not any(call[0] == "advance" for call in pipeline_stub.calls)


@pytest.mark.parametrize(
    ("returncode", "timed_out", "expected"),
    [(7, False, "failed_worker_exit"), (1, True, "failed_worker_timeout")],
)
def test_worker_failure_after_evidence_never_becomes_audit_ready(
    packet_factory,
    worker_exit_factory,
    pipeline_stub,
    run_watcher,
    returncode,
    timed_out,
    expected,
):
    item = worker_exit_factory(
        packet_factory(stage="EXECUTED"),
        returncode=returncode,
        timed_out=timed_out,
    )
    result = run_watcher([item])
    state = result.final["states"][item["task_id"]]
    assert state["status"] == expected
    assert result.final["audit_ready_ids"] == []
    assert not any(call[0] == "advance" for call in pipeline_stub.calls)


def test_missing_worker_exit_receipt_stops_at_deadline(
    packet_factory, pipeline_stub, run_watcher
):
    item = packet_factory()
    item.update(
        {
            "require_worker_exit": True,
            "worker_exit_path": "missing-worker-exit.json",
            "run_id": "missing-exit",
            "deadline_epoch": 0,
        }
    )
    result = run_watcher([item])
    state = result.final["states"][item["task_id"]]
    assert state["status"] == "failed_worker_timeout"
    assert not any(call[0] == "advance" for call in pipeline_stub.calls)


def test_finalize_requires_configured_worker_exit_receipt(
    packet_factory, pipeline_stub
):
    item = packet_factory()
    item.update(
        {
            "require_worker_exit": True,
            "worker_exit_path": "missing-worker-exit.json",
            "run_id": "missing-exit",
        }
    )
    outcome = watcher.finalize_experiment(item, "pytest fixture")
    assert outcome["status"] == "failed_no_worker_exit"
    assert not any(call[0] == "advance" for call in pipeline_stub.calls)


def test_invalid_worker_exit_receipt_is_fail_closed(
    packet_factory, worker_exit_factory, pipeline_stub, run_watcher
):
    item = worker_exit_factory(
        packet_factory(), schema="wrong-worker-exit-schema"
    )
    result = run_watcher([item])
    state = result.final["states"][item["task_id"]]
    assert state["status"] == "failed_invalid_worker_exit"
    assert not any(call[0] == "advance" for call in pipeline_stub.calls)


def test_boolean_worker_returncode_is_not_integer_zero(
    packet_factory, worker_exit_factory, pipeline_stub, run_watcher
):
    item = worker_exit_factory(packet_factory(), returncode=False)
    result = run_watcher([item])
    state = result.final["states"][item["task_id"]]
    assert state["status"] == "failed_invalid_worker_exit"
    assert not any(call[0] == "advance" for call in pipeline_stub.calls)


def test_incomplete_progress_alerts_and_does_not_advance(
    packet_factory, pipeline_stub, run_watcher
):
    result = run_watcher([packet_factory(progress=0)])
    state = result.final["states"]["BACKLOG-TEST-01"]
    assert result.returncode == 2
    assert state["status"] == "failed_incomplete_progress"
    assert not any(call[0] == "advance" for call in pipeline_stub.calls)


def test_pid_never_observed_alerts(packet_factory, run_watcher):
    result = run_watcher([packet_factory()], alive="never_observed")
    state = result.final["states"]["BACKLOG-TEST-01"]
    assert result.returncode == 2
    assert state["status"] == "failed_pid_never_observed"


def test_sealed_harness_terminal_handles_short_process_without_progress_marker(
    harness_packet, run_watcher
):
    result = run_watcher([harness_packet()], alive="never_observed")
    state = result.final["states"]["BACKLOG-HARNESS-WATCH"]
    assert result.returncode == 0
    assert state["status"] == "executed_valid"
    assert state["finalization"]["harness_terminal"]["status"] == "SEALED"


def test_aborted_harness_terminal_never_advances_pipeline(
    harness_packet, pipeline_stub, run_watcher
):
    result = run_watcher([harness_packet(status="ABORTED")], alive="never_observed")
    state = result.final["states"]["BACKLOG-HARNESS-WATCH"]
    assert result.returncode == 2
    assert state["status"] == "failed_harness_aborted"
    assert not any(call[0] == "advance" for call in pipeline_stub.calls)


def test_mutated_harness_terminal_never_advances_pipeline(
    harness_packet, pipeline_stub, run_watcher
):
    result = run_watcher([harness_packet(mutate=True)], alive="never_observed")
    state = result.final["states"]["BACKLOG-HARNESS-WATCH"]
    assert result.returncode == 2
    assert state["status"] == "failed_invalid_harness_terminal"
    assert not any(call[0] == "advance" for call in pipeline_stub.calls)


def test_required_missing_harness_terminal_is_fail_closed(
    packet_factory, pipeline_stub, run_watcher
):
    item = packet_factory()
    item["require_harness_terminal"] = True
    result = run_watcher([item])
    assert result.returncode == 2
    assert result.final["states"]["BACKLOG-TEST-01"]["status"] == "failed_no_harness_terminal"
    assert not any(call[0] == "advance" for call in pipeline_stub.calls)


def test_malformed_harness_terminal_is_fail_closed(
    packet_factory, watcher_repo, pipeline_stub, run_watcher
):
    item = packet_factory(receipt=False, progress=0)
    item["require_harness_terminal"] = True
    terminal = watcher_repo / item["packet_dir"] / "raw/run.terminal.json"
    terminal.write_text("{invalid", encoding="utf-8")
    result = run_watcher([item], alive="never_observed")
    assert result.returncode == 2
    assert result.final["states"]["BACKLOG-TEST-01"]["status"] == "failed_invalid_harness_terminal"


def test_unknown_but_parseable_harness_terminal_status_is_fail_closed(
    watcher_repo, packet_factory, pipeline_stub, run_watcher, monkeypatch
):
    item = packet_factory(receipt=True, progress=0)
    item["require_harness_terminal"] = True
    monkeypatch.setattr(
        watcher,
        "harness_terminal",
        lambda _packet: {
            "present": True,
            "valid": True,
            "status": "UNKNOWN",
            "sample_count": 0,
            "errors": [],
        },
    )
    result = run_watcher([item])
    assert result.final["states"]["BACKLOG-TEST-01"]["status"] == "failed_invalid_harness_terminal"
    assert not any(call[0] == "advance" for call in pipeline_stub.calls)
    assert not any(call[0] == "advance" for call in pipeline_stub.calls)


def test_unexpected_packet_stage_fails_validation(packet_factory, run_watcher):
    result = run_watcher([packet_factory(stage="PREREGISTERED")])
    state = result.final["states"]["BACKLOG-TEST-01"]
    assert result.returncode == 2
    assert state["status"] == "failed_validation"
    assert "unexpected pre-finalization stage" in state["finalization"]["advance"]["stderr"]


def test_advance_failure_fails_validation(packet_factory, pipeline_stub, run_watcher):
    pipeline_stub.advance_returncode = 1
    result = run_watcher([packet_factory()])
    assert result.returncode == 2
    assert result.final["states"]["BACKLOG-TEST-01"]["status"] == "failed_validation"


def test_gate_failure_fails_validation(packet_factory, pipeline_stub, run_watcher):
    pipeline_stub.gate_returncode = 1
    result = run_watcher([packet_factory()])
    assert result.returncode == 2
    assert result.final["pipeline_gate"]["returncode"] == 1
    assert result.final["completion_action"] == "inspect_alert_before_dispatch"


def test_unhealthy_service_stops_dispatch(packet_factory, run_watcher):
    result = run_watcher([packet_factory()], health=503)
    assert result.returncode == 2
    assert set(result.final["final_health"].values()) == {503}
    assert result.final["completion_action"] == "inspect_alert_before_dispatch"


def test_service_recovery_before_deadline_allows_completion(
    packet_factory, run_watcher, monkeypatch
):
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(watcher.time, "monotonic", lambda: next(ticks, 1.0))
    health = {
        "http://fixture/8080": [200, 200, 503, 200, 200],
        "http://fixture/8081": [200, 200, 503, 200, 200],
    }
    result = run_watcher(
        [packet_factory()], health=health, service_settle_seconds=10
    )
    assert result.returncode == 0
    assert set(result.final["final_health"].values()) == {200}


def test_mixed_multi_experiment_result_is_fail_closed(packet_factory, run_watcher):
    first = packet_factory("BACKLOG-GOOD")
    second = packet_factory("BACKLOG-BAD", receipt=False)
    result = run_watcher([first, second])
    assert result.returncode == 2
    assert result.final["states"]["BACKLOG-GOOD"]["status"] == "executed_valid"
    assert result.final["states"]["BACKLOG-BAD"]["status"] == "failed_no_receipt"
    assert result.final["completion_action"] == "inspect_alert_before_dispatch"


def test_staggered_multi_experiment_polling_skips_already_finished_item(
    packet_factory, run_watcher, monkeypatch
):
    first = packet_factory("BACKLOG-FIRST")
    second = packet_factory("BACKLOG-SECOND")
    sleeps = []
    monkeypatch.setattr(watcher.time, "sleep", sleeps.append)
    result = run_watcher(
        [first, second],
        alive={first["pid"]: 1, second["pid"]: 3},
    )
    assert result.returncode == 0
    assert sleeps == [5]
    finished = [event["task_id"] for event in result.events if event["event"] == "experiment_finished"]
    assert finished == ["BACKLOG-FIRST", "BACKLOG-SECOND"]


def test_already_executed_packet_is_idempotent(packet_factory, pipeline_stub, run_watcher):
    result = run_watcher([packet_factory(stage="EXECUTED")])
    assert result.returncode == 0
    assert not any(call[0] == "advance" for call in pipeline_stub.calls)
    advance = result.final["states"]["BACKLOG-TEST-01"]["finalization"]["advance"]
    assert advance["stdout"] == "already EXECUTED"


@pytest.mark.parametrize("stdout", ["", "not-json", "{"])
def test_pipeline_json_parser_rejects_invalid_output(stdout):
    assert watcher.parse_pipeline_json({"returncode": 0, "stdout": stdout}) is None


def test_pipeline_json_parser_rejects_failed_command():
    assert watcher.parse_pipeline_json({"returncode": 1, "stdout": "[]"}) is None


def test_queue_snapshot_exposes_pipeline_failures(pipeline_stub):
    pipeline_stub.status_returncode = 1
    pipeline_stub.next_returncode = 1
    snapshot = watcher.backlog_queue_snapshot()
    assert snapshot["status_returncode"] == 1
    assert snapshot["next_returncode"] == 1
    assert snapshot["valid"] is False
    assert snapshot["state_counts"] == {}
    assert snapshot["next_candidate"] is None


def test_experiment_mode_rebalance_is_compact_and_fail_closed(
    packet_factory, pipeline_stub, run_watcher
):
    pipeline_stub.rebalance_returncode = 1

    result = run_watcher([packet_factory()])

    priority = result.final["backlog_queue"]["priority_rebalance"]
    assert result.returncode == 2
    assert priority == {
        "requested": True,
        "returncode": 1,
        "valid": False,
        "report": None,
    }
    assert result.final["completion_action"] == "inspect_alert_before_dispatch"


def test_queue_refresh_failure_is_an_alert_in_experiment_mode(
    packet_factory, pipeline_stub, run_watcher
):
    pipeline_stub.next_returncode = 1
    result = run_watcher([packet_factory()])
    assert result.returncode == 2
    assert result.final["backlog_queue"]["valid"] is False
    assert result.final["completion_action"] == "inspect_alert_before_dispatch"


def test_queue_refresh_failure_does_not_reclassify_non_experiment_completion(
    packet_factory, pipeline_stub, run_watcher
):
    pipeline_stub.next_returncode = 1
    result = run_watcher([packet_factory()], experiment_mode=False)
    assert result.returncode == 0
    assert result.final["completion_action"] == "notify_completion"


def test_detached_launcher_warns_and_does_not_wait(launcher_harness, capsys):
    result = launcher_harness(detach=True)
    captured = capsys.readouterr()
    assert result.returncode == 0
    assert result.launch["controller_binding"] == "detached_no_completion_delivery"
    assert result.processes[1].wait_calls == 0
    assert "will not wake the controlling session" in captured.err


def test_foreground_launcher_delivers_completion_payload(launcher_harness, capsys):
    result = launcher_harness()
    captured = capsys.readouterr()
    assert result.returncode == 0
    assert result.launch["controller_binding"] == "foreground_until_watcher_completion"
    assert result.processes[1].wait_calls == 1
    assert '"event":"watcher_completed"' in captured.out
    assert '"next_id":"BACKLOG-NEXT"' in captured.out
    assert '"audit_ready_ids":["BACKLOG-TEST"]' in captured.out
    assert '"next_candidate"' not in captured.out
    assert '"poll_seconds":300' in captured.out


def test_launcher_persists_real_worker_exit_code(launcher_harness):
    result = launcher_harness(worker_returncode=7, watcher_returncode=2)
    exit_path = pathlib.Path(result.config["experiments"][0]["worker_exit_path"])
    receipt = json.loads(exit_path.read_text(encoding="utf-8"))
    assert result.returncode == 2
    assert result.config["experiments"][0]["require_worker_exit"] is True
    assert result.processes[0].wait_calls == 1
    assert receipt["task_id"] == "BACKLOG-TEST"
    assert receipt["pid"] == result.processes[0].pid
    assert receipt["returncode"] == 7
    assert receipt["run_id"] == result.config["experiments"][0]["run_id"]


def test_launcher_stops_worker_when_watcher_exits_early(launcher_harness, capsys):
    result = launcher_harness(
        watcher_exits_early=True,
        watcher_returncode=9,
        final_payload=None,
    )
    captured = capsys.readouterr()
    receipt_path = pathlib.Path(result.config["experiments"][0]["worker_exit_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert result.returncode == 9
    assert result.processes[0].terminated is True
    assert receipt["watcher_failed_early"] is True
    assert "watcher exited before worker completion" in captured.err


def test_supervisor_requests_tree_termination_when_watcher_exits(monkeypatch):
    worker = FakeProcess(1001)
    dead_watcher = FakeProcess(1002, returncode=9)
    dead_watcher.running = False
    requested_tree_modes = []

    def terminate(process, *, tree=False):
        requested_tree_modes.append(tree)
        process.terminate()

    monkeypatch.setattr(launcher, "terminate_if_running", terminate)
    result = launcher.supervise_worker(worker, dead_watcher, 10)

    assert result["watcher_failed_early"] is True
    assert requested_tree_modes == [True]


def test_supervisor_stops_real_worker_when_real_watcher_exits_early():
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    dead_watcher = subprocess.Popen(
        [sys.executable, "-c", "raise SystemExit(9)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        started = time.monotonic()
        result = launcher.supervise_worker(worker, dead_watcher, 10)
        elapsed = time.monotonic() - started
    finally:
        launcher.terminate_if_running(worker)
        launcher.terminate_if_running(dead_watcher)
    assert result["watcher_failed_early"] is True
    assert result["watcher_returncode"] == 9
    assert worker.poll() is not None
    assert elapsed < 5


def test_tree_termination_stops_real_child_process(tmp_path):
    child_pid_path = tmp_path / "child.pid"
    parent_code = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code, str(child_pid_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        start_new_session=os.name != "nt",
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while not child_pid_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child_pid_path.is_file()
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert watcher.process_alive(child_pid)
        launcher.terminate_if_running(parent, tree=True)
        deadline = time.monotonic() + 5
        while watcher.process_alive(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert parent.poll() is not None
        assert watcher.process_alive(child_pid) is False
    finally:
        launcher.terminate_if_running(parent, tree=True)
        if child_pid is not None and watcher.process_alive(child_pid):
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(child_pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                os.kill(child_pid, 9)


def test_launcher_marks_explicit_unmanaged_canary(launcher_harness):
    result = launcher_harness(unmanaged_canary=True)
    assert result.config["experiments"][0]["managed_backlog"] is False


def test_harness_launcher_rejects_detached_mode(launcher_harness, capsys):
    with pytest.raises(SystemExit) as raised:
        launcher_harness(detach=True, require_terminal=True)
    assert raised.value.code == 2
    assert "require foreground control" in capsys.readouterr().err


def test_launcher_can_require_the_new_harness_terminal(launcher_harness):
    result = launcher_harness(require_terminal=True, omit_progress=True)
    assert result.returncode == 0
    assert result.config["experiments"][0]["require_harness_terminal"] is True
    assert result.config["experiments"][0]["progress_glob"] == "raw/run.terminal.json"
    assert result.config["experiments"][0]["progress_mode"] == "files"
    assert result.config["experiments"][0]["expected_progress"] == 1


def test_jsonl_progress_counts_records_not_the_single_file(
    packet_factory, watcher_repo, pipeline_stub
):
    item = packet_factory(progress=0)
    samples = watcher_repo / item["packet_dir"] / "raw/samples.jsonl"
    samples.write_text("".join('{\"row\":%d}\n' % index for index in range(448)), encoding="utf-8")
    item.update({
        "progress_glob": "raw/samples.jsonl",
        "progress_mode": "jsonl_lines",
        "expected_progress": 448,
    })

    assert watcher.progress_value(watcher_repo / item["packet_dir"], item) == 448
    assert watcher.finalize_experiment(item, "fixture")["status"] == "executed_valid"


def test_launcher_persists_jsonl_progress_mode(launcher_harness):
    result = launcher_harness(progress_mode="jsonl_lines")
    assert result.config["experiments"][0]["progress_mode"] == "jsonl_lines"


def test_launcher_rejects_nonpositive_expected_progress(launcher_harness, capsys):
    with pytest.raises(SystemExit) as raised:
        launcher_harness(expected_progress=0)
    assert raised.value.code == 2
    assert "expected-progress must be positive" in capsys.readouterr().err


def test_legacy_launcher_still_requires_explicit_progress_contract(
    launcher_harness, capsys
):
    with pytest.raises(SystemExit) as raised:
        launcher_harness(omit_progress=True)
    assert raised.value.code == 2
    assert "legacy runs require" in capsys.readouterr().err


def test_experiment_spawn_failure_is_recorded(launcher_harness, capsys):
    result = launcher_harness(spawn_error_at=1)
    captured = capsys.readouterr()
    assert result.returncode == 2
    assert result.failure["phase"] == "experiment_spawn"
    assert "experiment launch failed" in captured.err


def test_existing_runner_logs_are_never_truncated(launcher_harness, capsys):
    result = launcher_harness(existing_logs=True, spawn_error_at=1)
    captured = capsys.readouterr()
    assert result.returncode == 2
    assert result.failure["phase"] == "log_initialization"
    assert result.popen_calls == []
    assert (result.packet / "runner.stdout.log").read_text(encoding="utf-8") == "preserved stdout"
    assert (result.packet / "runner.stderr.log").read_text(encoding="utf-8") == "preserved stderr"
    assert "runner logs already exist" in captured.err


def test_launcher_rejects_nonimplemented_packet_before_creating_logs(
    launcher_harness, capsys
):
    result = launcher_harness(task_state="PROPOSED", spawn_error_at=1)
    captured = capsys.readouterr()
    assert result.returncode == 2
    assert result.failure["phase"] == "packet_validation"
    assert result.popen_calls == []
    assert not (result.packet / "runner.stdout.log").exists()
    assert not (result.packet / "runner.stderr.log").exists()
    assert "requires manifest and packet stage IMPLEMENTED" in captured.err


def test_watcher_spawn_failure_terminates_unwatched_experiment(
    launcher_harness, capsys
):
    result = launcher_harness(spawn_error_at=2)
    captured = capsys.readouterr()
    assert result.returncode == 2
    assert result.failure["phase"] == "watcher_spawn"
    assert result.processes[0].terminated is True
    assert "watcher launch failed" in captured.err


def test_missing_final_is_fail_closed(launcher_harness, capsys):
    result = launcher_harness(final_payload=None)
    captured = capsys.readouterr()
    assert result.returncode == 3
    assert "watcher exited without FINAL.json" in captured.err


def test_invalid_final_is_fail_closed(launcher_harness, capsys):
    result = launcher_harness(final_payload="invalid")
    captured = capsys.readouterr()
    assert result.returncode == 3
    assert "invalid FINAL.json" in captured.err


def test_missing_final_preserves_nonzero_watcher_exit(launcher_harness, capsys):
    result = launcher_harness(final_payload=None, watcher_returncode=9)
    captured = capsys.readouterr()
    assert result.returncode == 9
    assert "watcher exited without FINAL.json" in captured.err


def test_terminate_if_running_is_noop_for_finished_process():
    process = FakeProcess(1)
    process.running = False
    launcher.terminate_if_running(process)
    assert process.terminated is False


def test_terminate_if_running_kills_after_timeout():
    class StubbornProcess(FakeProcess):
        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                self.running = True
                raise subprocess.TimeoutExpired("canary", timeout)
            self.running = False
            return 0

        def terminate(self):
            self.terminated = True

    process = StubbornProcess(1)
    launcher.terminate_if_running(process)
    assert process.terminated is True
    assert process.killed is True
    assert process.wait_calls == 2
