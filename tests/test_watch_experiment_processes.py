import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import tools.analysis.launch_watched_experiment as launcher
import tools.analysis.watch_experiment_processes as watcher


@pytest.fixture
def watcher_repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
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
            json.dumps({"stage": stage}) + "\n", encoding="utf-8"
        )
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
def pipeline_stub(watcher_repo, monkeypatch):
    state = SimpleNamespace(
        calls=[],
        advance_returncode=0,
        gate_returncode=0,
        status_returncode=0,
        next_returncode=0,
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
    def __init__(self, pid, returncode=0, on_wait=None):
        self.pid = pid
        self.returncode = returncode
        self.on_wait = on_wait
        self.wait_calls = 0
        self.terminated = False
        self.killed = False
        self.running = True

    def poll(self):
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
        spawn_error_at=None,
        final_payload="default",
        watcher_returncode=0,
    ):
        packet = root / "packet"
        watch_outdir = root / "watch"
        packet.mkdir(exist_ok=True)
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
                "status": "complete",
                "experiment_mode": experiment_mode,
                "completion_action": "dispatch_next_candidate",
                "backlog_queue": {"next_candidate": {"id": "BACKLOG-NEXT"}},
            } if final_payload == "default" else final_payload
            path.write_text(json.dumps(payload), encoding="utf-8")

        def popen(*args, **kwargs):
            index = len(popen_calls) + 1
            popen_calls.append((args, kwargs))
            if spawn_error_at == index:
                raise OSError(f"spawn {index} failed")
            process = FakeProcess(
                1000 + index,
                returncode=watcher_returncode if index == 2 else 0,
                on_wait=write_final if index == 2 and not detach else None,
            )
            processes.append(process)
            return process

        monkeypatch.setattr(launcher.subprocess, "Popen", popen)
        argv = [
            "launch_watched_experiment.py",
            "--task-id", "BACKLOG-TEST",
            "--packet-dir", "packet",
            "--progress-glob", "raw/*.json",
            "--expected-progress", "1",
            "--watch-id", "WATCH-TEST",
            "--watch-outdir", "watch",
        ]
        if experiment_mode:
            argv.append("--experiment-mode")
        if detach:
            argv.append("--detach-watcher")
        argv.extend(["--", sys.executable, "-c", "pass"])
        monkeypatch.setattr(sys, "argv", argv)
        returncode = launcher.main()
        launch_path = watch_outdir / "LAUNCH.json"
        failure_path = watch_outdir / "LAUNCH_FAILED.json"
        return SimpleNamespace(
            returncode=returncode,
            processes=processes,
            popen_calls=popen_calls,
            launch=json.loads(launch_path.read_text(encoding="utf-8")) if launch_path.exists() else None,
            failure=json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else None,
        )

    return run


def test_process_alive_detects_current_process():
    assert watcher.process_alive(os.getpid())


def test_process_alive_rejects_impossible_pid():
    assert not watcher.process_alive(2_000_000_000)


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
    assert any(event["event"] == "watcher_finished" for event in result.events)


def test_clean_completion_reports_empty_queue(packet_factory, run_watcher):
    result = run_watcher([packet_factory()])
    assert result.returncode == 0
    assert result.final["completion_action"] == "notify_queue_empty"


def test_non_experiment_mode_only_notifies(packet_factory, pipeline_stub, run_watcher):
    pipeline_stub.next_candidate = {"id": "BACKLOG-NEXT"}
    result = run_watcher([packet_factory()], experiment_mode=False)
    assert result.final["completion_action"] == "notify_completion"


def test_missing_receipt_alerts_and_does_not_dispatch(packet_factory, run_watcher):
    result = run_watcher([packet_factory(receipt=False)])
    state = result.final["states"]["BACKLOG-TEST-01"]
    assert result.returncode == 2
    assert state["status"] == "failed_no_receipt"
    assert result.final["completion_action"] == "inspect_alert_before_dispatch"


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
    assert '"next_candidate"' not in captured.out
    assert '"poll_seconds":300' in captured.out


def test_experiment_spawn_failure_is_recorded(launcher_harness, capsys):
    result = launcher_harness(spawn_error_at=1)
    captured = capsys.readouterr()
    assert result.returncode == 2
    assert result.failure["phase"] == "experiment_spawn"
    assert "experiment launch failed" in captured.err


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
