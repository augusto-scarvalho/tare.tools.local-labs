from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import tools.analysis.run_watched_experiment_wave as wave


@pytest.fixture
def wave_repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "config").mkdir()
    monkeypatch.setattr(wave, "ROOT", root)
    monkeypatch.setattr(wave, "LAUNCHER", root / "tools/analysis/launch_watched_experiment.py")

    items = []
    for task_id in ("BACKLOG-A", "BACKLOG-B"):
        packet = root / "runs/research" / task_id
        packet.mkdir(parents=True)
        (packet / "PIPELINE.json").write_text(
            json.dumps({"task_id": task_id, "stage": "IMPLEMENTED"}), encoding="utf-8"
        )
        items.append({"id": task_id, "state": "IMPLEMENTED", "packet_dir": f"runs/research/{task_id}"})
    (root / "config/research_backlog.json").write_text(json.dumps({"items": items}), encoding="utf-8")
    manifest = {
        "schema": "local-labs-watched-wave-v1",
        "wave_id": "WAVE-FIXTURE",
        "poll_seconds": 5,
        "items": [
            {
                "task_id": task_id,
                "command": ["python", "runner.py", "--task-id", task_id],
                "progress": {"mode": "jsonl_lines", "glob": "raw/samples.jsonl", "expected": 3},
            }
            for task_id in ("BACKLOG-A", "BACKLOG-B")
        ],
    }
    manifest_path = root / "wave.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return SimpleNamespace(root=root, manifest=manifest, manifest_path=manifest_path, out=root / "wave-run")


def install_launcher_stub(monkeypatch, fixture, *, fail_on: str | None = None):
    calls = []

    def run(argv, stdout_path, stderr_path):
        task_id = argv[argv.index("--task-id") + 1]
        watch_dir = __import__("pathlib").Path(argv[argv.index("--watch-outdir") + 1])
        calls.append(task_id)
        if task_id == fail_on:
            return 2
        packet = fixture.root / "runs/research" / task_id / "PIPELINE.json"
        packet_payload = json.loads(packet.read_text())
        packet_payload["stage"] = "EXECUTED"
        packet.write_text(json.dumps(packet_payload), encoding="utf-8")
        backlog_path = fixture.root / "config/research_backlog.json"
        backlog = json.loads(backlog_path.read_text())
        next(item for item in backlog["items"] if item["id"] == task_id)["state"] = "EXECUTED"
        backlog_path.write_text(json.dumps(backlog), encoding="utf-8")
        wave.write_json(watch_dir / "FINAL.json", {
            "schema": "local-labs-experiment-watch-final-v1",
            "status": "complete",
            "audit_ready_ids": [task_id],
        })
        return 0

    monkeypatch.setattr(wave, "run_launcher", run)
    return calls


def test_wave_dispatches_every_frozen_item_in_order(wave_repo, monkeypatch):
    calls = install_launcher_stub(monkeypatch, wave_repo)

    assert wave.execute(wave_repo.manifest_path, wave_repo.out) == 0
    assert calls == ["BACKLOG-A", "BACKLOG-B"]
    final = wave.read_json(wave_repo.out / "FINAL.json")
    assert final["completed_ids"] == ["BACKLOG-A", "BACKLOG-B"]


def test_wave_stops_before_dispatching_after_an_alert(wave_repo, monkeypatch):
    calls = install_launcher_stub(monkeypatch, wave_repo, fail_on="BACKLOG-A")

    assert wave.execute(wave_repo.manifest_path, wave_repo.out) == 2
    assert calls == ["BACKLOG-A"]
    assert wave.read_json(wave_repo.out / "STATE.json")["status"] == "stopped_fail_closed"


def test_wave_resumes_after_completed_item_without_relaunch(wave_repo, monkeypatch):
    calls = install_launcher_stub(monkeypatch, wave_repo)
    assert wave.execute(wave_repo.manifest_path, wave_repo.out) == 0
    calls.clear()

    assert wave.execute(wave_repo.manifest_path, wave_repo.out) == 0
    assert calls == []


def test_wave_recovers_crash_after_watcher_executed_before_state_write(
    wave_repo, monkeypatch
):
    calls = install_launcher_stub(monkeypatch, wave_repo)
    assert wave.execute(wave_repo.manifest_path, wave_repo.out) == 0
    state_path = wave_repo.out / "STATE.json"
    state = wave.read_json(state_path)
    state["completed"] = [state["completed"][0]]
    state["status"] = "running"
    wave.write_json(state_path, state)
    calls.clear()

    assert wave.execute(wave_repo.manifest_path, wave_repo.out) == 0
    assert calls == []
    recovered = wave.read_json(state_path)["completed"][1]
    assert recovered["task_id"] == "BACKLOG-B"
    assert recovered["recovered"] is True


def test_wave_manifest_rejects_ambiguous_progress_contract(wave_repo):
    wave_repo.manifest["items"][0]["progress"]["mode"] = "guess"
    with pytest.raises(ValueError, match="unsupported progress mode"):
        wave.validate_manifest(wave_repo.manifest)


def test_wave_manifest_rejects_missing_task_identity(wave_repo):
    wave_repo.manifest["items"][0]["task_id"] = None
    with pytest.raises(ValueError, match="unique strings"):
        wave.validate_manifest(wave_repo.manifest)
