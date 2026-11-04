"""Offline process-boundary tests; no model, GPU or network service is started."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "ops/qualified-model-fleet/embedding_reindex_server.sh"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc and flock")


@pytest.fixture
def runner(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    for name, body in {
        "nvidia-smi": "exit 1",
        "curl": "exit 1",
        "taskset": 'shift 2; exec "$@"',
        "sleep": "exit 0",
    }.items():
        f = binary / name
        f.write_text("#!/bin/bash\n" + body + "\n")
        f.chmod(0o755)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
               REINDEX_RUNTIME_DIR=str(runtime), REINDEX_MODEL=str(tmp_path / "model.gguf"),
               REINDEX_SERVER=str(tmp_path / "server"))

    def run(*args):
        return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True,
                              text=True, timeout=10)

    return run, runtime, env


def test_stale_pid_cannot_stop_an_unrelated_live_process(runner):
    run, runtime, _ = runner
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (runtime / "server.pid").write_text(str(proc.pid) + "\n")
        result = run("stop")
        assert result.returncode == 0
        assert "NOT_RUNNING" in result.stdout
        assert proc.poll() is None
        assert not (runtime / "server.pid").exists()
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_stop_terminates_the_recorded_configured_server(runner):
    run, runtime, env = runner
    server = Path(env["REINDEX_SERVER"])
    server.write_text("import time; time.sleep(30)\n")
    proc = subprocess.Popen([sys.executable, str(server), "-m", env["REINDEX_MODEL"],
                             "--port", "8082"])
    try:
        (runtime / "server.pid").write_text(str(proc.pid) + "\n")
        result = run("stop")
        assert result.returncode == 0
        assert "REINDEX_SERVER_STOPPED" in result.stdout
        proc.wait(timeout=5)
        assert not (runtime / "server.pid").exists()
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


@pytest.mark.parametrize("pid", ["-1", "0", "not-a-pid"])
def test_invalid_pid_is_never_signalled(runner, pid):
    run, runtime, _ = runner
    (runtime / "server.pid").write_text(pid + "\n")
    result = run("stop")
    assert result.returncode == 0
    assert "NOT_RUNNING" in result.stdout


def test_status_without_nvidia_selects_cpu(runner):
    run, _, _ = runner
    result = run("status")
    assert result.returncode == 0
    assert "suggested_mode=cpu" in result.stdout


def test_missing_server_exits_without_leaving_pid(runner):
    run, runtime, _ = runner
    result = run("start", "cpu")
    assert result.returncode != 0
    assert not (runtime / "server.pid").exists()
    assert "READY" not in result.stdout


def test_unknown_mode_does_not_start_a_process(runner):
    run, runtime, _ = runner
    result = run("start", "invalid")
    assert result.returncode == 2
    assert not (runtime / "server.pid").exists()
