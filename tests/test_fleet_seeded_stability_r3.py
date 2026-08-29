from __future__ import annotations

import json
import pathlib

from tools.research import run_fleet_seeded_stability_r3 as seeded


def identity_fixture() -> tuple[dict, dict, dict, dict, dict]:
    card = {
        "artifact": {"path": "/models/a.gguf", "sha256": "a" * 64, "bytes": 123},
        "runtime": {"binary": "/bin/llama-server"},
    }
    status = {"current_model": "m", "backend_healthy": True, "backend_pid": 7}
    process = {
        "pid": 7,
        "exe": "/bin/llama-server",
        "argv": ["/bin/llama-server", "-m", "/models/a.gguf", "--alias", "m"],
    }
    binary = {"resolved_path": "/bin/llama-server", "sha256": "b" * 64, "bytes": 456}
    artifact = {"resolved_path": "/models/a.gguf", "sha256": "a" * 64, "bytes": 123}
    return status, process, binary, artifact, card


def test_physical_identity_accepts_fully_bound_process():
    status, process, binary, artifact, card = identity_fixture()
    result = seeded.evaluate_process_identity("m", status, process, binary, artifact, card)
    assert result["pass"] is True
    assert all(result["checks"].values())


def test_physical_identity_rejects_alias_model_hash_and_process_mismatches():
    status, process, binary, artifact, card = identity_fixture()
    status["current_model"] = "wrong"
    process["pid"] = 8
    process["argv"][2] = "/models/other.gguf"
    artifact["sha256"] = "c" * 64
    result = seeded.evaluate_process_identity("m", status, process, binary, artifact, card)
    assert result["pass"] is False
    assert result["checks"]["gateway_alias_match"] is False
    assert result["checks"]["backend_pid_match"] is False
    assert result["checks"]["model_hash_match"] is False
    assert result["checks"]["command_model_matches_hashed_path"] is False


def test_http_argument_is_bound_to_sample(monkeypatch, tmp_path: pathlib.Path):
    samples = tmp_path / "samples.jsonl"
    seeded._pending_requests.clear()
    monkeypatch.setattr(seeded, "_original_http", lambda url, payload, timeout: (200, {"ok": True}))
    request = {"model": "m", "temperature": 0.2, "seed": 20260826}
    assert seeded._capturing_http("http://local/v1/chat/completions", request)[0] == 200
    seeded._capturing_append(samples, {"model": "m", "response": {"ok": True}})
    row = json.loads(samples.read_text(encoding="utf-8"))
    assert row["request"] == request
    assert row["request_capture"] == "http_json_argument"
    assert row["request_sha256"] == seeded.canonical_json_sha256(request)
    assert seeded._pending_requests == []


def test_sample_without_captured_request_fails_closed(tmp_path: pathlib.Path):
    seeded._pending_requests.clear()
    try:
        seeded._capturing_append(tmp_path / "samples.jsonl", {"model": "m"})
    except RuntimeError as exc:
        assert "no captured HTTP request" in str(exc)
    else:
        raise AssertionError("missing request capture must fail closed")


def test_gate_requires_exact_physical_binding():
    assert seeded._eq_gate("physical_routes_bound", 4, 4)["pass"] is True
    assert seeded._eq_gate("physical_routes_bound", 3, 4)["pass"] is False
