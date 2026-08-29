from __future__ import annotations

import json

import pytest

from tools.research import run_fleet_seeded_stability_r4 as seeded


def treatment(model: str = "qwen38") -> dict:
    return {
        "model": model,
        "temperature": 0.2,
        "top_p": 0.95,
        "seed": 20260826,
        "stream": False,
        "cache_prompt": False,
    }


def test_canary_is_excluded_but_complete_treatment_is_captured():
    canary = dict(treatment())
    canary["temperature"] = 0.0
    canary.pop("top_p")
    assert seeded.is_experimental_request(canary) is False
    assert seeded.is_experimental_request(treatment()) is True
    for key in ("top_p", "seed", "stream", "cache_prompt"):
        malformed = dict(treatment())
        malformed.pop(key)
        assert seeded.is_experimental_request(malformed) is False


def test_atomic_capture_pairs_request_with_response_id(monkeypatch, tmp_path):
    seeded._pending_calls.clear()
    monkeypatch.setattr(seeded, "_original_http", lambda url, payload, timeout: (200, {"id": "resp-1"}))
    request = treatment()
    assert seeded._capturing_http("http://local/v1/chat/completions", request)[0] == 200
    path = tmp_path / "samples.jsonl"
    seeded._capturing_append(path, {"response": {"id": "resp-1"}})
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["request"] == request
    assert row["request_sha256"] == seeded.canonical_json_sha256(request)
    assert row["captured_response_id"] == "resp-1"
    assert seeded._pending_calls == []


def test_canary_does_not_shift_atomic_queue(monkeypatch):
    seeded._pending_calls.clear()
    counter = iter(("canary", "treatment"))
    monkeypatch.setattr(
        seeded,
        "_original_http",
        lambda url, payload, timeout: (200, {"id": next(counter)}),
    )
    canary = dict(treatment())
    canary["temperature"] = 0.0
    canary.pop("top_p")
    seeded._capturing_http("http://local/v1/chat/completions", canary)
    seeded._capturing_http("http://local/v1/chat/completions", treatment())
    assert [call["response_id"] for call in seeded._pending_calls] == ["treatment"]


def test_response_id_mismatch_fails_closed(monkeypatch, tmp_path):
    seeded._pending_calls[:] = [{"request": treatment(), "response_id": "expected"}]
    with pytest.raises(RuntimeError, match="response id mismatch"):
        seeded._capturing_append(tmp_path / "samples.jsonl", {"response": {"id": "wrong"}})


def test_exact_binary_identity_is_required(monkeypatch):
    card = {
        "artifact": {"path": "/models/a.gguf", "sha256": "a" * 64, "bytes": 123},
        "runtime": {"binary": "/bin/llama-server"},
    }
    status = {"current_model": "m", "backend_healthy": True, "backend_pid": 7}
    process = {"pid": 7, "exe": "/bin/llama-server", "argv": ["/bin/llama-server", "-m", "/models/a.gguf", "--alias", "m"]}
    binary = {"resolved_path": "/bin/llama-server", "sha256": "b" * 64, "bytes": 456}
    artifact = {"resolved_path": "/models/a.gguf", "sha256": "a" * 64, "bytes": 123}
    monkeypatch.setattr(seeded, "_binary_ledger", lambda: {"/bin/llama-server": {"sha256": "b" * 64, "bytes": 456}})
    assert seeded.evaluate_process_identity("m", status, process, binary, artifact, card)["pass"] is True
    binary["sha256"] = "c" * 64
    result = seeded.evaluate_process_identity("m", status, process, binary, artifact, card)
    assert result["pass"] is False
    assert result["checks"]["binary_hash_match"] is False


def test_r3_hook_replacement_does_not_recurse(monkeypatch):
    card = {
        "artifact": {"path": "/models/a.gguf", "sha256": "a" * 64, "bytes": 123},
        "runtime": {"binary": "/bin/llama-server"},
    }
    status = {"current_model": "m", "backend_healthy": True, "backend_pid": 7}
    process = {"pid": 7, "exe": "/bin/llama-server", "argv": ["/bin/llama-server", "-m", "/models/a.gguf", "--alias", "m"]}
    binary = {"resolved_path": "/bin/llama-server", "sha256": "b" * 64, "bytes": 456}
    artifact = {"resolved_path": "/models/a.gguf", "sha256": "a" * 64, "bytes": 123}
    monkeypatch.setattr(seeded.r3, "evaluate_process_identity", seeded.evaluate_process_identity)
    monkeypatch.setattr(seeded, "_binary_ledger", lambda: {"/bin/llama-server": {"sha256": "b" * 64, "bytes": 456}})
    assert seeded.evaluate_process_identity("m", status, process, binary, artifact, card)["pass"] is True


def test_request_validator_preserves_captured_bytes(monkeypatch):
    suite, case_id, case = seeded.r2.base.cases()[0]
    expected = seeded.r2.base.fleet.payload_for("qwen38", suite, case)
    expected.update({"temperature": 0.2, "top_p": 0.95, "seed": 20260826})
    row = {
        "model": "qwen38",
        "suite": suite,
        "case_id": case_id,
        "request": expected,
        "request_sha256": seeded.canonical_json_sha256(expected),
        "captured_response_id": "resp",
        "response": {"id": "resp"},
    }
    original = json.dumps(row, sort_keys=True)
    assert seeded.validate_captured_requests([row])[0] is row
    assert json.dumps(row, sort_keys=True) == original
    row["request_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="request hash mismatch"):
        seeded.validate_captured_requests([row])
