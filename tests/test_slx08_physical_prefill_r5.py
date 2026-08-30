from __future__ import annotations

import json

from tools.research import run_slx08_physical_prefill_r5 as runner


def _service(pid: int, started: str):
    return {
        "systemd": {
            "ActiveState": "active",
            "ExecStart": f"{{ path=/usr/bin/python3 ; argv[]=/usr/bin/python3 gateway.py --preload qwen38 ; ignore_errors=no ; start_time=[{started}] ; pid={pid} }}",
        },
        "health_status": 200,
        "health": {"role": "qualified-model-gateway", "current_model": "qwen38"},
    }


def test_stable_service_identity_ignores_pid_and_start_time():
    assert runner.stable_service_identity(_service(10, "before")) == runner.stable_service_identity(_service(20, "after"))


class _Stream:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        chunks = [
            {"stop": False, "content": ""},
            {"stop": False, "content": " 7"},
            {"stop": True, "content": "", "slx08_prefill": {"route": "selected_block"}},
        ]
        return iter([f"data: {json.dumps(chunk)}\n".encode() for chunk in chunks] + [b"data: [DONE]\n"])


def test_stream_completion_uses_first_nonempty_content_and_final_telemetry(monkeypatch):
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *_args, **_kwargs: _Stream())
    status, final, content, ttft_ms = runner.stream_completion({"prompt": [1], "stream": True})
    assert status == 200
    assert content == " 7"
    assert ttft_ms >= 0
    assert final["slx08_prefill"]["route"] == "selected_block"


def test_absolute_semantic_floor_blocks_empty_relative_parity():
    metrics = {
        "physical_selected_block_prefill_requests": 64,
        "physical_dense_prefill_requests": 64,
        "selected_block_route_observation_rate": 1.0,
        "median_retained_attention_fraction": 0.5,
        "dense_accuracy": 0.0,
        "selected_block_accuracy": 0.0,
        "paired_accuracy_delta_ci95_low": 0.0,
        "paired_p50_ttft_speedup": 2.0,
        "paired_p95_ttft_speedup": 2.0,
        "original_service_restored": 1,
        "embedding_health": 200,
    }
    gates = runner.evaluate_gates(metrics)
    assert gates["semantic_noninferiority"]["pass"] is True
    assert gates["dense_semantic_floor"]["pass"] is False
    assert gates["treatment_semantic_floor"]["pass"] is False
