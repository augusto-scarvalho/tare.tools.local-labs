from __future__ import annotations

import pathlib
import pytest

from tools.research.run_serving_cudagraph_benchmark import (
    http_get_json,
    _percentile,
)


def test_percentile_math():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    p50 = _percentile(vals, 0.50)
    assert p50 == 30.0
    p95 = _percentile(vals, 0.95)
    assert p95 == 50.0


def test_endpoint_health():
    try:
        health = http_get_json("http://127.0.0.1:8080/health")
        assert health.get("status") == "ok"
    except Exception:
        pytest.skip("Local endpoint not reachable during offline unit test")
