import subprocess
import sys

from tools.research.run_fleet_regression_screen_r3 import evaluate_gates


def good_metrics():
    return {
        "promoted_r2_verified": True,
        "route_models_completed": 4,
        "recorded_requests": 448,
        "successful_response_rate": 1.0,
        "route_identity_verified": True,
        "exact_repeat_rate": 0.95,
        "retained_request_payloads": 448,
        "service_restarts": 0,
        "embedding_health": 200,
        "initial_model_restored": True,
    }


def test_longitudinal_contract_accepts_exact_boundary_fixture():
    assert all(row["pass"] for row in evaluate_gates(good_metrics()).values())


def test_longitudinal_contract_rejects_each_incomplete_fixture():
    fixtures = {
        "r2_binding": ("promoted_r2_verified", False),
        "route_coverage": ("route_models_completed", 3),
        "request_coverage": ("recorded_requests", 447),
        "request_integrity": ("successful_response_rate", 0.99),
        "route_identity": ("route_identity_verified", False),
        "repeatability": ("exact_repeat_rate", 0.949),
        "request_retention": ("retained_request_payloads", 447),
        "service_integrity": ("service_restarts", 1),
        "embedding_integrity": ("embedding_health", 500),
        "service_recovery": ("initial_model_restored", False),
    }
    for gate, (metric, value) in fixtures.items():
        metrics = good_metrics()
        metrics[metric] = value
        assert evaluate_gates(metrics)[gate]["pass"] is False


def test_runner_is_directly_invocable():
    completed = subprocess.run(
        [sys.executable, "tools/research/run_fleet_regression_screen_r3.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
