import pytest
from pathlib import Path

from tools.analysis.wsl_cpu_policy import (
    evaluate_policy,
    parse_cpu_set,
    parse_wsl_processors,
)


@pytest.fixture
def valid_snapshot():
    return {
        "configured_processors": 24,
        "kernel_online": "0-23",
        "experiment_allowed": "0-19",
        "serving_allowed": "0-23",
        "service_main_pid": "1234",
    }


def test_cpu_set_parser_handles_ranges_and_singletons():
    assert parse_cpu_set("0-3,8,10-11") == {0, 1, 2, 3, 8, 10, 11}


def test_cpu_set_parser_rejects_descending_ranges():
    with pytest.raises(ValueError, match="descending"):
        parse_cpu_set("3-1")


def test_wslconfig_fixture_reads_processor_limit():
    assert parse_wsl_processors("[wsl2]\nprocessors=24\nmemory=44GB\n") == 24


def test_canonical_policy_covers_systemd_interop_and_official_service():
    policy = Path(__file__).parents[1] / "ops/wsl/cpu-policy"
    manager = (policy / "90-local-labs-experiment-cpu.conf").read_text()
    interop = (policy / "local-labs-wsl-interop-affinity.service").read_text()
    serving = (policy / "llm-inference-cpu.conf").read_text()
    assert "CPUAffinity=0-19" in manager
    assert "taskset --pid --cpu-list 0-19 2" in interop
    assert "ExecCondition=/usr/bin/test /init -ef /proc/2/exe" in interop
    assert "CPUAffinity=0-23" in serving


def test_valid_policy_keeps_experiments_at_twenty_and_serving_at_twenty_four(valid_snapshot):
    report = evaluate_policy(valid_snapshot)
    assert report == {
        "ok": True,
        "configured_processors": 24,
        "kernel_online_vcpus": 24,
        "experiment_vcpus": 20,
        "serving_vcpus": 24,
        "service_main_pid": "1234",
        "errors": [],
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("configured_processors", 20, "configured_processors"),
        ("kernel_online", "0-19", "kernel_online"),
        ("experiment_allowed", "0-23", "experiment_allowed"),
        ("serving_allowed", "0-19", "serving_allowed"),
    ],
)
def test_policy_fails_closed_for_each_wrong_scope(valid_snapshot, field, value, message):
    valid_snapshot[field] = value
    report = evaluate_policy(valid_snapshot)
    assert report["ok"] is False
    assert any(message in error for error in report["errors"])
