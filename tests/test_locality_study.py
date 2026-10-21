"""Disposable native evidence only; no services, inference or operator files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.analysis import locality_study as study


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    native = {"session_id": "s1", "deployment": {"native_profile": {"model": "actual"}},
              "invocations": [{"started_at": 1792543891.0,
                               "ended_at": 1792543893.0, "status": "completed"},
                              {"started_at": 1792543895.0,
                               "ended_at": 1792543896.0, "status": "completed"}]}
    outcome = {"session_id": "s1", "status": "completed", "output": "answer"}
    usage = {"session_id": "s1", "calls": [
        {"call_id": "g1", "normalized": {"tokens": {"input_tokens": 20,
         "output_tokens": 5, "cache_read_tokens": 3}},
         "cost": {"known_amount": None, "currency": "USD", "complete": False}}],
        "coverage": "partial", "tokens": {"input_tokens": 20}, "unpriced_or_partial_calls": 1}
    events = [{"purpose": "count", "call_id": "c1", "session_id": "s1", "duration_ms": 900},
              {"purpose": "model", "call_id": "g1", "session_id": "s1", "duration_ms": 1000,
               "transport_calls": [{"first_content_ms": 125}]}]
    write(session / "native-agent.json", native)
    write(session / "outcome.json", outcome)
    write(session / "usage.json", usage)
    observations(session, events)
    trial = {"id": "t1", "case": "short", "route": "local", "session": str(session)}
    path = tmp_path / "study.json"
    write(path, {"schema": "tare.tools/locality-study/1", "trials": [trial]})
    return path, session, trial


def observations(session, events):
    (session / "observations.jsonl").write_text("\n".join(
        json.dumps({"event": event, "previous": None, "sequence": index})
        for index, event in enumerate(events)), encoding="utf-8")


def update(evidence, **fields):
    path, _, trial = evidence
    trial.update(fields)
    write(path, {"schema": "tare.tools/locality-study/1", "trials": [trial]})


def row(evidence):
    return study.analyze(evidence[0])["trials"][0]


def mutate(session, filename, **fields):
    path = session / filename
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(fields)
    write(path, value)


def test_native_measurements_and_hashes(evidence):
    report = study.analyze(evidence[0])
    result = report["trials"][0]
    assert result["active_wall_seconds"] == 3
    assert result["observed_generation_duration_ms"] == 1000
    assert result["generation_count"] == 1 and result["count_operations"] == 1
    assert result["first_content_ms_by_generation_transport"] == [[125]]
    assert result["quality"]["status"] == "unknown"
    assert result["energy"] is None
    assert len(report["source_sha256"]) == 5
    for path, sha in report["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
    assert report["input_bytes"] == sum(Path(p).stat().st_size for p in report["source_sha256"])


def test_missing_zero_partial_cost_never_free(evidence):
    result = row(evidence)
    assert result["cost"]["components"] == []
    assert "explicit_cost_components_missing" in result["cost"]["gaps"]
    components = [{"name": "API", "amount": "0", "currency": "USD", "basis": "call",
                   "source": "external invoice"},
                  {"name": "other", "amount": None, "currency": "USD", "basis": "trial",
                   "source": "unknown invoice"}]
    update(evidence, cost_components=components)
    result = row(evidence)
    assert result["cost"]["components"] == components
    assert result["cost"]["complete"] is False
    assert set(study.COST_GAPS) <= set(result["cost"]["gaps"])
    assert "unknown_component:other" in result["cost"]["gaps"]
    assert result["usage"]["calls"][0]["cost"]["known_amount"] is None
    assert result["usage"]["coverage"] == "partial"
    assert result["usage"]["unpriced_or_partial_calls"] == 1


@pytest.mark.parametrize("amount", ["-1", "NaN", "Infinity", 0, "", "1_000"])
def test_invalid_cost(evidence, amount):
    update(evidence, cost_components=[{"name": "x", "amount": amount, "currency": "USD",
                                      "basis": "call", "source": "receipt"}])
    with pytest.raises(ValueError):
        row(evidence)


def fleet(evidence, before, after, wrapped=False):
    root = evidence[0].parent
    paths = []
    for name, value in (("before", before), ("after", after)):
        path = root / (name + ".json")
        write(path, {"status": value, "observed_at": "2026-01-01T00:00:00Z"} if wrapped else value)
        paths.append(str(path))
    update(evidence, fleet_before=paths[0], fleet_after=paths[1])


def healthy(pid=42, model="actual"):
    return {"backend_pid": pid, "backend_healthy": True, "current_model": model,
            "last_switch_seconds": 99}


@pytest.mark.parametrize("wrapped", [False, True])
def test_cold_warm_and_no_double_count(evidence, wrapped):
    fleet(evidence, {"backend_pid": None, "backend_healthy": False, "current_model": None},
          healthy(), wrapped)
    result = row(evidence)
    assert result["residency"]["state"] == "cold_load"
    assert result["residency"]["last_switch_seconds"] == 99
    assert result["active_wall_seconds"] == 3
    assert result["observed_generation_duration_ms"] == 1000
    fleet(evidence, healthy(), healthy(), wrapped)
    result = row(evidence)["residency"]
    assert result["state"] == "warm"
    assert result["last_switch_seconds"] is None


@pytest.mark.parametrize("before,after", [
    (healthy(41), healthy(42)), (healthy(), healthy(model="alias")),
    (healthy(), healthy(pid=0)), (healthy(), healthy(pid=True)),
    ({"backend_pid": None, "backend_healthy": True, "current_model": None}, healthy()),
    ({"backend_pid": None, "backend_healthy": False, "current_model": "other"}, healthy()),
    (healthy(), dict(healthy(), backend_healthy=False))])
def test_ambiguous_residency(evidence, before, after):
    fleet(evidence, before, after)
    result = row(evidence)["residency"]
    assert result["state"] == "unknown" and result["reason"]
    assert result["last_switch_seconds"] is None


def test_remote_never_warm(evidence):
    fleet(evidence, healthy(), healthy())
    update(evidence, route="remote")
    assert row(evidence)["residency"]["state"] == "unknown"


@pytest.mark.parametrize("duration", [-1, "bad", float("nan"), float("inf")])
def test_invalid_generation_timing(evidence, duration):
    observations(evidence[1], [{"purpose": "model", "call_id": "g1", "session_id": "s1",
                                "duration_ms": duration}])
    with pytest.raises(ValueError):
        row(evidence)


def test_buffered_and_missing_durations_unknown(evidence):
    observations(evidence[1], [{"purpose": "model", "call_id": "g1", "session_id": "s1",
                                "transport_calls": [{}]}])
    result = row(evidence)
    assert result["observed_generation_duration_ms"] is None
    assert result["first_content_ms_by_generation_transport"] == [[None]]


@pytest.mark.parametrize("start,end,expected", [
    (1792543891.1159098, 1792544068.043707, 176.9277973),
    (1792543891, 1792543893, 2), (0, 0, 0), (-2, -1, 1),
    ("2026-01-01T00:00:00Z", "2026-01-01T00:00:02Z", 2)])
def test_native_epoch_active_timing_and_iso_compatibility(evidence, start, end, expected):
    mutate(evidence[1], "native-agent.json", invocations=[{"started_at": start, "ended_at": end}])
    assert row(evidence)["active_wall_seconds"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("invocations", [
    [{}], [{"started_at": 1792543891}],
    [{"started_at": "bad", "ended_at": 1792543893}],
    [{"started_at": None, "ended_at": None}],
    [{"started_at": True, "ended_at": 1792543893}],
    [{"started_at": {}, "ended_at": []}],
    [{"started_at": "2026-01-01T00:00:00", "ended_at": "2026-01-01T00:00:02"}],
    [{"started_at": 1, "ended_at": 2}, {"started_at": 3}]])
def test_unknown_active_timing(evidence, invocations):
    mutate(evidence[1], "native-agent.json", invocations=invocations)
    assert row(evidence)["active_wall_seconds"] is None


@pytest.mark.parametrize("start,end", [
    (1792543893.0, 1792543891.0),
    ("2026-01-01T00:00:02Z", "2026-01-01T00:00:01Z")])
def test_negative_active_elapsed_rejected(evidence, start, end):
    mutate(evidence[1], "native-agent.json", invocations=[{"started_at": start, "ended_at": end}])
    with pytest.raises(ValueError, match="negative timing"):
        row(evidence)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 10 ** 400])
@pytest.mark.parametrize("field", ["started_at", "ended_at"])
def test_nonfinite_active_timestamp_rejected(evidence, value, field):
    invocation = {"started_at": 1792543891.0, "ended_at": 1792543893.0, field: value}
    mutate(evidence[1], "native-agent.json", invocations=[invocation])
    with pytest.raises(ValueError):
        row(evidence)
    # Exercise timestamp validation itself, not only JSON's NaN/Infinity guard.
    with pytest.raises(ValueError):
        study.active_wall([invocation])


def test_failure_retained(evidence):
    mutate(evidence[1], "outcome.json", status="failed", output=None)
    result = study.analyze(evidence[0])["trials"]
    assert len(result) == 1 and result[0]["status"] == "failed"
    assert result[0]["quality"]["status"] == "unknown"


def test_duplicate_session_refused(evidence):
    path, _, trial = evidence
    write(path, {"schema": "tare.tools/locality-study/1",
                 "trials": [trial, dict(trial, id="t2")]})
    with pytest.raises(ValueError, match="duplicate session"):
        study.analyze(path)


@pytest.mark.parametrize("filename", ["native-agent.json", "outcome.json", "usage.json", "observations.jsonl"])
def test_identity_mismatch(evidence, filename):
    if filename == "observations.jsonl":
        observations(evidence[1], [{"purpose": "count", "session_id": "wrong"}])
    else:
        mutate(evidence[1], filename, session_id="wrong")
    with pytest.raises(ValueError, match="identity mismatch"):
        row(evidence)


@pytest.mark.parametrize("output", ["answer", {"z": [1, 2], "a": "é"}])
def test_external_assessment_binding(evidence, output):
    mutate(evidence[1], "outcome.json", output=output)
    answer = output if isinstance(output, str) else study.canonical(output)
    assessment = {"answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                  "checks": [{"id": "external-test", "passed": True}]}
    update(evidence, assessment=assessment)
    assert row(evidence)["quality"]["status"] == "passed"
    assessment["checks"][0]["passed"] = False
    update(evidence, assessment=assessment)
    assert row(evidence)["quality"]["status"] == "failed"
    assessment["answer_sha256"] = "0" * 64
    update(evidence, assessment=assessment)
    with pytest.raises(ValueError, match="answer hash mismatch"):
        row(evidence)


@pytest.mark.parametrize("checks", [[], [{"id": "x", "passed": 1}],
    [{"id": "x", "passed": True}, {"id": "x", "passed": False}]])
def test_assessment_checks_strict(evidence, checks):
    update(evidence, assessment={"answer_sha256": hashlib.sha256(b"answer").hexdigest(), "checks": checks})
    with pytest.raises(ValueError):
        row(evidence)


def test_energy_gross_only(evidence):
    sample = {"gpu_joules": 2.5, "attribution": "shared_gpu_gross", "source_sha256": "a" * 64}
    update(evidence, energy=sample)
    assert row(evidence)["energy"] == sample
    update(evidence, energy=dict(sample, attribution="whole_host"))
    with pytest.raises(ValueError):
        row(evidence)


def test_file_and_total_byte_ceilings(evidence, monkeypatch):
    monkeypatch.setattr(study, "FILE_LIMIT", 8)
    with pytest.raises(ValueError, match="byte ceiling"):
        row(evidence)
    monkeypatch.setattr(study, "FILE_LIMIT", 4 * 1024 * 1024)
    monkeypatch.setattr(study, "TOTAL_LIMIT", evidence[0].stat().st_size + 10)
    with pytest.raises(ValueError, match="byte ceiling"):
        row(evidence)


def test_cli_create_only(evidence, capsys):
    output = evidence[0].parent / "report.json"
    assert study.main([str(evidence[0]), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(output.read_text(encoding="utf-8"))
    original = output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        study.main([str(evidence[0]), "--output", str(output)])
    assert exc.value.code == 2 and output.read_bytes() == original
