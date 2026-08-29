import hashlib
import json

import pytest

from src.model_lifecycle.collectors.request import RequestResult
from src.model_lifecycle.experiment_harness import (
    ExperimentRun,
    HarnessStateError,
    PreflightContract,
    PreflightError,
    verify_run,
)


@pytest.fixture
def receipt_factory():
    def create(task_id="BACKLOG-HARNESS-TEST"):
        return {
            "schema": "local-labs-backlog-receipt-v1",
            "task_id": task_id,
            "provenance": {"fixture": True},
            "provenance_complete": True,
            "gates": {},
            "evidence": {"raw_samples": "raw/samples.jsonl"},
        }

    return create


@pytest.fixture
def good_request():
    return RequestResult(
        ok=True,
        answered=True,
        text="fixture answer",
        completion_tokens=8,
        prompt_tokens=5,
        predicted_n=8,
        predicted_ms=40.0,
    )


def _digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rewrite_terminal(raw, **changes):
    path = raw / "run.terminal.json"
    terminal = json.loads(path.read_text(encoding="utf-8"))
    terminal.update(changes)
    terminal.pop("terminal_fingerprint", None)
    terminal["terminal_fingerprint"] = _digest(terminal)
    path.write_text(json.dumps(terminal), encoding="utf-8")


def _rebind_file(raw, relative):
    path = raw / relative
    terminal = json.loads((raw / "run.terminal.json").read_text(encoding="utf-8"))
    terminal["files"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    _rewrite_terminal(raw, files=terminal["files"])


def _rewrite_journal(raw, events, *, renumber=True):
    previous = None
    for index, event in enumerate(events, 1):
        if renumber:
            event["seq"] = index
        event["prev_sha256"] = previous
        event["event_sha256"] = _digest(
            {key: value for key, value in event.items() if key != "event_sha256"}
        )
        previous = event["event_sha256"]
    path = raw / "run.events.jsonl"
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
    )
    _rebind_file(raw, "run.events.jsonl")
    _rewrite_terminal(raw, last_event_sha256=events[-1]["event_sha256"])


def test_successful_run_seals_and_verifies_without_changing_sample_schema(
    tmp_path, receipt_factory, good_request
):
    raw = tmp_path / "raw"
    sample = {"task_id": "fixture/1", "answer": 42, "correct": True}
    with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {"panel": ["fixture/1"]}) as run:
        run.preflight(
            PreflightContract("chat", require_server_timings=True),
            lambda: good_request,
        )
        run.record(sample)
        run.checkpoint("panel_complete", {"rows": 1})
        sealed_receipt = run.seal(receipt_factory())

    assert sealed_receipt["receipt_fingerprint"]
    assert json.loads((raw / "samples.jsonl").read_text(encoding="utf-8")) == sample
    report = verify_run(raw)
    assert report["valid"] is True
    assert report["status"] == "SEALED"
    assert report["sample_count"] == 1
    events = [json.loads(line) for line in (raw / "run.events.jsonl").read_text().splitlines()]
    assert [event["type"] for event in events] == [
        "STARTED", "PREFLIGHT_PASSED", "CHECKPOINT", "SEALED"
    ]


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (RequestResult(ok=False, answered=False, error="offline"), "transport/runtime failed"),
        (
            RequestResult(
                ok=True,
                answered=False,
                reasoning_text="thinking",
                completion_tokens=8,
                error="starved",
            ),
            "no usable answer",
        ),
        (RequestResult(ok=True, answered=True, text="", completion_tokens=8), "response text is empty"),
        (RequestResult(ok=True, answered=True, text="answer", completion_tokens=0), "completion token count"),
    ],
)
def test_preflight_failures_abort_instead_of_becoming_scientific_negatives(
    tmp_path, receipt_factory, result, message
):
    raw = tmp_path / "raw"
    with pytest.raises(PreflightError, match=message):
        with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
            run.preflight(PreflightContract("chat"), lambda: result)

    report = verify_run(raw)
    assert report["valid"] is True
    assert report["status"] == "ABORTED"
    assert not (raw / "receipt.json").exists()


def test_reasoning_only_can_be_an_explicit_measurement_contract(tmp_path, receipt_factory):
    raw = tmp_path / "raw"
    result = RequestResult(
        ok=True,
        answered=False,
        text="",
        reasoning_text="measured reasoning",
        completion_tokens=12,
        predicted_n=12,
        predicted_ms=60.0,
    )
    contract = PreflightContract(
        "reasoning-throughput",
        require_answered=False,
        require_text=False,
        require_server_timings=True,
    )
    with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
        assert run.preflight(contract, lambda: result) is result
        run.seal(receipt_factory())
    assert verify_run(raw)["valid"] is True


def test_preflight_requires_real_server_timings_when_declared(tmp_path):
    raw = tmp_path / "raw"
    result = RequestResult(ok=True, answered=True, text="answer", completion_tokens=3)
    with pytest.raises(PreflightError, match="server generation timings"):
        with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
            run.preflight(
                PreflightContract("timed-chat", require_server_timings=True),
                lambda: result,
            )


def test_service_experiment_cannot_seal_before_successful_restoration(
    tmp_path, receipt_factory
):
    raw = tmp_path / "raw"
    with pytest.raises(HarnessStateError, match="restoration"):
        with ExperimentRun(
            raw, "BACKLOG-HARNESS-TEST", {}, requires_restoration=True
        ) as run:
            run.record({"sample": 1})
            run.seal(receipt_factory())
    report = verify_run(raw)
    assert report["valid"] is True
    assert report["status"] == "ABORTED"


def test_successful_restoration_allows_service_experiment_to_seal(
    tmp_path, receipt_factory
):
    raw = tmp_path / "raw"
    with ExperimentRun(
        raw, "BACKLOG-HARNESS-TEST", {}, requires_restoration=True
    ) as run:
        run.restored({"model_restored": True, "embedding_http": 200})
        run.seal(receipt_factory())
    terminal = json.loads((raw / "run.terminal.json").read_text(encoding="utf-8"))
    assert terminal["restored"] is True
    assert verify_run(raw)["valid"] is True


def test_unhandled_exception_publishes_aborted_terminal_and_releases_lock(tmp_path):
    raw = tmp_path / "raw"
    with pytest.raises(RuntimeError, match="boom"):
        with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
            run.record({"partial": True})
            raise RuntimeError("boom")
    assert not (raw / ".experiment-run.lock").exists()
    report = verify_run(raw)
    assert report["valid"] is True
    assert report["status"] == "ABORTED"


def test_context_cannot_silently_finish_without_terminal_decision(tmp_path):
    raw = tmp_path / "raw"
    with pytest.raises(HarnessStateError, match="without seal"):
        with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}):
            pass
    assert verify_run(raw)["status"] == "ABORTED"


def test_existing_lock_and_existing_attempt_are_fail_closed(tmp_path, receipt_factory):
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / ".experiment-run.lock").write_text("owner", encoding="utf-8")
    with pytest.raises(HarnessStateError, match="another attempt"):
        ExperimentRun(locked, "BACKLOG-HARNESS-TEST", {}).__enter__()

    raw = tmp_path / "sealed"
    with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
        run.seal(receipt_factory())
    with pytest.raises(HarnessStateError, match="already exist"):
        ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}).__enter__()


@pytest.mark.parametrize("inputs", [{"bad": float("nan")}, {"bad": object()}])
def test_invalid_inputs_fail_before_lock_is_acquired(tmp_path, inputs):
    raw = tmp_path / "raw"
    with pytest.raises((TypeError, ValueError)):
        ExperimentRun(raw, "BACKLOG-HARNESS-TEST", inputs).__enter__()
    assert not (raw / ".experiment-run.lock").exists()


def test_receipt_with_incorrect_supplied_fingerprint_aborts(tmp_path, receipt_factory):
    raw = tmp_path / "raw"
    receipt = receipt_factory()
    receipt["receipt_fingerprint"] = "0" * 64
    with pytest.raises(HarnessStateError, match="receipt_fingerprint"):
        with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
            run.seal(receipt)
    assert verify_run(raw)["status"] == "ABORTED"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "wrong-schema", "receipt schema"),
        ("task_id", "BACKLOG-WRONG-TASK", "receipt task_id"),
    ],
)
def test_seal_rejects_receipt_identity_mismatch(
    tmp_path, receipt_factory, field, value, message
):
    raw = tmp_path / "raw"
    receipt = receipt_factory()
    receipt[field] = value
    with pytest.raises(HarnessStateError, match=message):
        with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
            run.seal(receipt)


def test_failed_restoration_cannot_be_misrecorded_as_success(tmp_path, receipt_factory):
    raw = tmp_path / "raw"
    with pytest.raises(HarnessStateError, match="restoration"):
        with ExperimentRun(
            raw, "BACKLOG-HARNESS-TEST", {}, requires_restoration=True
        ) as run:
            run.restored({"service_http": 500}, ok=False)
            run.seal(receipt_factory())


@pytest.mark.parametrize("mutation", ["change", "delete", "extra"])
def test_any_post_seal_raw_mutation_is_detected(tmp_path, receipt_factory, mutation):
    raw = tmp_path / "raw"
    with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
        run.record({"value": 1})
        run.seal(receipt_factory())
    if mutation == "change":
        (raw / "samples.jsonl").write_text('{"value":2}\n', encoding="utf-8")
    elif mutation == "delete":
        (raw / "samples.jsonl").unlink()
    else:
        (raw / "late.json").write_text("{}\n", encoding="utf-8")
    report = verify_run(raw)
    assert report["valid"] is False
    assert any(word in " ".join(report["errors"]) for word in ("changed", "missing", "unsealed"))


def test_terminal_fingerprint_tampering_is_detected(tmp_path, receipt_factory):
    raw = tmp_path / "raw"
    with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
        run.seal(receipt_factory())
    terminal_path = raw / "run.terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["sample_count"] = 99
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    report = verify_run(raw)
    assert report["valid"] is False
    assert "terminal fingerprint mismatch" in report["errors"]


@pytest.mark.parametrize(
    ("changes", "expected_error"),
    [
        ({"schema": "wrong-terminal-schema"}, "unsupported terminal schema"),
        ({"sample_count": 99}, "sample_count does not match"),
        (
            {"requires_restoration": True, "restored": False},
            "required restoration was not successful",
        ),
        ({"last_event_sha256": "0" * 64}, "does not bind the journal tail"),
        ({"status": "UNKNOWN"}, "status is not SEALED or ABORTED"),
        ({"status": "ABORTED"}, "status disagrees with the journal tail"),
    ],
)
def test_self_consistent_terminal_lies_are_rejected(
    tmp_path, receipt_factory, changes, expected_error
):
    raw = tmp_path / "raw"
    with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
        run.record({"fixture": True})
        run.seal(receipt_factory())
    _rewrite_terminal(raw, **changes)
    report = verify_run(raw)
    assert report["valid"] is False
    assert any(expected_error in error for error in report["errors"])


def test_terminal_beside_active_lock_is_rejected(tmp_path, receipt_factory):
    raw = tmp_path / "raw"
    with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
        run.seal(receipt_factory())
    (raw / ".experiment-run.lock").write_text("stale owner", encoding="utf-8")
    report = verify_run(raw)
    assert report["valid"] is False
    assert "active run lock remains beside terminal state" in report["errors"]


@pytest.mark.parametrize("payload", [[], None, "terminal"])
def test_parseable_non_object_terminal_is_invalid_not_an_exception(tmp_path, payload):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "run.terminal.json").write_text(json.dumps(payload), encoding="utf-8")
    report = verify_run(raw)
    assert report["valid"] is False
    assert report["status"] is None
    assert report["errors"] == ["terminal is not a JSON object"]


def test_non_utf8_terminal_is_invalid_not_an_exception(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "run.terminal.json").write_bytes(b"\xff\xfe")
    report = verify_run(raw)
    assert report["valid"] is False
    assert report["status"] is None
    assert "verification failed: UnicodeDecodeError" in report["errors"][0]


def test_nested_reserved_filename_remains_covered_by_manifest(tmp_path, receipt_factory):
    raw = tmp_path / "raw"
    nested = raw / "nested/run.terminal.json"
    with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
        nested.parent.mkdir()
        nested.write_text("before", encoding="utf-8")
        run.seal(receipt_factory())
    nested.write_text("after", encoding="utf-8")
    report = verify_run(raw)
    assert report["valid"] is False
    assert any("sealed files changed" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("field", "value", "recompute_receipt_fingerprint", "expected_error"),
    [
        ("provenance", {"fixture": False}, False, "receipt fingerprint mismatch"),
        ("task_id", "BACKLOG-WRONG-TASK", True, "receipt task_id does not match terminal"),
        ("schema", "wrong-receipt-schema", True, "receipt schema mismatch"),
    ],
)
def test_self_consistent_outer_manifest_does_not_hide_receipt_lies(
    tmp_path,
    receipt_factory,
    field,
    value,
    recompute_receipt_fingerprint,
    expected_error,
):
    raw = tmp_path / "raw"
    with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
        run.seal(receipt_factory())
    receipt_path = raw / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[field] = value
    if recompute_receipt_fingerprint:
        receipt.pop("receipt_fingerprint", None)
        receipt["receipt_fingerprint"] = _digest(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    _rebind_file(raw, "receipt.json")
    report = verify_run(raw)
    assert report["valid"] is False
    assert expected_error in report["errors"]


def test_journal_hash_chain_tampering_is_detected(tmp_path, receipt_factory):
    raw = tmp_path / "raw"
    with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
        run.checkpoint("halfway")
        run.seal(receipt_factory())
    journal = raw / "run.events.jsonl"
    events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    events[0]["payload"]["inputs_sha256"] = "0" * 64
    journal.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    report = verify_run(raw)
    assert report["valid"] is False
    assert any("event digest" in error for error in report["errors"])


def test_non_object_journal_event_is_invalid_not_an_exception(tmp_path, receipt_factory):
    raw = tmp_path / "raw"
    with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
        run.seal(receipt_factory())
    journal = raw / "run.events.jsonl"
    events = journal.read_text(encoding="utf-8").splitlines()
    events[0] = "[]"
    journal.write_text("\n".join(events) + "\n", encoding="utf-8")
    _rebind_file(raw, "run.events.jsonl")
    report = verify_run(raw)
    assert report["valid"] is False
    assert "journal line 1 is not an object" in report["errors"]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [("schema", "unsupported schema"), ("sequence", "non-monotonic seq")],
)
def test_rehashed_journal_metadata_tampering_is_detected(
    tmp_path, receipt_factory, mutation, expected
):
    raw = tmp_path / "raw"
    with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
        run.checkpoint("middle")
        run.seal(receipt_factory())
    events = [
        json.loads(line)
        for line in (raw / "run.events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if mutation == "schema":
        events[1]["schema"] = "wrong-journal-schema"
    else:
        events[1]["seq"] = 99
    _rewrite_journal(raw, events, renumber=False)
    report = verify_run(raw)
    assert report["valid"] is False
    assert any(expected in error for error in report["errors"])


@pytest.mark.parametrize("mutation", ["task_id", "chain"])
def test_rehashed_journal_semantic_tampering_is_detected(
    tmp_path, receipt_factory, mutation
):
    raw = tmp_path / "raw"
    with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
        run.checkpoint("halfway")
        run.seal(receipt_factory())
    journal = raw / "run.events.jsonl"
    events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    if mutation == "task_id":
        for event in events:
            event["task_id"] = "BACKLOG-WRONG-TASK"
    else:
        events[1]["prev_sha256"] = "0" * 64
    for index, event in enumerate(events):
        if index > 1 or (index > 0 and mutation == "task_id"):
            event["prev_sha256"] = events[index - 1]["event_sha256"]
        event["event_sha256"] = _digest(
            {key: value for key, value in event.items() if key != "event_sha256"}
        )
    journal.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
    )
    _rebind_file(raw, "run.events.jsonl")
    _rewrite_terminal(raw, last_event_sha256=events[-1]["event_sha256"])
    report = verify_run(raw)
    assert report["valid"] is False
    expected = "different task_id" if mutation == "task_id" else "breaks the hash chain"
    assert any(expected in error for error in report["errors"])


def test_rehashed_journal_cannot_remove_required_restoration(
    tmp_path, receipt_factory
):
    raw = tmp_path / "raw"
    with ExperimentRun(
        raw, "BACKLOG-HARNESS-TEST", {}, requires_restoration=True
    ) as run:
        run.restored({"service_http": 200})
        run.seal(receipt_factory())
    events = [
        json.loads(line)
        for line in (raw / "run.events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    _rewrite_journal(raw, [event for event in events if event["type"] != "RESTORED"])
    report = verify_run(raw)
    assert report["valid"] is False
    assert "SEALED journal lacks a final successful restoration event" in report["errors"]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("receipt_sha256", "0" * 64, "SEALED journal receipt_sha256 mismatch"),
        (
            "receipt_fingerprint",
            "0" * 64,
            "SEALED journal receipt_fingerprint mismatch",
        ),
        ("sample_count", 99, "terminal journal event sample_count does not match samples"),
    ],
)
def test_rehashed_sealed_event_lies_are_rejected(
    tmp_path, receipt_factory, field, value, expected
):
    raw = tmp_path / "raw"
    with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
        run.record({"fixture": True})
        run.seal(receipt_factory())
    events = [
        json.loads(line)
        for line in (raw / "run.events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    events[-1]["payload"][field] = value
    _rewrite_journal(raw, events)
    report = verify_run(raw)
    assert report["valid"] is False
    assert expected in report["errors"]


def test_rehashed_started_event_cannot_change_restoration_contract(
    tmp_path, receipt_factory
):
    raw = tmp_path / "raw"
    with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
        run.seal(receipt_factory())
    events = [
        json.loads(line)
        for line in (raw / "run.events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    events[0]["payload"]["requires_restoration"] = True
    _rewrite_journal(raw, events)
    report = verify_run(raw)
    assert report["valid"] is False
    assert "STARTED restoration contract does not match terminal" in report["errors"]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("failed_then_sealed", "PREFLIGHT_FAILED must transition directly to ABORTED"),
        ("duplicate_started", "exactly one STARTED"),
        ("unknown_event", "unknown event types"),
        ("missing_inputs", "STARTED inputs_sha256 is missing or invalid"),
        ("duplicate_terminal", "exactly one terminal event at the end"),
    ],
)
def test_rehashed_impossible_journal_sequences_are_rejected(
    tmp_path, receipt_factory, mutation, expected
):
    raw = tmp_path / "raw"
    with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
        run.seal(receipt_factory())
    events = [
        json.loads(line)
        for line in (raw / "run.events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if mutation == "failed_then_sealed":
        injected = dict(events[0])
        injected["type"] = "PREFLIGHT_FAILED"
        injected["payload"] = {"errors": ["fixture failure"]}
        events.insert(-1, injected)
    elif mutation == "duplicate_started":
        events.insert(1, json.loads(json.dumps(events[0])))
    elif mutation == "unknown_event":
        injected = dict(events[0])
        injected["type"] = "UNRECOGNIZED_EVENT"
        events.insert(-1, injected)
    elif mutation == "duplicate_terminal":
        events.insert(-1, json.loads(json.dumps(events[-1])))
    else:
        events[0]["payload"].pop("inputs_sha256")
    _rewrite_journal(raw, events)
    report = verify_run(raw)
    assert report["valid"] is False
    assert any(expected in error for error in report["errors"])


def test_replay_scores_immutable_samples_without_gpu(tmp_path, receipt_factory):
    raw = tmp_path / "raw"
    with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
        run.record({"correct": True})
        run.record({"correct": False})
        run.record({"correct": True})
        run.seal(receipt_factory())

    scorer = lambda rows: {"correct": sum(row["correct"] for row in rows), "n": len(rows)}
    report = verify_run(raw, scorer, expected_replay={"correct": 2, "n": 3})
    assert report["valid"] is True
    assert report["replay_matches"] is True
    mismatch = verify_run(raw, scorer, expected_replay={"correct": 3, "n": 3})
    assert mismatch["valid"] is False
    assert mismatch["replay_matches"] is False


def test_scorer_failure_is_verification_data_not_a_crash(tmp_path, receipt_factory):
    raw = tmp_path / "raw"
    with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
        run.record({"value": 1})
        run.seal(receipt_factory())

    def broken(_rows):
        raise ValueError("bad scorer")

    report = verify_run(raw, broken)
    assert report["valid"] is False
    assert "scorer replay failed" in report["errors"][0]


def test_unfinished_temporary_artifact_prevents_seal(tmp_path, receipt_factory):
    raw = tmp_path / "raw"
    with pytest.raises(HarnessStateError, match="temporary artifact"):
        with ExperimentRun(raw, "BACKLOG-HARNESS-TEST", {}) as run:
            (raw / "partial.json.tmp").write_text("partial", encoding="utf-8")
            run.seal(receipt_factory())
    # The abort terminal intentionally covers the partial file as failure evidence.
    assert verify_run(raw)["status"] == "ABORTED"
