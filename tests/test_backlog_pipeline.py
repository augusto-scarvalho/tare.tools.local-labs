from __future__ import annotations

import copy
import json
import pathlib

import pytest

from tools.analysis.backlog_pipeline import (
    DEFAULT_MANIFEST,
    MANIFEST_SCHEMA,
    RECEIPT_SCHEMA,
    REVIEW_SCHEMA,
    PRIORITY_POLICY_SCHEMA,
    admit_item,
    advance_item,
    canonical_json_sha256,
    externalize_source,
    gate_repository,
    implementation_digest,
    load_json,
    rank_backlog,
    rebalance_priorities,
    scaffold_packet,
    select_next,
    sha256_file,
    _validate_receipt,
    validate_independent_review,
    validate_manifest,
    validate_priority_policy,
)


def _task(*, state: str = "PROPOSED", evidence_class: str = "artifact_requalification") -> dict:
    required = {
        "provenance",
        "receipt_fingerprint",
        "acceptance_gates",
        "raw_samples",
        "artifact_hashes",
        "dataset_hashes",
        "scorer_hashes",
        "independent_evaluation",
    }
    return {
        "id": "BACKLOG-TEST-01",
        "priority": 0,
        "title": "Exercise the guarded backlog pipeline",
        "state": state,
        "evidence_class": evidence_class,
        "packet_dir": "runs/research/BACKLOG-TEST-01",
        "depends_on": [],
        "source_artifacts": [],
        "required_evidence": sorted(required),
        "acceptance_gates": [
            {"id": "quality", "metric": "quality", "operator": "eq", "threshold": True}
        ],
        "allowed_claim_codes": ["TEST_QUALIFIED", "TEST_REJECTED"],
        "forbidden_claims": ["production claim"],
        "next_action": "Freeze and run the test packet.",
        "blocker": None,
        "state_history": [
            {"from": None, "to": state, "actor": "Codex", "at": "2026-08-25T00:00:00Z"}
        ],
    }


def _write_manifest(root: pathlib.Path, task: dict | None = None) -> pathlib.Path:
    path = root / "config" / "research_backlog.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "updated_at": "2026-08-25T00:00:00Z",
                "policy": {"independent_review_required": True},
                "items": [task or _task()],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _priority_policy(item_id: str = "BACKLOG-TEST-01", *, score: int = 5) -> dict:
    dimensions = {
        "ecosystem_leverage": {"weight": 35, "description": "Cross-project gain."},
        "community_innovation": {"weight": 25, "description": "Public novelty."},
        "information_per_cost": {"weight": 20, "description": "Efficient evidence."},
        "evidence_readiness": {"weight": 10, "description": "Ready inputs."},
        "downstream_unlock": {"weight": 10, "description": "Unblocks decisions."},
    }
    return {
        "schema": PRIORITY_POLICY_SCHEMA,
        "updated_at": "2026-08-29T00:00:00Z",
        "dimensions": dimensions,
        "priority_bands": [
            {"priority": 0, "min_score": 80},
            {"priority": 1, "min_score": 60},
            {"priority": 2, "min_score": 40},
            {"priority": 3, "min_score": 0},
        ],
        "aging": {
            "after_days": 30,
            "period_days": 30,
            "points_per_period": 2,
            "max_bonus": 10,
        },
        "assessments": {
            item_id: {
                "scores": {name: score for name in dimensions},
                "reason": "Fixture portfolio assessment.",
                "change_trigger": "Fixture evidence changed.",
                "actor": "Independent fixture reviewer",
                "reviewed_at": "2026-08-29T00:00:00Z",
            }
        },
    }


def _complete_preregistration(path: pathlib.Path) -> None:
    path.write_text(
        """# Frozen test

## Hypothesis

The implementation passes its frozen quality gate.

## Frozen inputs

- No external input.

## Command

```powershell
python tools/run_test.py
```

## Factors

One deterministic control and one implementation arm.

## Acceptance gates

- `quality`: `quality eq True`

## Abort conditions

Abort on missing provenance or receipt data.

## Allowed claims

- `TEST_QUALIFIED`
""",
        encoding="utf-8",
    )


def _valid_receipt(task: dict) -> dict:
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "task_id": task["id"],
        "provenance": {
            "schema": "local-labs-experiment-provenance-v1",
            "command": "python tools/run_test.py",
            "repository": {"head": "a" * 40},
            "script": {"sha256": "b" * 64},
            "inputs": [],
            "environment": {"python": "3.12", "platform": "test"},
        },
        "provenance_complete": True,
        "gates": {
            "quality": {
                "metric": "quality",
                "operator": "eq",
                "threshold": True,
                "actual": True,
                "pass": True,
            }
        },
        "evidence": {code: True for code in task["required_evidence"]},
    }
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    return receipt


def _valid_review(receipt_sha256: str, implementation: str, verdict: str = "APPROVED") -> dict:
    negative = verdict in {"REJECTED", "BLOCKED"}
    return {
        "schema": REVIEW_SCHEMA,
        "reviewer": "Codex independent fixture",
        "verdict": verdict,
        "receipt_sha256": receipt_sha256,
        "implementation_digest": implementation,
        "promise": "The bounded experiment changes the target product decision.",
        "value": "Avoids investing in a treatment that does not deliver its promised gain.",
        "promise_met": not negative,
        "falsification_attempt": "Attack the decisive frozen quality gate.",
        "falsification_evidence": "Recomputed retained sample and receipt rows.",
        "falsification_outcome": "Failure reproduced." if negative else "No reversing failure found.",
        "decision_impact": "Reverse the target decision." if negative else "No decision change.",
        "false_negative_check": "Recompute with a stricter scorer and inspect retained controls.",
        "result_reversing": negative,
        "materiality": "RESULT_REVERSING" if negative else "NON_BLOCKING",
        "remedy": "RESCORE_RETAINED" if negative else "ACCEPT",
        "smallest_remedy": "Reuse retained evidence; do not rerun the full experiment.",
        "retained_evidence_reusable": True,
        "findings": [],
    }


def test_current_backlog_is_valid_and_selects_adapter_requalification():
    assert gate_repository(DEFAULT_MANIFEST) == []
    manifest = load_json(DEFAULT_MANIFEST)
    assert len(manifest["items"]) >= 15
    next_item = select_next(manifest)
    if next_item is not None:
        assert next_item["state"] == "PROPOSED"


def test_rank_uses_explicit_assessment_without_rewriting_unassessed_items():
    assessed = _task()
    assessed["priority"] = 2
    fallback = copy.deepcopy(assessed)
    fallback["id"] = "BACKLOG-TEST-02"
    fallback["packet_dir"] = "runs/research/BACKLOG-TEST-02"
    fallback["priority"] = 1
    manifest = {"schema": MANIFEST_SCHEMA, "items": [fallback, assessed]}

    report = rank_backlog(
        manifest,
        _priority_policy(),
        generated_at="2026-08-29T00:00:00Z",
    )

    assert [row["id"] for row in report["items"]] == ["BACKLOG-TEST-01", "BACKLOG-TEST-02"]
    first, second = report["items"]
    assert (first["score"], first["recommended_priority"], first["score_source"]) == (
        100.0,
        0,
        "explicit_assessment",
    )
    assert (second["score"], second["recommended_priority"], second["score_source"]) == (
        None,
        1,
        "current_priority",
    )
    assert assessed["priority"] == 2


def test_rank_hides_superseded_history_by_default_but_can_restore_it():
    predecessor = _task()
    successor = copy.deepcopy(predecessor)
    successor["id"] = "BACKLOG-TEST-02"
    successor["packet_dir"] = "runs/research/BACKLOG-TEST-02"
    successor["supersedes"] = predecessor["id"]
    manifest = {"schema": MANIFEST_SCHEMA, "items": [predecessor, successor]}
    policy = _priority_policy()

    current = rank_backlog(manifest, policy, generated_at="2026-08-30T00:00:00Z")
    historical = rank_backlog(
        manifest,
        policy,
        include_superseded=True,
        generated_at="2026-08-30T00:00:00Z",
    )

    assert [row["id"] for row in current["items"]] == [successor["id"]]
    assert current["superseded_history_count"] == 1
    assert {row["id"] for row in historical["items"]} == {predecessor["id"], successor["id"]}


def test_rank_does_not_churn_legacy_history_for_an_unrelated_policy_digest():
    task = _task()
    task["priority"] = 0
    task["priority_score"] = 100.0
    task["priority_history"] = [{
        "from": 1,
        "to": 0,
        "score": 100.0,
        "actor": "legacy fixture",
        "at": "2026-08-29T00:00:00Z",
        "reason": "Same assessment, older global policy digest.",
        "change_trigger": "Unrelated assessment was added.",
        "policy_sha256": "0" * 64,
    }]

    row = rank_backlog(
        {"schema": MANIFEST_SCHEMA, "items": [task]},
        _priority_policy(),
        generated_at="2026-08-30T00:00:00Z",
    )["items"][0]

    assert row["changed"] is False


def test_priority_policy_fails_closed_on_unknown_item_and_partial_scores():
    manifest = {"schema": MANIFEST_SCHEMA, "items": [_task()]}
    policy = _priority_policy("BACKLOG-UNKNOWN")
    policy["assessments"]["BACKLOG-UNKNOWN"]["scores"].pop("downstream_unlock")

    errors = validate_priority_policy(policy, manifest)

    assert any("unknown backlog item" in error for error in errors)
    assert any("scores must match" in error for error in errors)


def test_rank_adds_bounded_aging_without_changing_manifest():
    task = _task()
    task["priority"] = 1
    task["state_history"][0]["at"] = "2026-01-01T00:00:00Z"
    manifest = {"schema": MANIFEST_SCHEMA, "items": [task]}
    policy = _priority_policy(score=3)

    row = rank_backlog(
        manifest,
        policy,
        generated_at="2026-03-15T00:00:00Z",
    )["items"][0]

    assert (row["base_score"], row["waiting_days"], row["aging_bonus"], row["score"]) == (
        60.0,
        73,
        4.0,
        64.0,
    )
    assert task["priority"] == 1


def test_rebalance_dry_run_is_read_only_and_apply_is_atomic(tmp_path: pathlib.Path):
    task = _task()
    task["priority"] = 0
    manifest_path = _write_manifest(tmp_path, task)
    policy_path = tmp_path / "config/backlog_priority_policy.json"
    policy_path.write_text(json.dumps(_priority_policy(score=0)), encoding="utf-8")
    before = manifest_path.read_bytes()

    dry_run = rebalance_priorities(
        manifest_path,
        policy_path,
        "",
        apply=False,
        root=tmp_path,
        at="2026-08-29T01:00:00Z",
    )

    assert dry_run["mode"] == "dry_run"
    assert dry_run["change_count"] == 1
    assert manifest_path.read_bytes() == before

    applied = rebalance_priorities(
        manifest_path,
        policy_path,
        "Codex portfolio fixture",
        apply=True,
        root=tmp_path,
        at="2026-08-29T01:00:00Z",
    )
    item = load_json(manifest_path)["items"][0]

    assert applied["applied_ids"] == ["BACKLOG-TEST-01"]
    assert (item["priority"], item["priority_score"], item["state"]) == (3, 0.0, "PROPOSED")
    assert item["priority_history"][-1]["from"] == 0
    assert item["priority_history"][-1]["to"] == 3
    assert item["priority_history"][-1]["assessment_sha256"]
    assert gate_repository(manifest_path, root=tmp_path) == []

    revised_policy = load_json(policy_path)
    revised_policy["updated_at"] = "2026-08-29T02:00:00Z"
    revised_policy["assessments"]["BACKLOG-TEST-01"]["reason"] = "Revised fixture rationale."
    policy_path.write_text(json.dumps(revised_policy), encoding="utf-8")
    before_revision = manifest_path.read_bytes()
    revised = rebalance_priorities(
        manifest_path,
        policy_path,
        "",
        apply=False,
        root=tmp_path,
        at="2026-08-29T02:00:00Z",
    )
    assert revised["change_count"] == 1
    assert manifest_path.read_bytes() == before_revision


def test_select_next_uses_persisted_score_only_inside_priority_band():
    lower_score = _task()
    lower_score["id"] = "BACKLOG-A"
    lower_score["packet_dir"] = "runs/research/BACKLOG-A"
    lower_score["priority_score"] = 81
    higher_score = copy.deepcopy(lower_score)
    higher_score["id"] = "BACKLOG-Z"
    higher_score["packet_dir"] = "runs/research/BACKLOG-Z"
    higher_score["priority_score"] = 99
    manifest = {"schema": MANIFEST_SCHEMA, "items": [lower_score, higher_score]}

    assert select_next(manifest)["id"] == "BACKLOG-Z"


def test_admit_item_is_validated_and_atomic(tmp_path: pathlib.Path):
    manifest_path = _write_manifest(tmp_path)
    spec = _task()
    spec["id"] = "BACKLOG-TEST-02"
    spec["packet_dir"] = "runs/research/BACKLOG-TEST-02"
    for field in ("state", "state_history", "blocker"):
        spec.pop(field, None)
    spec_path = tmp_path / "config" / "admission.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    admitted = admit_item(manifest_path, spec_path, "Codex executor", root=tmp_path)

    assert admitted == "BACKLOG-TEST-02"
    manifest = load_json(manifest_path)
    item = next(item for item in manifest["items"] if item["id"] == admitted)
    assert item["state"] == "PROPOSED"
    assert item["state_history"][0]["actor"] == "Codex executor"
    assert gate_repository(manifest_path, root=tmp_path) == []

    before = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="duplicate backlog item"):
        admit_item(manifest_path, spec_path, "Codex executor", root=tmp_path)
    assert manifest_path.read_bytes() == before


def test_manifest_rejects_dependency_cycles():
    first = _task()
    second = copy.deepcopy(first)
    second["id"] = "BACKLOG-TEST-02"
    second["packet_dir"] = "runs/research/BACKLOG-TEST-02"
    first["depends_on"] = [second["id"]]
    second["depends_on"] = [first["id"]]
    manifest = {"schema": MANIFEST_SCHEMA, "items": [first, second]}

    errors = validate_manifest(manifest)

    assert any("dependency cycle" in error for error in errors)


def test_external_source_receipt_keeps_portable_manifest_valid(tmp_path: pathlib.Path):
    task = _task()
    task["source_artifacts"] = ["runs/research/local-large.gguf"]
    manifest_path = _write_manifest(tmp_path, task)
    source = tmp_path / task["source_artifacts"][0]
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"physical model derivative")
    evidence_relative = "runs/research/source_hashes.json"
    evidence = tmp_path / evidence_relative
    evidence.write_text(
        json.dumps(
            {
                task["source_artifacts"][0]: {
                    "sha256": sha256_file(source),
                    "bytes": source.stat().st_size,
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    externalize_source(
        manifest_path,
        task["id"],
        task["source_artifacts"][0],
        evidence_relative,
        "Codex portability",
        root=tmp_path,
    )
    source.unlink()

    assert validate_manifest(load_json(manifest_path), tmp_path) == []


def test_missing_source_without_external_receipt_is_rejected(tmp_path: pathlib.Path):
    task = _task()
    task["source_artifacts"] = ["runs/research/missing.gguf"]
    manifest_path = _write_manifest(tmp_path, task)

    errors = validate_manifest(load_json(manifest_path), tmp_path)

    assert any("missing path" in error for error in errors)


def test_scaffold_is_draft_only_and_placeholders_block_preregistration(tmp_path: pathlib.Path):
    manifest_path = _write_manifest(tmp_path)
    packet_dir = scaffold_packet(manifest_path, "BACKLOG-TEST-01", "Executor", root=tmp_path)

    receipt_template = load_json(packet_dir / "RECEIPT.template.json")
    assert receipt_template["gates"]["quality"]["threshold"] is True
    assert set(receipt_template["evidence"]) == set(_task()["required_evidence"])
    review_template = load_json(packet_dir / "REVIEW.template.json")
    assert review_template["schema"] == REVIEW_SCHEMA
    assert review_template["verdict"] == "PENDING"
    assert set(review_template) >= {
        "promise", "value", "falsification_attempt", "false_negative_check", "materiality", "remedy"
    }

    with pytest.raises(ValueError, match="placeholders"):
        advance_item(
            manifest_path,
            "BACKLOG-TEST-01",
            "PREREGISTERED",
            "Executor",
            root=tmp_path,
        )

    assert load_json(manifest_path)["items"][0]["state"] == "PROPOSED"
    assert load_json(packet_dir / "PIPELINE.json")["stage"] == "PROPOSED"


def test_full_happy_path_requires_valid_receipt_and_independent_review(tmp_path: pathlib.Path):
    manifest_path = _write_manifest(tmp_path)
    packet_dir = scaffold_packet(manifest_path, "BACKLOG-TEST-01", "Executor", root=tmp_path)
    _complete_preregistration(packet_dir / "PRE_REGISTRATION.md")
    advance_item(manifest_path, "BACKLOG-TEST-01", "PREREGISTERED", "Executor", root=tmp_path)

    implementation = tmp_path / "tools" / "run_test.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("print('measured')\n", encoding="utf-8")
    advance_item(
        manifest_path,
        "BACKLOG-TEST-01",
        "IMPLEMENTED",
        "Executor",
        root=tmp_path,
        implementation_paths=["tools/run_test.py"],
    )

    receipt_path = packet_dir / "raw" / "receipt.json"
    receipt_path.write_text(
        json.dumps(_valid_receipt(load_json(manifest_path)["items"][0]), indent=2) + "\n",
        encoding="utf-8",
    )
    advance_item(manifest_path, "BACKLOG-TEST-01", "EXECUTED", "Executor", root=tmp_path)

    (packet_dir / "RESULT.md").write_text("# Result\n\nAll frozen gates passed.\n", encoding="utf-8")
    packet = load_json(packet_dir / "PIPELINE.json")
    review = _valid_review(
        sha256_file(receipt_path),
        implementation_digest(packet["implementation"]),
    )
    review["reviewer"] = "Executor"
    review_path = packet_dir / "REVIEW.json"
    review_path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="independent of the executor"):
        advance_item(
            manifest_path,
            "BACKLOG-TEST-01",
            "VERIFIED",
            "Executor",
            root=tmp_path,
            claim_codes=["TEST_QUALIFIED"],
        )

    review["reviewer"] = "Codex"
    review_path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    advance_item(
        manifest_path,
        "BACKLOG-TEST-01",
        "VERIFIED",
        "Codex",
        root=tmp_path,
        claim_codes=["TEST_QUALIFIED"],
    )
    advance_item(manifest_path, "BACKLOG-TEST-01", "PROMOTED", "Codex", root=tmp_path)

    assert gate_repository(manifest_path, root=tmp_path) == []
    assert load_json(manifest_path)["items"][0]["state"] == "PROMOTED"


def test_frugal_review_requires_falsification_and_false_negative_search():
    review = _valid_review("a" * 64, "b" * 64)
    review["falsification_attempt"] = ""
    review["false_negative_check"] = ""

    errors = validate_independent_review(
        review,
        executor="Executor",
        receipt_sha256="a" * 64,
        implementation_digest_value="b" * 64,
        expected_verdict="APPROVED",
        require_frugal_contract=True,
    )

    assert "review requires falsification_attempt" in errors
    assert "review requires false_negative_check" in errors


def test_frugal_review_cannot_block_on_a_non_material_technical_finding():
    review = _valid_review("a" * 64, "b" * 64, "BLOCKED")
    review["result_reversing"] = False
    review["materiality"] = "NON_BLOCKING"

    errors = validate_independent_review(
        review,
        executor="Executor",
        receipt_sha256="a" * 64,
        implementation_digest_value="b" * 64,
        expected_verdict="BLOCKED",
        require_frugal_contract=True,
    )

    assert "negative review requires a proved result-reversing failure" in errors
    assert "negative review requires RESULT_REVERSING materiality" in errors


def test_frugal_review_forbids_full_rerun_while_retained_evidence_is_reusable():
    review = _valid_review("a" * 64, "b" * 64, "REJECTED")
    review["remedy"] = "RERUN_FULL"
    review["retained_evidence_reusable"] = True

    errors = validate_independent_review(
        review,
        executor="Executor",
        receipt_sha256="a" * 64,
        implementation_digest_value="b" * 64,
        expected_verdict="REJECTED",
        require_frugal_contract=True,
    )

    assert "full rerun is forbidden while retained evidence is reusable" in errors


def test_executed_to_blocked_requires_independent_frugal_review(tmp_path: pathlib.Path):
    manifest_path = _write_manifest(tmp_path)
    packet_dir = scaffold_packet(manifest_path, "BACKLOG-TEST-01", "Executor", root=tmp_path)
    _complete_preregistration(packet_dir / "PRE_REGISTRATION.md")
    advance_item(manifest_path, "BACKLOG-TEST-01", "PREREGISTERED", "Executor", root=tmp_path)
    implementation = tmp_path / "tools" / "run_test.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("print('measured')\n", encoding="utf-8")
    advance_item(
        manifest_path,
        "BACKLOG-TEST-01",
        "IMPLEMENTED",
        "Executor",
        root=tmp_path,
        implementation_paths=["tools/run_test.py"],
    )
    receipt_path = packet_dir / "raw" / "receipt.json"
    receipt_path.write_text(
        json.dumps(_valid_receipt(load_json(manifest_path)["items"][0]), indent=2) + "\n",
        encoding="utf-8",
    )
    advance_item(manifest_path, "BACKLOG-TEST-01", "EXECUTED", "Executor", root=tmp_path)
    (packet_dir / "RESULT.md").write_text("# Result\n\nAudit found a material defect.\n", encoding="utf-8")
    packet = load_json(packet_dir / "PIPELINE.json")
    review = _valid_review(
        sha256_file(receipt_path),
        implementation_digest(packet["implementation"]),
        "BLOCKED",
    )
    (packet_dir / "REVIEW.json").write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires an independent actor"):
        advance_item(
            manifest_path,
            "BACKLOG-TEST-01",
            "BLOCKED",
            "Executor",
            root=tmp_path,
            reason="Result-reversing audit defect.",
        )

    advance_item(
        manifest_path,
        "BACKLOG-TEST-01",
        "BLOCKED",
        "Codex independent fixture",
        root=tmp_path,
        reason="Result-reversing audit defect.",
    )
    assert gate_repository(manifest_path, root=tmp_path) == []


def test_invalid_receipt_cannot_advance_and_keeps_state_implemented(tmp_path: pathlib.Path):
    manifest_path = _write_manifest(tmp_path)
    packet_dir = scaffold_packet(manifest_path, "BACKLOG-TEST-01", "Executor", root=tmp_path)
    _complete_preregistration(packet_dir / "PRE_REGISTRATION.md")
    advance_item(manifest_path, "BACKLOG-TEST-01", "PREREGISTERED", "Executor", root=tmp_path)
    implementation = tmp_path / "tools" / "run_test.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("print('measured')\n", encoding="utf-8")
    advance_item(
        manifest_path,
        "BACKLOG-TEST-01",
        "IMPLEMENTED",
        "Executor",
        root=tmp_path,
        implementation_paths=["tools/run_test.py"],
    )
    (packet_dir / "raw" / "receipt.json").write_text('{"gates":{"quality":true}}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing provenance"):
        advance_item(manifest_path, "BACKLOG-TEST-01", "EXECUTED", "Executor", root=tmp_path)

    assert load_json(manifest_path)["items"][0]["state"] == "IMPLEMENTED"
    assert load_json(packet_dir / "PIPELINE.json")["stage"] == "IMPLEMENTED"


def test_receipt_cannot_change_frozen_gate_or_lie_about_pass():
    task = _task()
    receipt = _valid_receipt(task)
    receipt["gates"]["quality"]["threshold"] = False
    receipt["gates"]["quality"]["actual"] = False
    receipt["gates"]["quality"]["pass"] = True
    receipt.pop("receipt_fingerprint")
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)

    errors = _validate_receipt(receipt, task, require_passing_gates=True)

    assert any("changed frozen threshold" in error for error in errors)
    assert any("pass does not match" in error for error in errors)
    assert any("non-passing gate" in error for error in errors)


def test_proxy_evidence_class_can_never_be_promoted():
    task = _task(state="PROMOTED", evidence_class="proxy_realization")
    task["required_evidence"] = sorted(
        {
            "provenance",
            "receipt_fingerprint",
            "acceptance_gates",
            "raw_samples",
            "real_implementation",
            "semantic_parity",
            "paired_baseline",
            "hardware_metrics",
        }
    )
    errors = validate_manifest({"schema": MANIFEST_SCHEMA, "items": [task]})
    assert any("cannot be promoted" in error for error in errors)
