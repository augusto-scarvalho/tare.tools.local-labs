#!/usr/bin/env python3
"""Fail-closed research backlog and implementation pipeline.

The pipeline turns the current research backlog into an executable state
machine. It freezes preregistration before implementation, implementation before
execution, execution before review, and review before promotion. Gemini may
execute work, but it cannot independently approve its own packet.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import pathlib
import re
import sys
import tempfile
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "config" / "research_backlog.json"
DEFAULT_PRIORITY_POLICY = ROOT / "config" / "backlog_priority_policy.json"

MANIFEST_SCHEMA = "local-labs-backlog-v1"
PRIORITY_POLICY_SCHEMA = "local-labs-backlog-priority-policy-v1"
PRIORITY_REPORT_SCHEMA = "local-labs-backlog-priority-report-v1"
PACKET_SCHEMA = "local-labs-backlog-packet-v1"
REVIEW_SCHEMA = "local-labs-independent-review-v1"
RECEIPT_SCHEMA = "local-labs-backlog-receipt-v1"

STATES = {
    "PROPOSED",
    "PREREGISTERED",
    "IMPLEMENTED",
    "EXECUTED",
    "VERIFIED",
    "PROMOTED",
    "REJECTED",
    "BLOCKED",
}
TERMINAL_STATES = {"PROMOTED", "REJECTED"}
DEPENDENCY_SUCCESS_STATES = {"PROMOTED"}
PROMOTABLE_CLASSES = {
    "artifact_requalification",
    "external_reproduction",
    "human_calibration",
    "kv_codec",
    "mechanism_research",
    "model_training",
    "model_requalification",
    "provenance_reconciliation",
    "distillation",
    "serving_runtime",
    "kernel_hardware",
    "packed_artifact",
}
GATE_OPERATORS = {"eq", "ne", "ge", "gt", "le", "lt"}

LEGAL_TRANSITIONS: dict[str | None, set[str]] = {
    None: {"PROPOSED", "BLOCKED"},
    "PROPOSED": {"PREREGISTERED", "BLOCKED"},
    "PREREGISTERED": {"IMPLEMENTED", "BLOCKED"},
    "IMPLEMENTED": {"EXECUTED", "BLOCKED"},
    "EXECUTED": {"VERIFIED", "REJECTED", "BLOCKED"},
    "VERIFIED": {"PROMOTED", "REJECTED", "BLOCKED"},
    "BLOCKED": {"PROPOSED"},
    "PROMOTED": set(),
    "REJECTED": set(),
}

BASE_EVIDENCE = {
    "provenance",
    "receipt_fingerprint",
    "acceptance_gates",
    "raw_samples",
}
CLASS_EVIDENCE: dict[str, set[str]] = {
    "artifact_requalification": {
        "artifact_hashes",
        "dataset_hashes",
        "scorer_hashes",
        "independent_evaluation",
    },
    "model_training": {
        "model_hash",
        "dataset_hashes",
        "seed",
        "training_trace",
        "checkpoint_hashes",
        "independent_evaluation",
    },
    "distillation": {
        "model_hash",
        "teacher_samples",
        "student_samples",
        "dataset_hashes",
        "actual_scores",
        "independent_evaluation",
    },
    "serving_runtime": {
        "effective_route",
        "service_identity",
        "paired_baseline",
        "recovery_state",
        "hardware_metrics",
    },
    "kernel_hardware": {
        "compiled_artifact_hash",
        "semantic_parity",
        "paired_baseline",
        "hardware_metrics",
    },
    "proxy_realization": {
        "real_implementation",
        "semantic_parity",
        "paired_baseline",
        "hardware_metrics",
    },
    "packed_artifact": {
        "packed_artifact_hash",
        "realized_bytes",
        "measured_vram",
        "throughput",
        "quality",
    },
    "mechanism_research": {
        "falsifiable_hypothesis",
        "invariant_controls",
        "failure_reproduction",
        "invalidation_rules",
        "semantic_parity",
        "independent_evaluation",
    },
    "model_requalification": {
        "artifact_hashes",
        "license_identity",
        "runtime_identity",
        "quality_panel",
        "role_gates",
        "independent_evaluation",
    },
    "provenance_reconciliation": {
        "artifact_hashes",
        "publisher_receipts",
        "lineage_resolution",
        "independent_evaluation",
    },
    "human_calibration": {
        "blind_label_packet",
        "rater_provenance",
        "inter_rater_agreement",
        "scorer_hashes",
        "independent_evaluation",
    },
    "kv_codec": {
        "packed_artifact_hash",
        "route_receipts",
        "full_distribution_scores",
        "retrieval_scores",
        "task_scores",
        "hardware_metrics",
    },
    "external_reproduction": {
        "source_revision",
        "dependency_hashes",
        "build_receipts",
        "correctness_receipts",
        "end_to_end_artifact",
        "independent_evaluation",
    },
}

PREREGISTRATION_HEADINGS = {
    "## Hypothesis",
    "## Frozen inputs",
    "## Command",
    "## Factors",
    "## Acceptance gates",
    "## Abort conditions",
    "## Allowed claims",
}
PLACEHOLDER_PATTERNS = ("TODO", "TBD", "[FILL", "<FILL")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _write_json_atomic(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        temporary = pathlib.Path(handle.name)
    temporary.replace(path)


def resolve_repo_path(root: pathlib.Path, relative: str, *, must_exist: bool = False) -> pathlib.Path:
    raw = pathlib.Path(relative)
    if raw.is_absolute():
        raise ValueError(f"absolute path is forbidden: {relative}")
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes repository: {relative}") from exc
    if must_exist and not resolved.exists():
        raise ValueError(f"missing path: {relative}")
    return resolved


def _external_source_receipt_errors(
    root: pathlib.Path, source: str, receipt: Any, *, prefix: str
) -> list[str]:
    """Validate a portable receipt for a source artifact intentionally kept out of Git."""
    if not isinstance(receipt, dict):
        return [f"{prefix}: external source receipt for {source} must be an object"]
    expected_sha = receipt.get("sha256")
    expected_bytes = receipt.get("bytes")
    evidence_relative = receipt.get("evidence")
    errors: list[str] = []
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        errors.append(f"{prefix}: external source receipt for {source} has invalid sha256")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        errors.append(f"{prefix}: external source receipt for {source} has invalid bytes")
    if not isinstance(evidence_relative, str) or not evidence_relative:
        errors.append(f"{prefix}: external source receipt for {source} has invalid evidence path")
        return errors
    try:
        evidence_path = resolve_repo_path(root, evidence_relative, must_exist=True)
        evidence = load_json(evidence_path)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        errors.append(f"{prefix}: external source evidence for {source} is invalid: {exc}")
        return errors
    record = evidence.get(source)
    if not isinstance(record, dict):
        errors.append(f"{prefix}: external source evidence does not contain {source}")
    elif record.get("sha256") != expected_sha or record.get("bytes") != expected_bytes:
        errors.append(f"{prefix}: external source evidence disagrees for {source}")
    return errors


def required_evidence(evidence_class: str) -> set[str]:
    return BASE_EVIDENCE | CLASS_EVIDENCE.get(evidence_class, set())


def _validate_history(history: Any, expected_state: str, prefix: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(history, list) or not history:
        return [f"{prefix}: state_history must be a non-empty list"]
    previous: str | None = None
    for index, event in enumerate(history):
        if not isinstance(event, dict):
            errors.append(f"{prefix}: history[{index}] must be an object")
            continue
        source = event.get("from")
        target = event.get("to")
        if source != previous:
            errors.append(f"{prefix}: history[{index}] from={source!r}, expected {previous!r}")
        if target not in LEGAL_TRANSITIONS.get(source, set()):
            errors.append(f"{prefix}: illegal transition {source!r} -> {target!r}")
        if not event.get("actor") or not event.get("at"):
            errors.append(f"{prefix}: history[{index}] requires actor and at")
        previous = target
    if previous != expected_state:
        errors.append(f"{prefix}: history ends at {previous!r}, state is {expected_state!r}")
    return errors


def _validate_priority_history(history: Any, prefix: str) -> list[str]:
    if history is None:
        return []
    if not isinstance(history, list):
        return [f"{prefix}: priority_history must be a list"]
    errors: list[str] = []
    for index, event in enumerate(history):
        label = f"{prefix}: priority_history[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{label} must be an object")
            continue
        if any(
            isinstance(event.get(field), bool) or event.get(field) not in range(4)
            for field in ("from", "to")
        ):
            errors.append(f"{label} requires from/to priorities from 0 to 3")
        score = event.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 100:
            errors.append(f"{label} requires score from 0 to 100")
        if not all(event.get(field) for field in ("actor", "at", "reason", "change_trigger")):
            errors.append(f"{label} requires actor, at, reason and change_trigger")
        if not re.fullmatch(r"[0-9a-f]{64}", str(event.get("policy_sha256", ""))):
            errors.append(f"{label} requires a policy SHA-256")
    return errors


def _find_cycles(items_by_id: dict[str, dict]) -> list[str]:
    errors: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(item_id: str, chain: list[str]) -> None:
        if item_id in visiting:
            start = chain.index(item_id) if item_id in chain else 0
            errors.append("dependency cycle: " + " -> ".join([*chain[start:], item_id]))
            return
        if item_id in visited:
            return
        visiting.add(item_id)
        for dependency in items_by_id[item_id].get("depends_on", []):
            if dependency in items_by_id:
                visit(dependency, [*chain, item_id])
        visiting.remove(item_id)
        visited.add(item_id)

    for item_id in items_by_id:
        visit(item_id, [])
    return errors


def validate_manifest(manifest: dict, root: pathlib.Path = ROOT) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append(f"manifest: schema must be {MANIFEST_SCHEMA}")
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        return [*errors, "manifest: items must be a non-empty list"]

    items_by_id: dict[str, dict] = {}
    for index, item in enumerate(items):
        prefix = f"items[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str) or not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]+", item_id):
            errors.append(f"{prefix}: invalid id {item_id!r}")
            continue
        if item_id in items_by_id:
            errors.append(f"{prefix}: duplicate id {item_id}")
        items_by_id[item_id] = item
        if item.get("state") not in STATES:
            errors.append(f"{item_id}: invalid state {item.get('state')!r}")
        if not isinstance(item.get("priority"), int) or item["priority"] not in range(4):
            errors.append(f"{item_id}: priority must be an integer from 0 to 3")
        priority_score = item.get("priority_score")
        if priority_score is not None and (
            isinstance(priority_score, bool)
            or not isinstance(priority_score, (int, float))
            or not 0 <= priority_score <= 100
        ):
            errors.append(f"{item_id}: priority_score must be numeric from 0 to 100")
        errors.extend(_validate_priority_history(item.get("priority_history"), item_id))
        evidence_class = item.get("evidence_class")
        if evidence_class not in CLASS_EVIDENCE:
            errors.append(f"{item_id}: unknown evidence_class {evidence_class!r}")
        declared = item.get("required_evidence")
        if not isinstance(declared, list) or not all(isinstance(code, str) and code for code in declared):
            errors.append(f"{item_id}: required_evidence must be a list")
        elif evidence_class in CLASS_EVIDENCE:
            if len(declared) != len(set(declared)):
                errors.append(f"{item_id}: required_evidence contains duplicates")
            missing = sorted(required_evidence(evidence_class) - set(declared))
            if missing:
                errors.append(f"{item_id}: missing required evidence declarations: {', '.join(missing)}")
        gates = item.get("acceptance_gates")
        if not isinstance(gates, list) or not gates:
            errors.append(f"{item_id}: acceptance_gates must be non-empty")
        else:
            gate_ids: set[str] = set()
            for gate_index, gate in enumerate(gates):
                if not isinstance(gate, dict):
                    errors.append(f"{item_id}: gate[{gate_index}] must be an object")
                    continue
                gate_id = gate.get("id")
                if not isinstance(gate_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]+", gate_id):
                    errors.append(f"{item_id}: gate[{gate_index}] has invalid id")
                    continue
                if gate_id in gate_ids:
                    errors.append(f"{item_id}: gate[{gate_index}] has missing/duplicate id")
                gate_ids.add(gate_id)
                for field in ("metric", "operator", "threshold"):
                    if field not in gate:
                        errors.append(f"{item_id}: gate {gate_id!r} missing {field}")
                if not isinstance(gate.get("metric"), str) or not gate.get("metric"):
                    errors.append(f"{item_id}: gate {gate_id!r} has invalid metric")
                if gate.get("operator") not in GATE_OPERATORS:
                    errors.append(f"{item_id}: gate {gate_id!r} has unsupported operator")
                if gate.get("operator") in {"ge", "gt", "le", "lt"} and (
                    isinstance(gate.get("threshold"), bool)
                    or not isinstance(gate.get("threshold"), (int, float))
                ):
                    errors.append(f"{item_id}: gate {gate_id!r} requires a numeric threshold")
        if not item.get("allowed_claim_codes"):
            errors.append(f"{item_id}: allowed_claim_codes must be non-empty")
        if not item.get("forbidden_claims"):
            errors.append(f"{item_id}: forbidden_claims must be non-empty")
        if not item.get("next_action"):
            errors.append(f"{item_id}: next_action is required")
        if item.get("state") == "BLOCKED" and not item.get("blocker"):
            errors.append(f"{item_id}: BLOCKED item requires blocker")
        if item.get("state") == "PROMOTED" and evidence_class not in PROMOTABLE_CLASSES:
            errors.append(f"{item_id}: evidence class {evidence_class} cannot be promoted")
        errors.extend(_validate_history(item.get("state_history"), item.get("state"), item_id))

        try:
            packet_dir = resolve_repo_path(root, item.get("packet_dir", ""))
            packet_dir.relative_to((root / "runs" / "research").resolve())
        except (ValueError, TypeError):
            errors.append(f"{item_id}: packet_dir must stay under runs/research")
        external_receipts = item.get("external_source_receipts", {})
        if not isinstance(external_receipts, dict):
            errors.append(f"{item_id}: external_source_receipts must be an object")
            external_receipts = {}
        source_artifacts = item.get("source_artifacts", [])
        for external_source in external_receipts:
            if external_source not in source_artifacts:
                errors.append(f"{item_id}: external receipt source is not declared: {external_source}")
        for source in source_artifacts:
            try:
                source_path = resolve_repo_path(root, source)
                receipt = external_receipts.get(source)
                if not source_path.exists() and receipt is None:
                    raise ValueError(f"missing path: {source}")
                if receipt is not None:
                    errors.extend(
                        _external_source_receipt_errors(
                            root, source, receipt, prefix=item_id
                        )
                    )
            except ValueError as exc:
                errors.append(f"{item_id}: {exc}")

    for item_id, item in items_by_id.items():
        dependencies = item.get("depends_on")
        if not isinstance(dependencies, list):
            errors.append(f"{item_id}: depends_on must be a list")
            continue
        for dependency in dependencies:
            if dependency == item_id:
                errors.append(f"{item_id}: cannot depend on itself")
            elif dependency not in items_by_id:
                errors.append(f"{item_id}: unknown dependency {dependency}")
    errors.extend(_find_cycles(items_by_id))
    return errors


def item_by_id(manifest: dict, item_id: str) -> dict:
    for item in manifest.get("items", []):
        if item.get("id") == item_id:
            return item
    raise KeyError(f"unknown backlog item: {item_id}")


def externalize_source(
    manifest_path: pathlib.Path,
    item_id: str,
    source: str,
    evidence: str,
    actor: str,
    *,
    root: pathlib.Path = ROOT,
) -> dict:
    """Bind a large local source to committed hash evidence without rewriting raw data."""
    manifest = load_json(manifest_path)
    errors = validate_manifest(manifest, root)
    if errors:
        raise ValueError("manifest is invalid:\n" + "\n".join(errors))
    task = item_by_id(manifest, item_id)
    if source not in task.get("source_artifacts", []):
        raise ValueError(f"{item_id}: source artifact is not declared: {source}")
    source_path = resolve_repo_path(root, source, must_exist=True)
    if not source_path.is_file():
        raise ValueError(f"{item_id}: source artifact is not a file: {source}")
    evidence_path = resolve_repo_path(root, evidence, must_exist=True)
    evidence_payload = load_json(evidence_path)
    record = evidence_payload.get(source)
    actual = {"sha256": sha256_file(source_path), "bytes": source_path.stat().st_size}
    if not isinstance(record, dict) or record.get("sha256") != actual["sha256"] or record.get("bytes") != actual["bytes"]:
        raise ValueError(f"{item_id}: evidence does not match source artifact: {source}")

    candidate = copy.deepcopy(manifest)
    candidate_task = item_by_id(candidate, item_id)
    candidate_task.setdefault("external_source_receipts", {})[source] = {
        **actual,
        "evidence": pathlib.Path(evidence).as_posix(),
    }
    event = {"actor": actor, "at": utc_now(), "source": source, "evidence": pathlib.Path(evidence).as_posix()}
    candidate_task.setdefault("portability_history", []).append(event)
    candidate["updated_at"] = event["at"]
    candidate_errors = validate_manifest(candidate, root)
    if candidate_errors:
        raise ValueError("externalization blocked:\n" + "\n".join(candidate_errors))
    _write_json_atomic(manifest_path, candidate)
    return candidate_task["external_source_receipts"][source]


def dependencies_satisfied(item: dict, manifest: dict) -> tuple[bool, list[str]]:
    items = {candidate["id"]: candidate for candidate in manifest["items"]}
    blockers = [
        dependency
        for dependency in item.get("depends_on", [])
        if items[dependency].get("state") not in DEPENDENCY_SUCCESS_STATES
    ]
    return not blockers, blockers


def select_next(manifest: dict) -> dict | None:
    candidates = []
    for item in manifest.get("items", []):
        ready, _ = dependencies_satisfied(item, manifest)
        if item.get("state") == "PROPOSED" and ready:
            candidates.append(item)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item["priority"],
            -float(item.get("priority_score", 0)),
            item["id"],
        ),
    )


def validate_priority_policy(policy: dict, manifest: dict) -> list[str]:
    """Validate the small, explicit portfolio policy used by rank/rebalance."""
    errors: list[str] = []
    if policy.get("schema") != PRIORITY_POLICY_SCHEMA:
        errors.append(f"priority policy: schema must be {PRIORITY_POLICY_SCHEMA}")
    if not policy.get("updated_at"):
        errors.append("priority policy: updated_at is required")
    dimensions = policy.get("dimensions")
    if not isinstance(dimensions, dict) or not dimensions:
        errors.append("priority policy: dimensions must be a non-empty object")
        dimensions = {}
    weights: list[float] = []
    for name, definition in dimensions.items():
        if not isinstance(name, str) or not name:
            errors.append("priority policy: dimension names must be non-empty strings")
            continue
        if not isinstance(definition, dict):
            errors.append(f"priority policy: dimension {name} must be an object")
            continue
        weight = definition.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
            errors.append(f"priority policy: dimension {name} requires a positive weight")
        else:
            weights.append(float(weight))
        if not definition.get("description"):
            errors.append(f"priority policy: dimension {name} requires a description")
    if weights and abs(sum(weights) - 100.0) > 1e-9:
        errors.append("priority policy: dimension weights must sum to 100")

    bands = policy.get("priority_bands")
    if not isinstance(bands, list) or len(bands) != 4:
        errors.append("priority policy: priority_bands must contain P0 through P3")
        bands = []
    seen_priorities: set[int] = set()
    previous_minimum = float("inf")
    for index, band in enumerate(bands):
        if not isinstance(band, dict):
            errors.append(f"priority policy: band[{index}] must be an object")
            continue
        priority = band.get("priority")
        minimum = band.get("min_score")
        if isinstance(priority, bool) or priority not in range(4) or priority in seen_priorities:
            errors.append(f"priority policy: band[{index}] has invalid/duplicate priority")
        else:
            seen_priorities.add(priority)
            if priority != index:
                errors.append("priority policy: priority bands must be ordered P0 through P3")
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not 0 <= minimum <= 100:
            errors.append(f"priority policy: band[{index}] requires min_score from 0 to 100")
        elif minimum >= previous_minimum:
            errors.append("priority policy: priority band thresholds must strictly descend")
        else:
            previous_minimum = float(minimum)
    if bands and seen_priorities != set(range(4)):
        errors.append("priority policy: priority_bands must define each priority exactly once")
    if bands and isinstance(bands[-1], dict) and bands[-1].get("min_score") != 0:
        errors.append("priority policy: P3 band must start at score 0")

    aging = policy.get("aging")
    if not isinstance(aging, dict):
        errors.append("priority policy: aging must be an object")
    else:
        for field in ("after_days", "period_days", "points_per_period", "max_bonus"):
            value = aging.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                errors.append(f"priority policy: aging {field} must be non-negative")
        if aging.get("period_days") == 0:
            errors.append("priority policy: aging period_days must be positive")

    item_ids = {item.get("id") for item in manifest.get("items", []) if isinstance(item, dict)}
    assessments = policy.get("assessments")
    if not isinstance(assessments, dict):
        errors.append("priority policy: assessments must be an object")
        assessments = {}
    dimension_names = set(dimensions)
    for item_id, assessment in assessments.items():
        label = f"priority policy: assessment {item_id}"
        if item_id not in item_ids:
            errors.append(f"{label} references an unknown backlog item")
        if not isinstance(assessment, dict):
            errors.append(f"{label} must be an object")
            continue
        scores = assessment.get("scores")
        if not isinstance(scores, dict) or set(scores) != dimension_names:
            errors.append(f"{label} scores must match the configured dimensions exactly")
        else:
            for name, score in scores.items():
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 5:
                    errors.append(f"{label} score {name} must be numeric from 0 to 5")
        for field in ("reason", "actor", "reviewed_at", "change_trigger"):
            if not assessment.get(field):
                errors.append(f"{label} requires {field}")
    return errors


def _priority_from_score(score: float, policy: dict) -> int:
    for band in policy["priority_bands"]:
        if score >= float(band["min_score"]):
            return int(band["priority"])
    raise ValueError("priority policy has no matching score band")


def _aging_bonus(item: dict, policy: dict, generated_at: str) -> tuple[int, float]:
    history = item.get("state_history") or []
    if not history or not isinstance(history[-1], dict) or not history[-1].get("at"):
        return 0, 0.0
    try:
        start = dt.datetime.fromisoformat(str(history[-1]["at"]).replace("Z", "+00:00"))
        end = dt.datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        return 0, 0.0
    waiting_days = max(0, (end - start).days)
    aging = policy["aging"]
    after_days = float(aging["after_days"])
    if waiting_days < after_days:
        return waiting_days, 0.0
    periods = 1 + int((waiting_days - after_days) // float(aging["period_days"]))
    bonus = min(float(aging["max_bonus"]), periods * float(aging["points_per_period"]))
    return waiting_days, round(bonus, 2)


def rank_backlog(
    manifest: dict,
    policy: dict,
    *,
    include_terminal: bool = False,
    generated_at: str | None = None,
) -> dict:
    """Return an explained ranking without mutating backlog state or priority."""
    errors = validate_priority_policy(policy, manifest)
    if errors:
        raise ValueError("priority policy is invalid:\n" + "\n".join(errors))
    timestamp = generated_at or utc_now()
    policy_digest = canonical_json_sha256(policy)
    weights = {name: float(value["weight"]) for name, value in policy["dimensions"].items()}
    assessments = policy["assessments"]
    rows: list[dict[str, Any]] = []
    for item in manifest["items"]:
        if not include_terminal and item["state"] in TERMINAL_STATES:
            continue
        assessment = assessments.get(item["id"])
        if assessment is None:
            base_score = None
            score = None
            waiting_days, _ = _aging_bonus(item, policy, timestamp)
            aging_bonus = 0.0
            source = "current_priority"
            reason = "No new assessment; preserve the current priority."
            recommended = item["priority"]
        else:
            base_score = sum(weights[name] * float(assessment["scores"][name]) / 5.0 for name in weights)
            base_score = round(base_score, 2)
            waiting_days, aging_bonus = _aging_bonus(item, policy, timestamp)
            score = round(min(100.0, base_score + aging_bonus), 2)
            source = "explicit_assessment"
            reason = assessment["reason"]
            recommended = _priority_from_score(score, policy)
        ready, blockers = dependencies_satisfied(item, manifest)
        priority_history = item.get("priority_history") or []
        last_policy_digest = (
            priority_history[-1].get("policy_sha256")
            if priority_history and isinstance(priority_history[-1], dict)
            else None
        )
        rows.append({
            "id": item["id"],
            "title": item["title"],
            "state": item["state"],
            "current_priority": item["priority"],
            "recommended_priority": recommended,
            "score": score,
            "base_score": base_score,
            "aging_bonus": aging_bonus,
            "waiting_days": waiting_days,
            "score_source": source,
            "eligible_now": item["state"] == "PROPOSED" and ready,
            "dependency_blockers": blockers,
            "reason": reason,
            "changed": assessment is not None and (
                item["priority"] != recommended
                or item.get("priority_score") != score
                or last_policy_digest != policy_digest
            ),
        })
    rows.sort(
        key=lambda row: (
            row["score_source"] != "explicit_assessment",
            -float(row["score"] or 0),
            row["current_priority"],
            row["id"],
        )
    )
    for index, row in enumerate(rows, start=1):
        row["portfolio_rank"] = index
    return {
        "schema": PRIORITY_REPORT_SCHEMA,
        "generated_at": timestamp,
        "manifest_updated_at": manifest.get("updated_at"),
        "policy_sha256": policy_digest,
        "assessed_count": sum(row["score_source"] == "explicit_assessment" for row in rows),
        "change_count": sum(row["changed"] for row in rows),
        "items": rows,
    }


def rebalance_priorities(
    manifest_path: pathlib.Path,
    policy_path: pathlib.Path,
    actor: str,
    *,
    apply: bool = False,
    root: pathlib.Path = ROOT,
    at: str | None = None,
) -> dict:
    """Dry-run or atomically apply explicit priority assessments only."""
    manifest = load_json(manifest_path)
    manifest_errors = validate_manifest(manifest, root)
    if manifest_errors:
        raise ValueError("manifest is invalid:\n" + "\n".join(manifest_errors))
    policy = load_json(policy_path)
    timestamp = at or utc_now()
    report = rank_backlog(manifest, policy, generated_at=timestamp)
    report["mode"] = "apply" if apply else "dry_run"
    if not apply:
        return report
    if not actor.strip():
        raise ValueError("--actor is required with --apply")
    candidate = copy.deepcopy(manifest)
    rows = {row["id"]: row for row in report["items"]}
    changed_ids: list[str] = []
    for item_id, assessment in policy["assessments"].items():
        item = item_by_id(candidate, item_id)
        row = rows.get(item_id)
        if row is None or not row["changed"]:
            continue
        previous = item["priority"]
        item["priority"] = row["recommended_priority"]
        item["priority_score"] = row["score"]
        item.setdefault("priority_history", []).append({
            "from": previous,
            "to": row["recommended_priority"],
            "score": row["score"],
            "actor": actor,
            "at": timestamp,
            "reason": assessment["reason"],
            "change_trigger": assessment["change_trigger"],
            "policy_sha256": report["policy_sha256"],
        })
        changed_ids.append(item_id)
    report["applied_ids"] = changed_ids
    if not changed_ids:
        return report
    candidate["updated_at"] = timestamp
    candidate_errors = validate_manifest(candidate, root)
    if candidate_errors:
        raise ValueError("rebalance blocked:\n" + "\n".join(candidate_errors))
    _write_json_atomic(manifest_path, candidate)
    return report


def _validate_preregistration(path: pathlib.Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing preregistration: {path}"]
    text = path.read_text(encoding="utf-8")
    missing = sorted(heading for heading in PREREGISTRATION_HEADINGS if heading not in text)
    if missing:
        errors.append("preregistration missing headings: " + ", ".join(missing))
    upper = text.upper()
    placeholders = [pattern for pattern in PLACEHOLDER_PATTERNS if pattern in upper]
    if placeholders:
        errors.append("preregistration contains placeholders: " + ", ".join(placeholders))
    if "```" not in text:
        errors.append("preregistration must freeze an executable command in a code fence")
    return errors


def implementation_digest(entries: list[dict]) -> str:
    normalized = sorted(
        ({"path": entry.get("path"), "sha256": entry.get("sha256")} for entry in entries),
        key=lambda entry: str(entry["path"]),
    )
    return canonical_json_sha256(normalized)


def _verify_recorded_file(root: pathlib.Path, record: Any, label: str) -> tuple[pathlib.Path | None, list[str]]:
    errors: list[str] = []
    if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
        return None, [f"packet: missing {label} path/SHA-256"]
    try:
        path = resolve_repo_path(root, record["path"], must_exist=True)
    except ValueError as exc:
        return None, [f"packet: {label}: {exc}"]
    if not path.is_file():
        errors.append(f"packet: {label} is not a file: {record['path']}")
    else:
        actual = sha256_file(path)
        if actual != record["sha256"]:
            errors.append(f"packet: {label} SHA-256 mismatch for {record['path']}")
    return path, errors


def _receipt_gates(receipt: dict) -> dict | None:
    gates = receipt.get("gates")
    if isinstance(gates, dict):
        return gates
    summary = receipt.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("gates"), dict):
        return summary["gates"]
    return None


def evaluate_gate(actual: Any, operator: str, threshold: Any) -> bool:
    if operator == "eq":
        return actual == threshold
    if operator == "ne":
        return actual != threshold
    if operator in {"ge", "gt", "le", "lt"}:
        if isinstance(actual, bool) or isinstance(threshold, bool):
            raise TypeError(f"operator {operator} requires numeric values")
        if not isinstance(actual, (int, float)) or not isinstance(threshold, (int, float)):
            raise TypeError(f"operator {operator} requires numeric values")
        if operator == "ge":
            return actual >= threshold
        if operator == "gt":
            return actual > threshold
        if operator == "le":
            return actual <= threshold
        return actual < threshold
    raise ValueError(f"unsupported gate operator: {operator}")


def _validate_receipt(receipt: dict, task: dict, *, require_passing_gates: bool) -> list[str]:
    errors: list[str] = []
    if receipt.get("schema") != RECEIPT_SCHEMA:
        errors.append(f"receipt: schema must be {RECEIPT_SCHEMA}")
    if receipt.get("task_id") != task.get("id"):
        errors.append("receipt: task_id mismatch")
    provenance = receipt.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("receipt: missing provenance object")
    else:
        if provenance.get("schema") != "local-labs-experiment-provenance-v1":
            errors.append("receipt: invalid provenance schema")
        if not provenance.get("command"):
            errors.append("receipt: missing command")
        if len(provenance.get("repository", {}).get("head", "")) != 40:
            errors.append("receipt: missing repository HEAD")
        if len(provenance.get("script", {}).get("sha256", "")) != 64:
            errors.append("receipt: missing script SHA-256")
        for input_record in provenance.get("inputs", []):
            if input_record.get("status") != "HASHED" or len(input_record.get("sha256", "")) != 64:
                errors.append(f"receipt: unbound input {input_record.get('path')}")
        environment = provenance.get("environment", {})
        if not environment.get("python") or not environment.get("platform"):
            errors.append("receipt: incomplete environment identity")
    if receipt.get("provenance_complete") is not True:
        errors.append("receipt: provenance_complete must be true")

    fingerprint = receipt.get("receipt_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        errors.append("receipt: missing receipt_fingerprint")
    else:
        content = copy.deepcopy(receipt)
        content.pop("receipt_fingerprint", None)
        if canonical_json_sha256(content) != fingerprint:
            errors.append("receipt: fingerprint mismatch")

    gates = _receipt_gates(receipt)
    if not isinstance(gates, dict) or not gates:
        errors.append("receipt: missing acceptance gates")
    else:
        declared = {gate["id"]: gate for gate in task["acceptance_gates"]}
        missing_gates = sorted(set(declared) - set(gates))
        unknown_gates = sorted(set(gates) - set(declared))
        if missing_gates:
            errors.append("receipt: missing declared gates: " + ", ".join(missing_gates))
        if unknown_gates:
            errors.append("receipt: unknown gates: " + ", ".join(unknown_gates))
        for gate_id in sorted(set(declared) & set(gates)):
            expected = declared[gate_id]
            observed = gates[gate_id]
            if not isinstance(observed, dict):
                errors.append(f"receipt: gate {gate_id} must record metric/operator/threshold/actual/pass")
                continue
            for field in ("metric", "operator", "threshold"):
                if observed.get(field) != expected.get(field):
                    errors.append(f"receipt: gate {gate_id} changed frozen {field}")
            if "actual" not in observed or "pass" not in observed:
                errors.append(f"receipt: gate {gate_id} requires actual and pass")
                continue
            try:
                recomputed = evaluate_gate(observed["actual"], expected["operator"], expected["threshold"])
            except (TypeError, ValueError) as exc:
                errors.append(f"receipt: gate {gate_id}: {exc}")
                continue
            if observed["pass"] is not recomputed:
                errors.append(f"receipt: gate {gate_id} pass does not match recomputed result")
            if require_passing_gates and not recomputed:
                errors.append(f"receipt: non-passing gate: {gate_id}")

    evidence = receipt.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("receipt: missing machine-readable evidence map")
    else:
        missing = sorted(code for code in task["required_evidence"] if not evidence.get(code))
        if missing:
            errors.append("receipt: missing evidence: " + ", ".join(missing))
    return errors


def validate_packet(
    packet_dir: pathlib.Path,
    task: dict,
    root: pathlib.Path = ROOT,
    *,
    packet_data: dict | None = None,
) -> list[str]:
    errors: list[str] = []
    pipeline_path = packet_dir / "PIPELINE.json"
    if packet_data is None and not pipeline_path.is_file():
        return [f"{task['id']}: missing {pipeline_path.relative_to(root)}"]
    try:
        packet = packet_data if packet_data is not None else load_json(pipeline_path)
    except Exception as exc:
        return [f"{task['id']}: invalid PIPELINE.json: {exc}"]

    prefix = task["id"]
    if packet.get("schema") != PACKET_SCHEMA:
        errors.append(f"{prefix}: packet schema must be {PACKET_SCHEMA}")
    if packet.get("task_id") != task["id"]:
        errors.append(f"{prefix}: packet task_id mismatch")
    if packet.get("evidence_class") != task["evidence_class"]:
        errors.append(f"{prefix}: packet evidence_class mismatch")
    stage = packet.get("stage")
    if stage != task.get("state"):
        errors.append(f"{prefix}: packet stage {stage!r} != manifest state {task.get('state')!r}")
    if stage not in STATES:
        errors.append(f"{prefix}: invalid packet stage {stage!r}")
        return errors
    errors.extend(_validate_history(packet.get("state_history"), stage, f"{prefix} packet"))
    if packet.get("state_history") != task.get("state_history"):
        errors.append(f"{prefix}: packet and manifest histories differ")
    executor = str(packet.get("executor", ""))
    for event in packet.get("state_history", []):
        actor = str(event.get("actor", "")) if isinstance(event, dict) else ""
        if isinstance(event, dict) and event.get("to") in {"VERIFIED", "PROMOTED"}:
            if not actor or actor.casefold() == executor.casefold() or "gemini" in actor.casefold():
                errors.append(f"{prefix}: {event.get('to')} transition requires an independent actor")

    preregistered_states = STATES - {"PROPOSED", "BLOCKED"}
    implemented_states = {"IMPLEMENTED", "EXECUTED", "VERIFIED", "PROMOTED", "REJECTED"}
    executed_states = {"EXECUTED", "VERIFIED", "PROMOTED", "REJECTED"}
    result_states = {"VERIFIED", "PROMOTED", "REJECTED"}

    if stage in preregistered_states:
        prereg_path, prereg_errors = _verify_recorded_file(root, packet.get("preregistration"), "preregistration")
        errors.extend(prereg_errors)
        if prereg_path:
            errors.extend(f"{prefix}: {error}" for error in _validate_preregistration(prereg_path))

    if stage in implemented_states:
        implementation = packet.get("implementation")
        if not isinstance(implementation, list) or not implementation:
            errors.append(f"{prefix}: implementation file list is empty")
        else:
            for index, record in enumerate(implementation):
                _, record_errors = _verify_recorded_file(root, record, f"implementation[{index}]")
                errors.extend(f"{prefix}: {error}" for error in record_errors)
            if packet.get("implementation_digest") != implementation_digest(implementation):
                errors.append(f"{prefix}: implementation_digest mismatch")

    receipt: dict | None = None
    receipt_path: pathlib.Path | None = None
    if stage in executed_states:
        receipt_path, receipt_errors = _verify_recorded_file(root, packet.get("execution"), "execution receipt")
        errors.extend(f"{prefix}: {error}" for error in receipt_errors)
        if receipt_path:
            try:
                receipt = load_json(receipt_path)
                errors.extend(
                    f"{prefix}: {error}"
                    for error in _validate_receipt(
                        receipt,
                        task,
                        require_passing_gates=stage in {"VERIFIED", "PROMOTED"},
                    )
                )
            except Exception as exc:
                errors.append(f"{prefix}: invalid execution receipt: {exc}")

    if stage in result_states:
        _, result_errors = _verify_recorded_file(root, packet.get("result"), "result")
        errors.extend(f"{prefix}: {error}" for error in result_errors)

    claim_codes = packet.get("claim_codes")
    if not isinstance(claim_codes, list):
        errors.append(f"{prefix}: claim_codes must be a list")
    else:
        invalid_claims = sorted(set(claim_codes) - set(task["allowed_claim_codes"]))
        if invalid_claims:
            errors.append(f"{prefix}: unauthorized claim codes: {', '.join(invalid_claims)}")
        if stage in {"VERIFIED", "PROMOTED"} and not claim_codes:
            errors.append(f"{prefix}: verified/promoted packet requires at least one claim code")

    if stage in {"VERIFIED", "PROMOTED"}:
        review_path, review_errors = _verify_recorded_file(root, packet.get("review"), "independent review")
        errors.extend(f"{prefix}: {error}" for error in review_errors)
        if review_path:
            try:
                review = load_json(review_path)
                if review.get("schema") != REVIEW_SCHEMA:
                    errors.append(f"{prefix}: invalid review schema")
                reviewer = str(review.get("reviewer", ""))
                if not reviewer or reviewer.casefold() == executor.casefold() or "gemini" in reviewer.casefold():
                    errors.append(f"{prefix}: review must be independent of Gemini/executor")
                if review.get("verdict") != "APPROVED":
                    errors.append(f"{prefix}: independent review verdict must be APPROVED")
                if receipt_path and review.get("receipt_sha256") != sha256_file(receipt_path):
                    errors.append(f"{prefix}: review is not bound to the current receipt")
                if review.get("implementation_digest") != packet.get("implementation_digest"):
                    errors.append(f"{prefix}: review is not bound to the current implementation")
                if not isinstance(review.get("findings"), list):
                    errors.append(f"{prefix}: review findings must be a list")
            except Exception as exc:
                errors.append(f"{prefix}: invalid review: {exc}")

    if stage == "PROMOTED" and task["evidence_class"] not in PROMOTABLE_CLASSES:
        errors.append(f"{prefix}: {task['evidence_class']} evidence cannot be promoted")
    if stage == "REJECTED" and receipt is not None:
        gates = _receipt_gates(receipt) or {}
        if gates and all(isinstance(value, dict) and value.get("pass") is True for value in gates.values()):
            errors.append(f"{prefix}: REJECTED packet has no failed gate")
    return errors


def gate_repository(manifest_path: pathlib.Path = DEFAULT_MANIFEST, root: pathlib.Path = ROOT) -> list[str]:
    try:
        manifest = load_json(manifest_path)
    except Exception as exc:
        return [f"manifest load failure: {exc}"]
    errors = validate_manifest(manifest, root)
    known_packet_paths: set[pathlib.Path] = set()
    for task in manifest.get("items", []):
        try:
            packet_dir = resolve_repo_path(root, task["packet_dir"])
        except ValueError:
            continue
        pipeline_path = packet_dir / "PIPELINE.json"
        if pipeline_path.exists():
            known_packet_paths.add(pipeline_path.resolve())
            errors.extend(validate_packet(packet_dir, task, root))
        elif task.get("state") not in {"PROPOSED", "BLOCKED"}:
            errors.append(f"{task['id']}: state {task['state']} requires a packet")
    research_root = root / "runs" / "research"
    if research_root.exists():
        for pipeline_path in research_root.rglob("PIPELINE.json"):
            if pipeline_path.resolve() not in known_packet_paths:
                errors.append(f"unregistered pipeline packet: {pipeline_path.relative_to(root)}")
    return errors


def _preregistration_template(task: dict) -> str:
    gates = "\n".join(
        f"- `{gate['id']}`: `{gate['metric']} {gate['operator']} {gate['threshold']}`"
        for gate in task["acceptance_gates"]
    )
    inputs = "\n".join(f"- `{path}`" for path in task.get("source_artifacts", [])) or "- None"
    allowed = "\n".join(f"- `{code}`" for code in task["allowed_claim_codes"])
    return f"""# {task['id']} preregistration

Task: {task['title']}
Evidence class: `{task['evidence_class']}`

## Hypothesis

[FILL: one falsifiable causal hypothesis]

## Frozen inputs

{inputs}

[FILL: add exact SHA-256 identities for every input]

## Command

```powershell
[FILL: exact executable command with output under this packet]
```

## Factors

[FILL: independent variables, controls, sample count, seed, hardware/runtime]

## Acceptance gates

{gates}

## Abort conditions

[FILL: safety, fit, service, data and provenance abort conditions]

## Allowed claims

{allowed}

Claims outside these codes are forbidden even if a metric looks favorable.
"""


def _receipt_template(task: dict) -> dict:
    return {
        "schema": RECEIPT_SCHEMA,
        "task_id": task["id"],
        "provenance": {},
        "provenance_complete": False,
        "gates": {
            gate["id"]: {
                "metric": gate["metric"],
                "operator": gate["operator"],
                "threshold": gate["threshold"],
                "actual": None,
                "pass": None,
            }
            for gate in task["acceptance_gates"]
        },
        "evidence": {code: None for code in task["required_evidence"]},
        "receipt_fingerprint": None,
    }


def _review_template() -> dict:
    return {
        "schema": REVIEW_SCHEMA,
        "reviewer": None,
        "verdict": "PENDING",
        "receipt_sha256": None,
        "implementation_digest": None,
        "findings": [],
    }


def admit_item(
    manifest_path: pathlib.Path,
    spec_path: pathlib.Path,
    actor: str,
    *,
    root: pathlib.Path = ROOT,
) -> str:
    """Admit one new PROPOSED item through a validated, atomic mutation."""
    if not actor.strip():
        raise ValueError("admission actor is required")

    repository_errors = gate_repository(manifest_path, root)
    if repository_errors:
        raise ValueError("repository gate is not clean:\n" + "\n".join(repository_errors))

    manifest = load_json(manifest_path)
    resolved_spec = spec_path.resolve()
    try:
        resolved_spec.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("admission spec must stay inside the repository") from exc
    spec = load_json(resolved_spec)
    if set(spec).intersection({"state", "state_history", "blocker"}):
        raise ValueError("admission spec cannot set state, state_history, or blocker")

    item_id = spec.get("id")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError("admission spec requires an id")
    if any(item.get("id") == item_id for item in manifest["items"]):
        raise ValueError(f"duplicate backlog item: {item_id}")

    packet_dir_value = spec.get("packet_dir")
    if not isinstance(packet_dir_value, str):
        raise ValueError("admission spec requires packet_dir")
    packet_dir = resolve_repo_path(root, packet_dir_value)
    if packet_dir.exists():
        raise FileExistsError(f"packet path already exists: {packet_dir}")

    admitted_at = utc_now()
    item = copy.deepcopy(spec)
    item["state"] = "PROPOSED"
    item["blocker"] = None
    item["state_history"] = [
        {"from": None, "to": "PROPOSED", "actor": actor, "at": admitted_at}
    ]

    candidate = copy.deepcopy(manifest)
    candidate["items"].append(item)
    candidate["updated_at"] = admitted_at
    errors = validate_manifest(candidate, root)
    if errors:
        raise ValueError("admission blocked:\n" + "\n".join(errors))

    _write_json_atomic(manifest_path, candidate)
    return item_id


def scaffold_packet(manifest_path: pathlib.Path, item_id: str, actor: str, root: pathlib.Path = ROOT) -> pathlib.Path:
    manifest = load_json(manifest_path)
    errors = validate_manifest(manifest, root)
    if errors:
        raise ValueError("manifest is invalid:\n" + "\n".join(errors))
    task = item_by_id(manifest, item_id)
    if task["state"] != "PROPOSED":
        raise ValueError(f"{item_id}: scaffold requires PROPOSED, found {task['state']}")
    ready, blockers = dependencies_satisfied(task, manifest)
    if not ready:
        raise ValueError(f"{item_id}: unsatisfied dependencies: {', '.join(blockers)}")
    packet_dir = resolve_repo_path(root, task["packet_dir"])
    if packet_dir.exists():
        raise FileExistsError(f"packet already exists: {packet_dir}")
    packet_dir.mkdir(parents=True)
    (packet_dir / "raw").mkdir()
    prereg_path = packet_dir / "PRE_REGISTRATION.md"
    prereg_path.write_text(_preregistration_template(task), encoding="utf-8", newline="\n")
    _write_json_atomic(packet_dir / "RECEIPT.template.json", _receipt_template(task))
    _write_json_atomic(packet_dir / "REVIEW.template.json", _review_template())
    packet = {
        "schema": PACKET_SCHEMA,
        "task_id": task["id"],
        "evidence_class": task["evidence_class"],
        "executor": actor,
        "stage": "PROPOSED",
        "state_history": copy.deepcopy(task["state_history"]),
        "preregistration": {},
        "implementation": [],
        "implementation_digest": None,
        "execution": {},
        "result": {},
        "review": {},
        "claim_codes": [],
    }
    _write_json_atomic(packet_dir / "PIPELINE.json", packet)
    return packet_dir


def _file_record(root: pathlib.Path, path: pathlib.Path) -> dict:
    resolved = path.resolve()
    relative = resolved.relative_to(root.resolve()).as_posix()
    return {"path": relative, "sha256": sha256_file(resolved)}


def advance_item(
    manifest_path: pathlib.Path,
    item_id: str,
    target: str,
    actor: str,
    *,
    root: pathlib.Path = ROOT,
    implementation_paths: list[str] | None = None,
    claim_codes: list[str] | None = None,
    reason: str | None = None,
) -> None:
    target = target.upper()
    manifest = load_json(manifest_path)
    errors = validate_manifest(manifest, root)
    if errors:
        raise ValueError("manifest is invalid:\n" + "\n".join(errors))
    task = item_by_id(manifest, item_id)
    source = task["state"]
    if target not in LEGAL_TRANSITIONS.get(source, set()):
        raise ValueError(f"{item_id}: illegal transition {source} -> {target}")
    if source == "PROPOSED" and target == "PREREGISTERED":
        ready, blockers = dependencies_satisfied(task, manifest)
        if not ready:
            raise ValueError(f"{item_id}: unsatisfied dependencies: {', '.join(blockers)}")

    packet_dir = resolve_repo_path(root, task["packet_dir"])
    pipeline_path = packet_dir / "PIPELINE.json"
    packet: dict | None = load_json(pipeline_path) if pipeline_path.exists() else None
    if target not in {"BLOCKED", "PROPOSED"} and packet is None:
        raise ValueError(f"{item_id}: scaffold the packet before advancing")

    event = {"from": source, "to": target, "actor": actor, "at": utc_now()}
    if reason:
        event["reason"] = reason
    candidate_manifest = copy.deepcopy(manifest)
    candidate_task = item_by_id(candidate_manifest, item_id)
    candidate_task["state"] = target
    candidate_task["state_history"].append(event)
    candidate_task["blocker"] = reason if target == "BLOCKED" else None
    candidate_manifest["updated_at"] = event["at"]

    candidate_packet: dict | None = copy.deepcopy(packet) if packet is not None else None
    if candidate_packet is not None:
        candidate_packet["stage"] = target
        candidate_packet["state_history"].append(event)
        if target == "PREREGISTERED":
            prereg = packet_dir / "PRE_REGISTRATION.md"
            prereg_errors = _validate_preregistration(prereg)
            if prereg_errors:
                raise ValueError("\n".join(prereg_errors))
            candidate_packet["preregistration"] = _file_record(root, prereg)
        if target == "IMPLEMENTED":
            if not implementation_paths:
                raise ValueError("IMPLEMENTED requires --implementation file(s)")
            records = []
            for relative in implementation_paths:
                path = resolve_repo_path(root, relative, must_exist=True)
                if not path.is_file():
                    raise ValueError(f"implementation path is not a file: {relative}")
                records.append(_file_record(root, path))
            candidate_packet["implementation"] = sorted(records, key=lambda record: record["path"])
            candidate_packet["implementation_digest"] = implementation_digest(candidate_packet["implementation"])
        if target == "EXECUTED":
            receipt = packet_dir / "raw" / "receipt.json"
            if not receipt.is_file():
                raise ValueError(f"missing execution receipt: {receipt}")
            candidate_packet["execution"] = _file_record(root, receipt)
        if target in {"VERIFIED", "REJECTED"}:
            result = packet_dir / "RESULT.md"
            if not result.is_file():
                raise ValueError(f"missing result: {result}")
            candidate_packet["result"] = _file_record(root, result)
            if claim_codes:
                candidate_packet["claim_codes"] = sorted(set(claim_codes))
            if target == "VERIFIED":
                review = packet_dir / "REVIEW.json"
                if not review.is_file():
                    raise ValueError(f"missing independent review: {review}")
                candidate_packet["review"] = _file_record(root, review)
        if target == "PROMOTED" and claim_codes:
            candidate_packet["claim_codes"] = sorted(set(claim_codes))

    candidate_errors = validate_manifest(candidate_manifest, root)
    if candidate_packet is not None:
        candidate_errors.extend(
            validate_packet(packet_dir, candidate_task, root, packet_data=candidate_packet)
        )
    if candidate_errors:
        raise ValueError("transition blocked:\n" + "\n".join(candidate_errors))

    if candidate_packet is not None:
        _write_json_atomic(pipeline_path, candidate_packet)
    _write_json_atomic(manifest_path, candidate_manifest)


def _print_errors(errors: list[str], *, json_output: bool = False) -> None:
    if json_output:
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2, ensure_ascii=False))
        return
    if not errors:
        print("BACKLOG PIPELINE: PASS")
        return
    print(f"BACKLOG PIPELINE: FAIL ({len(errors)} error(s))")
    for error in errors:
        print(f"- {error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gate_parser = subparsers.add_parser("gate", help="validate manifest and all registered packets")
    gate_parser.add_argument("--json", action="store_true", dest="json_output")

    next_parser = subparsers.add_parser("next", help="select the highest-priority ready item")
    next_parser.add_argument("--json", action="store_true", dest="json_output")

    status_parser = subparsers.add_parser("status", help="show backlog state summary")
    status_parser.add_argument("--json", action="store_true", dest="json_output")

    rank_parser = subparsers.add_parser("rank", help="explain the current portfolio ranking")
    rank_parser.add_argument("--policy", type=pathlib.Path, default=DEFAULT_PRIORITY_POLICY)
    rank_parser.add_argument("--json", action="store_true", dest="json_output")
    rank_parser.add_argument("--explain", action="store_true", help="show score source and rationale")
    rank_parser.add_argument("--include-terminal", action="store_true")

    rebalance_parser = subparsers.add_parser(
        "rebalance", help="dry-run or apply explicit policy assessments"
    )
    rebalance_parser.add_argument("--policy", type=pathlib.Path, default=DEFAULT_PRIORITY_POLICY)
    mode = rebalance_parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="preview changes (default)")
    mode.add_argument("--apply", action="store_true", help="atomically update assessed priorities")
    rebalance_parser.add_argument("--actor")
    rebalance_parser.add_argument("--json", action="store_true", dest="json_output")

    admit_parser = subparsers.add_parser("admit", help="admit one validated PROPOSED item from a JSON spec")
    admit_parser.add_argument("spec", type=pathlib.Path)
    admit_parser.add_argument("--actor", required=True)

    externalize_parser = subparsers.add_parser(
        "externalize-source",
        help="bind a declared large source artifact to committed hash evidence",
    )
    externalize_parser.add_argument("item_id")
    externalize_parser.add_argument("source")
    externalize_parser.add_argument("--evidence", required=True)
    externalize_parser.add_argument("--actor", required=True)

    scaffold_parser = subparsers.add_parser("scaffold", help="create a draft packet for a proposed item")
    scaffold_parser.add_argument("item_id")
    scaffold_parser.add_argument("--actor", required=True)

    advance_parser = subparsers.add_parser("advance", help="perform a validated state transition")
    advance_parser.add_argument("item_id")
    advance_parser.add_argument("--to", required=True, choices=sorted(STATES))
    advance_parser.add_argument("--actor", required=True)
    advance_parser.add_argument("--implementation", action="append", default=[])
    advance_parser.add_argument("--claim-code", action="append", default=[])
    advance_parser.add_argument("--reason")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    try:
        if args.command == "gate":
            errors = gate_repository(manifest_path)
            _print_errors(errors, json_output=args.json_output)
            return 1 if errors else 0
        if args.command == "admit":
            item_id = admit_item(manifest_path, args.spec, args.actor)
            print(f"{item_id}: admitted as PROPOSED")
            return 0
        if args.command == "externalize-source":
            receipt = externalize_source(
                manifest_path,
                args.item_id,
                args.source,
                args.evidence,
                args.actor,
            )
            print(json.dumps(receipt, indent=2, ensure_ascii=False))
            return 0
        if args.command == "rebalance":
            report = rebalance_priorities(
                manifest_path,
                args.policy.resolve(),
                args.actor or "",
                apply=args.apply,
            )
            if args.json_output:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                verb = "Applied" if args.apply else "Would apply"
                changes = [row for row in report["items"] if row["changed"]]
                print(f"{verb} {len(changes)} assessed priority update(s).")
                for row in changes:
                    print(
                        f"{row['id']}: P{row['current_priority']} -> "
                        f"P{row['recommended_priority']} score={row['score']:.2f}"
                    )
            return 0
        manifest = load_json(manifest_path)
        errors = validate_manifest(manifest)
        if errors:
            _print_errors(errors)
            return 1
        if args.command == "next":
            item = select_next(manifest)
            if args.json_output:
                print(json.dumps(item, indent=2, ensure_ascii=False))
            elif item is None:
                print("No dependency-ready PROPOSED item.")
            else:
                print(f"{item['id']} P{item['priority']}: {item['title']}")
                print(f"Next action: {item['next_action']}")
            return 0
        if args.command == "status":
            summary = [
                {
                    "id": item["id"],
                    "priority": item["priority"],
                    "priority_score": item.get("priority_score"),
                    "state": item["state"],
                    "title": item["title"],
                }
                for item in sorted(
                    manifest["items"],
                    key=lambda item: (
                        item["priority"],
                        -float(item.get("priority_score", 0)),
                        item["id"],
                    ),
                )
            ]
            if args.json_output:
                print(json.dumps(summary, indent=2, ensure_ascii=False))
            else:
                for item in summary:
                    print(f"P{item['priority']} {item['state']:<14} {item['id']}: {item['title']}")
            return 0
        if args.command == "rank":
            policy = load_json(args.policy.resolve())
            report = rank_backlog(
                manifest,
                policy,
                include_terminal=args.include_terminal,
            )
            if args.json_output:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                for row in report["items"]:
                    score_text = f"{row['score']:.2f}" if row["score"] is not None else "unassessed"
                    line = (
                        f"#{row['portfolio_rank']:03d} {row['id']} "
                        f"score={score_text} P{row['current_priority']}"
                        f"->P{row['recommended_priority']} {row['state']}"
                    )
                    print(line)
                    if args.explain:
                        print(f"  {row['score_source']}: {row['reason']}")
            return 0
        if args.command == "scaffold":
            packet_dir = scaffold_packet(manifest_path, args.item_id, args.actor)
            print(packet_dir)
            return 0
        if args.command == "advance":
            advance_item(
                manifest_path,
                args.item_id,
                args.to,
                args.actor,
                implementation_paths=args.implementation,
                claim_codes=args.claim_code,
                reason=args.reason,
            )
            print(f"{args.item_id}: advanced to {args.to}")
            return 0
    except Exception as exc:
        print(f"BACKLOG PIPELINE ERROR: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
