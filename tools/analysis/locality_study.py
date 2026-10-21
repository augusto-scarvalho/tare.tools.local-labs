#!/usr/bin/env python3
"""Frugal EXP-3 offline measurement; never a scheduler or a live qualification.

Usage: python tools/analysis/locality_study.py STUDY.json [--output NEW.json]
Input schema: tare.tools/locality-study/1, with 1..12 trials. Each trial has
id, case, route (local/remote), session (absolute directory), optional absolute
fleet_before/fleet_after JSON paths, cost_components, assessment and energy.
Fleet snapshots accept the fleet object itself, or {"status": <fleet object>,
"observed_at": ...}; backend_pid, backend_healthy and current_model are used.
Costs are explicit components {name, amount: decimal string|null, currency,
basis, source}; no totals or monetary winner are inferred. Native usage costs
are retained separately, including incomplete coverage. Missing is not free.
Assessment hashes bind UTF-8 output text, or compact sorted-key JSON with
ensure_ascii=False for structured output. Checks are external declarations,
not executed or independently certified here. Assessment shape is
{answer_sha256, checks: [{id, passed: boolean}]}, with nonempty unique check IDs.
Energy shape is {gpu_joules: number, attribution: "shared_gpu_gross",
source_sha256: hex digest}; this is caller-supplied sampled GPU gross only,
not a claim to have read or verified the original energy sampler's source.
Every consumed file (including the study) is hashed, bounded to 4 MiB/file
and 32 MiB total. Invocation timestamps are numeric Unix epoch seconds;
aware ISO timestamps are also accepted. Missing/malformed timestamps remain
unknown; nonfinite numbers and negative elapsed durations are rejected.
Switch duration is diagnostic only, never added to already inclusive wall time.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re

FILE_LIMIT = 4 * 1024 * 1024
TOTAL_LIMIT = 32 * 1024 * 1024
COST_GAPS = ["whole_host_energy", "electricity", "hardware_allocation",
             "cloud_quota_allocation"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")
    return value


def obj(value):
    require(isinstance(value, dict), "expected JSON object")
    return value


def array(value):
    require(isinstance(value, list), "expected JSON array")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def reject_constant(value):
    raise ValueError("nonfinite JSON number: " + value)


class Sources:
    def __init__(self):
        self.total = 0
        self.hashes = {}

    def read(self, path, jsonl=False):
        path = Path(path).resolve()
        remaining = min(FILE_LIMIT, TOTAL_LIMIT - self.total)
        require(remaining >= 0, "total byte ceiling exceeded")
        with path.open("rb") as stream:
            raw = stream.read(remaining + 1)
        require(len(raw) <= remaining, "input byte ceiling exceeded: " + str(path))
        self.total += len(raw)
        self.hashes[str(path)] = digest(raw)
        source = raw.decode("utf-8")
        if jsonl:
            return [obj(json.loads(line, parse_constant=reject_constant))
                    for line in source.splitlines() if line.strip()]
        return obj(json.loads(source, parse_constant=reject_constant))


def absolute(value):
    path = Path(text(value, "path"))
    require(path.is_absolute(), "source path must be absolute")
    return path.resolve()


def timing(value):
    if value is None:
        return None
    require(type(value) in (int, float), "invalid timing")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite, "invalid timing")
    require(value >= 0, "negative timing")
    return value


def timing_sum(values):
    return timing(sum(values)) if values and None not in values else None


def invocation_timestamp(value):
    if type(value) in (int, float):
        # Epochs may precede 1970; validate finiteness, not epoch sign.
        timing(abs(value))
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.timestamp() if parsed.tzinfo is not None else None
        except (ValueError, OverflowError, OSError):
            pass
    return None


def active_wall(invocations):
    durations = []
    for invocation in array(invocations):
        invocation = obj(invocation)
        start = invocation_timestamp(invocation.get("started_at"))
        end = invocation_timestamp(invocation.get("ended_at"))
        durations.append(None if start is None or end is None else timing(end - start))
    return timing_sum(durations)


def residency(route, model, before, after):
    result = {"state": "unknown", "reason": "incomplete or incompatible PID/model/health evidence",
              "last_switch_seconds": None}
    if route == "remote":
        result["reason"] = "remote residency is not observable"
        return result
    if before is None or after is None:
        return result
    before = obj(before.get("status")) if isinstance(before.get("status"), dict) else before
    after = obj(after.get("status")) if isinstance(after.get("status"), dict) else after
    pid = after.get("backend_pid")
    if not (type(pid) is int and pid > 0 and after.get("backend_healthy") is True
            and after.get("current_model") == model):
        return result
    before_pid = before.get("backend_pid")
    absent_pid = before_pid is None or (type(before_pid) is int and before_pid == 0)
    if (absent_pid and before.get("current_model") in (None, "")
            and before.get("backend_healthy") is False):
        result.update(state="cold_load", reason="absent backend became healthy deployment model",
                      last_switch_seconds=timing(after.get("last_switch_seconds")))
    elif (type(before.get("backend_pid")) is int and before.get("backend_pid") == pid
          and before.get("backend_healthy") is True and before.get("current_model") == model):
        result.update(state="warm", reason="same healthy PID and deployment model")
    return result


def decimal_amount(value):
    if value is None:
        return
    require(isinstance(value, str) and re.fullmatch(r"[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value),
            "amount must be a nonnegative finite decimal string or null")
    try:
        amount = Decimal(value)
        require(amount.is_finite() and amount >= 0, "invalid cost amount")
    except InvalidOperation as exc:
        raise ValueError("invalid cost amount") from exc


def costs(trial):
    components = array(trial.get("cost_components", []))
    gaps = list(COST_GAPS)
    if not components:
        gaps.append("explicit_cost_components_missing")
    for component in components:
        component = obj(component)
        for field in ("name", "currency", "basis", "source"):
            text(component.get(field), "cost " + field)
        decimal_amount(component.get("amount"))
        if component.get("amount") is None:
            gaps.append("unknown_component:" + component["name"])
    return {"components": components, "gaps": gaps, "complete": False}


def quality(trial, outcome):
    if "assessment" not in trial:
        return {"status": "unknown", "reason": "no external assessment"}
    assessment = obj(trial["assessment"])
    require("output" in outcome and outcome["output"] is not None, "assessment needs outcome output")
    output = outcome["output"]
    answer = output if isinstance(output, str) else canonical(output)
    answer_hash = digest(answer.encode("utf-8"))
    require(assessment.get("answer_sha256") == answer_hash, "answer hash mismatch")
    checks = array(assessment.get("checks"))
    require(bool(checks), "assessment needs nonempty checks")
    names = set()
    for check in checks:
        check = obj(check)
        name = text(check.get("id"), "check id")
        require(name not in names and type(check.get("passed")) is bool,
                "checks must be unique and boolean")
        names.add(name)
    return {"status": "passed" if all(c["passed"] for c in checks) else "failed",
            "answer_sha256": answer_hash, "checks": checks, "basis": "external assessment"}


def energy(trial):
    if "energy" not in trial:
        return None
    sample = obj(trial["energy"])
    require(sample.get("attribution") == "shared_gpu_gross", "energy must be shared_gpu_gross")
    require(isinstance(sample.get("source_sha256"), str) and
            re.fullmatch(r"[0-9a-f]{64}", sample["source_sha256"]), "invalid energy source hash")
    require(sample.get("gpu_joules") is not None, "energy needs gpu_joules")
    timing(sample["gpu_joules"])
    return sample


def analyze(study_path):
    sources = Sources()
    study = sources.read(study_path)
    require(study.get("schema") == "tare.tools/locality-study/1", "unsupported study schema")
    trials = array(study.get("trials"))
    require(1 <= len(trials) <= 12, "study needs 1..12 trials")
    ids, sessions, identities, rows = set(), set(), set(), []
    for trial in trials:
        trial = obj(trial)
        trial_id = text(trial.get("id"), "trial id")
        require(trial_id not in ids, "duplicate trial id")
        ids.add(trial_id)
        case = text(trial.get("case"), "case")
        route = trial.get("route")
        require(route in ("local", "remote"), "invalid route")
        session = absolute(trial.get("session"))
        require(session not in sessions, "duplicate session")
        sessions.add(session)
        native = sources.read(session / "native-agent.json")
        outcome = sources.read(session / "outcome.json")
        usage = sources.read(session / "usage.json")
        observations = sources.read(session / "observations.jsonl", jsonl=True)
        sid = text(native.get("session_id"), "session_id")
        require(sid not in identities, "duplicate session identity")
        identities.add(sid)
        require(outcome.get("session_id") == sid and usage.get("session_id") == sid,
                "session identity mismatch")
        model = text(obj(obj(native.get("deployment")).get("native_profile")).get("model"), "model")
        generation, first_content = [], []
        counts = 0
        for wrapper in observations:
            event = obj(wrapper.get("event"))
            purpose = event.get("purpose")
            if "session_id" in event or purpose in ("model", "count"):
                require(event.get("session_id") == sid, "observation session identity mismatch")
            if purpose == "count":
                counts += 1
            if purpose != "model":
                continue
            text(event.get("call_id"), "generation call_id")
            generation.append(timing(event.get("duration_ms")))
            transports = array(event.get("transport_calls", []))
            first_content.append([timing(obj(call).get("first_content_ms")) for call in transports])
        snapshots = [sources.read(absolute(trial[key])) if key in trial else None
                     for key in ("fleet_before", "fleet_after")]
        # Preserve native pricing and coverage verbatim, not an invented total.
        for call in array(usage.get("calls", [])):
            obj(call)
        rows.append({"id": trial_id, "case": case, "route": route, "session": str(session),
                     "session_id": sid, "model": model, "status": outcome.get("status"),
                     "invocations": native.get("invocations", []),
                     "active_wall_seconds": active_wall(native.get("invocations", [])),
                     "generation_count": len(generation), "count_operations": counts,
                     "generation_duration_ms": generation,
                     "observed_generation_duration_ms": timing_sum(generation),
                     "first_content_ms_by_generation_transport": first_content,
                     "residency": residency(route, model, *snapshots), "usage": usage,
                     "cost": costs(trial), "quality": quality(trial, outcome),
                     "energy": energy(trial)})
    return {"schema": "tare.tools/locality-study-report/1", "trials": rows,
            "source_sha256": sources.hashes, "input_bytes": sources.total,
            "limitations": ["offline evidence only; no live experiment qualification",
                            "GPU energy is shared gross, not whole-host or money",
                            "cost gaps remain unknown; no winner or routing recommendation"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study")
    parser.add_argument("--output", help="create-only JSON report")
    args = parser.parse_args(argv)
    try:
        report = json.dumps(analyze(args.study), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as stream:
                stream.write(report)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
