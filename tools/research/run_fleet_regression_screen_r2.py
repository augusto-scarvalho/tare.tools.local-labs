#!/usr/bin/env python3
"""Fresh R2 fleet screen with request retention and pre-receipt terminal binding."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import canonical_json_sha256, sha256_file
from tools.research import run_fleet_regression_screen as base

TASK_ID = "BACKLOG-FLEET-REGRESSION-SCREEN-02"
SOURCE_HASHES = {
    "config/research_backlog_admissions/BACKLOG-FLEET-REGRESSION-SCREEN-02.json": "04d4d41a6aca177eebd5cff44adf0892f6675b07eb665531f2da4d745fbaee03",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    "workloads/gsm8k.jsonl": "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    "runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl": "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
    "tools/benchmarks/agent_suite_v2.py": "14d0a1b76d4d729228678f215ecefa3254aef214eb65ac9d8d7061bccc0dc59e",
    "tools/research/run_fleet_regression_screen.py": "7cbf942375c120970ac78fb40f7f15050ebddb7ec5372cb40bd721009faf1de3",
    "runs/research/BACKLOG-FLEET-REGRESSION-SCREEN-01/raw/receipt.json": "d303490b152babcc8f590b0b840fb14c4da02c865430f8398bdca7023a0eeb94",
    "docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md": "a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04",
}

_original_build = base.build_provenance
_original_write = base.write_json
_active_outdir: pathlib.Path | None = None


def _gate(actual: Any, threshold: Any, operator: str = "eq") -> dict[str, Any]:
    passed = actual == threshold if operator == "eq" else actual >= threshold
    return {"operator": operator, "threshold": threshold, "actual": actual, "pass": passed}


def _bound_build(*args: Any, **kwargs: Any) -> dict[str, Any]:
    assert _active_outdir is not None
    raw = _active_outdir / "raw"
    rows = base.read_jsonl(raw / "samples.jsonl")
    metrics = json.loads((raw / "actual_scores.json").read_text(encoding="utf-8"))
    metrics["retained_request_payloads"] = sum(isinstance(row.get("request"), dict) for row in rows)
    metrics["final_runner_state_bound"] = True
    _original_write(raw / "actual_scores.json", metrics)
    state_path = raw / "runner_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    passed = (
        metrics["route_models_completed"] == 4
        and metrics["recorded_requests"] == 448
        and metrics["successful_response_rate"] == 1.0
        and metrics["route_identity_verified"] is True
        and metrics["exact_repeat_rate"] >= 0.95
        and metrics["retained_request_payloads"] == 448
        and metrics["service_restarts"] == 0
        and metrics["embedding_health"] == 200
        and metrics["initial_model_restored"] is True
    )
    state.update({
        "status": "completed",
        "claim": "QUALIFIED_TEXT_FLEET_SCREEN_COMPLETE_R2" if passed else "QUALIFIED_TEXT_FLEET_SCREEN_REJECTED_R2",
        "final_state_bound_before_receipt": True,
    })
    _original_write(state_path, state)
    return _original_build(*args, **kwargs)


def _guarded_write(path: pathlib.Path, value: object) -> None:
    if path.name == "runner_state.json" and (path.parent / "receipt.json").is_file():
        return
    _original_write(path, value)


def configure(outdir: pathlib.Path) -> None:
    global _active_outdir
    _active_outdir = outdir
    pipeline = json.loads((outdir / "PIPELINE.json").read_text(encoding="utf-8"))
    prereg = ROOT / pipeline["preregistration"]["path"]
    if sha256_file(prereg) != pipeline["preregistration"]["sha256"]:
        raise ValueError("pipeline preregistration binding mismatch")
    base.TASK_ID = TASK_ID
    base.PRE_REG_SHA256 = pipeline["preregistration"]["sha256"]
    base.SOURCE_HASHES = SOURCE_HASHES
    base.build_provenance = _bound_build
    base.write_json = _guarded_write
    base.__file__ = __file__


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    configure(outdir)
    receipt = base.execute(outdir)
    raw = outdir / "raw"
    metrics = json.loads((raw / "actual_scores.json").read_text(encoding="utf-8"))
    receipt["gates"]["request_retention"] = {"metric": "retained_request_payloads", **_gate(metrics["retained_request_payloads"], 448)}
    receipt["gates"]["terminal_binding"] = {"metric": "final_runner_state_bound", **_gate(metrics["final_runner_state_bound"], True)}
    receipt["evidence"]["request_payloads"] = "raw/samples.jsonl"
    receipt["evidence"]["final_state_binding"] = "raw/runner_state.json"
    receipt.pop("receipt_fingerprint", None)
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    _original_write(raw / "receipt.json", receipt)
    result = outdir / "RESULT.md"
    result.write_text(result.read_text(encoding="utf-8").replace(
        "QUALIFIED_TEXT_FLEET_SCREEN_COMPLETE_R1", "QUALIFIED_TEXT_FLEET_SCREEN_COMPLETE_R2"
    ).replace(
        "QUALIFIED_TEXT_FLEET_SCREEN_REJECTED_R1", "QUALIFIED_TEXT_FLEET_SCREEN_REJECTED_R2"
    ), encoding="utf-8", newline="\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        base.selfcheck()
        return 0
    outdir = args.outdir.resolve()
    receipt = execute(outdir)
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    advance = base.run_text([sys.executable, str(ROOT / "tools/analysis/backlog_pipeline.py"), "advance", TASK_ID, "--to", "EXECUTED", "--actor", "Codex executor"])
    print(json.dumps({"pipeline_advance": advance}, indent=2), flush=True)
    return 0 if advance["returncode"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
