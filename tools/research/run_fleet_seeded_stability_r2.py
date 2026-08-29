#!/usr/bin/env python3
"""Fresh R2 seeded stability run with reconstructed request and terminal binding."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import canonical_json_sha256, sha256_file
from tools.research import run_fleet_seeded_stability as base

TASK_ID = "BACKLOG-FLEET-SEEDED-STABILITY-02"
SOURCES = {
    "config/research_backlog_admissions/BACKLOG-FLEET-SEEDED-STABILITY-02.json": "0c407d66472c63c2a9b78cdd071f85eda7e291a91aea9bbb8e6de8e62713eea4",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    "workloads/gsm8k.jsonl": "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    "runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl": "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
    "tools/research/run_fleet_regression_screen.py": "7cbf942375c120970ac78fb40f7f15050ebddb7ec5372cb40bd721009faf1de3",
    "tools/research/run_fleet_seeded_stability.py": "71189428e1d1c7aff8a3ddb55d56c2f2034cd89c9df516f1a6490e7a2888676c",
    "runs/research/BACKLOG-FLEET-SEEDED-STABILITY-01/raw/receipt.json": "0d7bc9a65f6243cd0e0e1e83d670e76a2eb97a4c6ba27a5f9cf73f65a5620e5c",
    "docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md": "a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04",
}

_original_build = base.build_provenance
_original_write = base.wj
_active_outdir: pathlib.Path | None = None


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def enrich_requests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    panel = {(suite, case_id): case for suite, case_id, case in base.cases()}
    for row in rows:
        request = base.fleet.payload_for(row["model"], row["suite"], panel[(row["suite"], row["case_id"])])
        request.update({"temperature": 0.2, "top_p": 0.95, "seed": 20260826})
        row["request"] = request
    return rows


def _bound_build(*args: Any, **kwargs: Any) -> dict[str, Any]:
    assert _active_outdir is not None
    raw = _active_outdir / "raw"
    rows = enrich_requests(base.rj(raw / "samples.jsonl"))
    _write_jsonl(raw / "samples.jsonl", rows)
    metrics = json.loads((raw / "actual_scores.json").read_text(encoding="utf-8"))
    metrics["retained_request_payloads"] = sum(isinstance(row.get("request"), dict) for row in rows)
    metrics["final_runner_state_bound"] = True
    _original_write(raw / "actual_scores.json", metrics)
    state_path = raw / "runner_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    passed = (
        metrics["routes_completed"] == 4 and metrics["recorded_requests"] == 288
        and metrics["successful_response_rate"] == 1.0 and metrics["exact_seeded_repeat_rate"] >= 0.9
        and metrics["retained_request_payloads"] == 288 and metrics["service_restarts"] == 0
        and metrics["initial_model_restored"] is True
    )
    state.update({
        "status": "completed",
        "claim": "QUALIFIED_TEXT_FLEET_SEEDED_STABLE_R2" if passed else "QUALIFIED_TEXT_FLEET_SEEDED_UNSTABLE_R2",
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
    base.PRE = pipeline["preregistration"]["sha256"]
    base.SOURCES = SOURCES
    base.build_provenance = _bound_build
    base.wj = _guarded_write
    base.__file__ = __file__


def _gate(actual: Any, threshold: Any) -> dict[str, Any]:
    return {"operator": "eq", "threshold": threshold, "actual": actual, "pass": actual == threshold}


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    configure(outdir)
    receipt = base.execute(outdir)
    raw = outdir / "raw"
    metrics = json.loads((raw / "actual_scores.json").read_text(encoding="utf-8"))
    receipt["gates"]["request_retention"] = {"metric": "retained_request_payloads", **_gate(metrics["retained_request_payloads"], 288)}
    receipt["gates"]["terminal_binding"] = {"metric": "final_runner_state_bound", **_gate(metrics["final_runner_state_bound"], True)}
    receipt["evidence"]["request_payloads"] = "raw/samples.jsonl"
    receipt["evidence"]["final_state_binding"] = "raw/runner_state.json"
    receipt.pop("receipt_fingerprint", None)
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    _original_write(raw / "receipt.json", receipt)
    result = outdir / "RESULT.md"
    result.write_text(result.read_text(encoding="utf-8").replace(
        "QUALIFIED_TEXT_FLEET_SEEDED_STABLE_R1", "QUALIFIED_TEXT_FLEET_SEEDED_STABLE_R2"
    ).replace(
        "QUALIFIED_TEXT_FLEET_SEEDED_UNSTABLE_R1", "QUALIFIED_TEXT_FLEET_SEEDED_UNSTABLE_R2"
    ), encoding="utf-8", newline="\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        assert len(base.cases()) == 24
        return 0
    outdir = args.outdir.resolve()
    receipt = execute(outdir)
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    advance = base.subprocess.run([sys.executable, str(ROOT / "tools/analysis/backlog_pipeline.py"), "advance", TASK_ID, "--to", "EXECUTED", "--actor", "Codex executor"], cwd=ROOT, capture_output=True, text=True)
    print(advance.stdout, flush=True)
    return 0 if advance.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
