#!/usr/bin/env python3
"""Resume R6 after correcting only pre-measurement tokenizer ordering."""
from __future__ import annotations

import argparse
import json
import pathlib

from tools.research import run_slx08_relevance_prefill_r6 as r6

TASK_ID = "BACKLOG-SLX08-RELEVANCE-PREFILL-07"
PRE_REG_SHA256 = "039ae6af84c070e1b042c8883b767910ec304df0d5725fc42f7a76cc08469c2c"
SOURCE_HASHES = {
    "runs/research/BACKLOG-SLX08-PHYSICAL-PREFILL-05/raw/receipt.json": "ed54519837bf23f174fcf4fdeef451a5d8e776fc8d3857b91b5e61b5190eb4eb",
    "runs/research/BACKLOG-SLX08-PHYSICAL-PREFILL-05/REVIEW.json": "1415bc422459a095e93b0474fca0d262474a8c87480ac618f3402bf477f96de6",
    "runs/autonomous/SLX08-RELEVANCE-R6-2026-08-30/FINAL.json": "d4968708056ee0ed301aae5070d9514cd62834aa759738cfc54c061327e3aef0",
    "runs/autonomous/SLX08-RELEVANCE-R6-2026-08-30/WORKER_EXIT.json": "275e8883fdbe34459c96c11fb94a0fc59d982200ca63df1bcd78b0039d8ccc75",
    "tools/research/run_slx08_relevance_prefill_r6.py": "fccac2347d78c3307448fe30c4cdc25363863e01a935286263a86df034f847e2",
}


def freeze_fixtures() -> list[dict]:
    status, gateway = r6.base.health(8080)
    if status != 200 or not isinstance(gateway, dict) or gateway.get("current_model") != "qwen38" or not gateway.get("backend_healthy"):
        raise RuntimeError(f"healthy qwen38 tokenizer backend required: {status}: {gateway}")
    backend_port = gateway.get("backend_port")
    if not isinstance(backend_port, int):
        raise RuntimeError(f"gateway did not expose an integer backend port: {gateway}")
    original_base = r6.base.TEMP_BASE
    try:
        r6.base.TEMP_BASE = f"http://127.0.0.1:{backend_port}"
        return [r6.build_fixture(case_id) for case_id in range(r6.base.PAIRS)]
    finally:
        r6.base.TEMP_BASE = original_base


def execute(outdir: pathlib.Path) -> tuple[dict, dict]:
    fixtures = freeze_fixtures()
    if len(fixtures) != 64 or any(len(fixture["tokens"]) != 4096 for fixture in fixtures):
        raise RuntimeError("frozen fixture set has the wrong shape")

    r6.TASK_ID = TASK_ID
    r6.PRE_REG_SHA256 = PRE_REG_SHA256
    r6.SOURCE_HASHES = SOURCE_HASHES
    r6.build_fixture = lambda case_id: fixtures[case_id]

    original_provenance = r6.build_provenance
    original_write_json = r6.base.write_json
    wrapper_path = pathlib.Path(__file__).resolve()

    def bound_provenance(**kwargs):
        kwargs["script_path"] = wrapper_path
        input_paths = list(kwargs["input_paths"])
        if wrapper_path not in input_paths:
            input_paths.append(wrapper_path)
        kwargs["input_paths"] = input_paths
        return original_provenance(**kwargs)

    def write_json_with_failure_receipt(path: pathlib.Path, value) -> None:
        if path.name == "failure_reproduction.json":
            value = {
                "r6_pre_measurement_failure": {
                    "measured_requests": 0,
                    "worker_exit": SOURCE_HASHES["runs/autonomous/SLX08-RELEVANCE-R6-2026-08-30/WORKER_EXIT.json"],
                    "watcher_final": SOURCE_HASHES["runs/autonomous/SLX08-RELEVANCE-R6-2026-08-30/FINAL.json"],
                    "cause": "experimental tokenizer called before temporary server startup",
                    "r7_change": "freeze fixtures through healthy qwen38 backend before maintenance",
                },
                "r7_invalid_selection_controls": value,
            }
        original_write_json(path, value)

    r6.build_provenance = bound_provenance
    r6.base.write_json = write_json_with_failure_receipt
    try:
        return r6.execute(outdir)
    finally:
        r6.build_provenance = original_provenance
        r6.base.write_json = original_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=r6.ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt, metrics = execute(args.outdir.resolve())
    passed = all(gate["pass"] for gate in receipt["gates"].values())
    claim = "SLX08_RELEVANCE_SELECTED_PREFILL_QUALIFIED_R7" if passed else "SLX08_RELEVANCE_SELECTED_PREFILL_REJECTED_R7"
    failed = [gate for gate, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review. Dense/naive/relevance accuracy: "
        f"{metrics['dense_accuracy']:.4f}/{metrics['naive_accuracy']:.4f}/{metrics['relevance_accuracy']:.4f}; relevance p50 TTFT speedup "
        f"{metrics['relevance_vs_dense_p50_ttft_speedup']:.4f}x; failed gates: {', '.join(failed) if failed else 'none'}. "
        "Bounded to client-selected server token compaction on the frozen R7 panel.\n",
        encoding="utf-8",
    )
    print(json.dumps({"claim": claim, "metrics": metrics, "gates": receipt["gates"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
