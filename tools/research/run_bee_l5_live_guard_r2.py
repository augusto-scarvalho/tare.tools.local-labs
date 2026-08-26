#!/usr/bin/env python3
"""Corrected token-piece parser for frozen BEE-L5 live-guard protocol."""
from __future__ import annotations

import argparse
import json
import pathlib

from tools.research import run_bee_l5_live_guard as core

ROOT = pathlib.Path(__file__).resolve().parents[2]
TASK_ID = "BACKLOG-BEE-L5-LIVE-GUARD-02"


def tokenize(text: str):
    response = core.post_json("/tokenize", {"content": text, "with_pieces": True})
    tokens = response.get("tokens") or []
    if tokens and not all(isinstance(token, dict) and isinstance(token.get("piece"), str) for token in tokens):
        raise RuntimeError(f"unexpected /tokenize response: {response}")
    return [token["piece"] for token in tokens]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    core.TASK_ID = TASK_ID
    core.tokenize = tokenize
    core.EXPECTED.update({
        ROOT / "config/research_backlog_admissions/BACKLOG-BEE-L5-LIVE-GUARD-02.json": "41f4b27a04347120c01a7658936e3f8a2b748cee8cc1a544668a52183fe700b4",
        ROOT / "runs/research/BACKLOG-BEE-L5-LIVE-GUARD-02/PRE_REGISTRATION.md": "b4731587185a2bbd6d2720b78bad7eeb951e01e181e0c571ddc915f163f975e2",
        ROOT / "runs/research/BACKLOG-BEE-L5-LIVE-GUARD-01/ABORTED.md": "ae1da170d097684dcfb72e8d603bcb78f5a1f19f8266b64a768fbe157d10cc52",
        ROOT / "tools/research/run_bee_l5_live_guard.py": "7ce5e9cd47fde2d5e53c923b012d4cef2e665b27b634f917e3f6e311e79ab5f8",
    })
    receipt, metrics = core.run(args.outdir.resolve())
    passed = all(row["pass"] for row in receipt["gates"].values())
    claim = "BEE_L5_LIVE_GUARD_QUALIFIED_R2" if passed else "BEE_L5_FALSE_POSITIVE_CONFIRMED_R2"
    failed = [gate for gate, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent AGY review.\n\n"
        f"Real teacher traces: {metrics['real_legitimate_traces']}; false alarms: {metrics['teacher_false_positives']} ({metrics['false_alarm_fpr']:.4%}). "
        f"Live pathological baselines: {metrics['live_pathological_baselines']}; guard triggers/aborts: {metrics['stream_aborts_confirmed']}/25; "
        f"median trigger token: {metrics['median_trigger_token']}; median savings: {metrics['median_token_savings']:.4%}; "
        f"guard p95: {metrics['guard_p95_us_per_token']:.3f} us/token. Failed gates: {', '.join(failed) if failed else 'none'}. "
        "This is a client-side streaming intervention, not server integration.\n", encoding="utf-8")
    print(json.dumps({"claim": claim, "metrics": metrics, "gates": receipt["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
