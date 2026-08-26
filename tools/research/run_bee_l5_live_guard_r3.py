#!/usr/bin/env python3
"""Direct-file bootstrap for frozen BEE-L5 R2 protocol."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.research import run_bee_l5_live_guard as core
from tools.research.run_bee_l5_live_guard_r2 import tokenize

TASK_ID = "BACKLOG-BEE-L5-LIVE-GUARD-03"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    core.TASK_ID = TASK_ID
    core.tokenize = tokenize
    core.EXPECTED.update({
        ROOT / "config/research_backlog_admissions/BACKLOG-BEE-L5-LIVE-GUARD-02.json": "41f4b27a04347120c01a7658936e3f8a2b748cee8cc1a544668a52183fe700b4",
        ROOT / "runs/research/BACKLOG-BEE-L5-LIVE-GUARD-02/PRE_REGISTRATION.md": "b4731587185a2bbd6d2720b78bad7eeb951e01e181e0c571ddc915f163f975e2",
        ROOT / "runs/research/BACKLOG-BEE-L5-LIVE-GUARD-02/ABORTED.md": "57783ba9b1d080b104ce696082c46babcfde3f9dfae265f0f350405de74ec3ad",
        ROOT / "tools/research/run_bee_l5_live_guard_r2.py": "3a0d325f614708f2029fcc1f40c3c904f80a3d4f6f9c4320ec6cf664f122d118",
        ROOT / "config/research_backlog_admissions/BACKLOG-BEE-L5-LIVE-GUARD-03.json": "88995a1931c9137d6d5a07a0fd1894d3f24801e36076f95c680c0d6e757f2020",
        ROOT / "runs/research/BACKLOG-BEE-L5-LIVE-GUARD-03/PRE_REGISTRATION.md": "89e7fb7ad4e521426f48c2dc8965fc692c21e09cfe3bf8104da6fe435b10055b",
    })
    receipt, metrics = core.run(args.outdir.resolve())
    passed = all(row["pass"] for row in receipt["gates"].values())
    claim = "BEE_L5_LIVE_GUARD_QUALIFIED_R3" if passed else "BEE_L5_FALSE_POSITIVE_CONFIRMED_R3"
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
