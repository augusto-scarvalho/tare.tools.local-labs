#!/usr/bin/env python3
"""R10 import-safe wrapper for the frozen ADAPT06 forensic scorer."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import run_adapt06_forensic_rebind_r8 as r8  # noqa: E402
from tools.research import run_adapt06_forensic_rebind_r9 as r9  # noqa: E402

TASK_ID = "BACKLOG-ADAPT06-SLOP-LIVE-10"
EXPECTED = dict(r9.EXPECTED)
EXPECTED.update({
    ROOT / "config/research_backlog_admissions/BACKLOG-ADAPT06-SLOP-LIVE-10.json": "a42b31f138d66867a5958df7b5727a74cda8f3a128ba738d144240ad748e4600",
    ROOT / "runs/research/BACKLOG-ADAPT06-SLOP-LIVE-10/PRE_REGISTRATION.md": "17462d2a0be792dc48e4410deca2114da5456a9e8af014ecbd19585f7b14eaec",
    ROOT / "runs/research/BACKLOG-ADAPT06-SLOP-LIVE-09/PIPELINE.json": "6d9fec5661df7eaeb6b92a08ca3a06c71b18dbaaf92f10ea64eb654c674fe1f4",
    ROOT / "tools/research/run_adapt06_forensic_rebind_r9.py": "c972779c9fb3d00b92431ef932fe1b940679b023a9a47e298cfdc5c40c1aadeb",
    ROOT / "runs/research/BACKLOG-ADAPT06-SLOP-LIVE-09/runner.stderr.log": "69a5b43b1e682368f5dc4b935bff8f762d10e8b59782d5b96751744a986616dd",
    ROOT / "runs/autonomous/FULL-BACKLOG-2026-08-29/watchers/002-BACKLOG-ADAPT06-SLOP-LIVE-09/WORKER_EXIT.json": "715343282c22a85c9fbd7e3aafdf28d26b14b87e6bf0ad2296c2bf7abab7c21d",
})


def execute(outdir: pathlib.Path):
    r8.TASK_ID = TASK_ID
    r8.EXPECTED = EXPECTED
    r8.verify_run = r9.verify_with_terminal_files
    r8.__file__ = __file__
    return r8.execute(outdir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt, metrics = execute(args.outdir.resolve())
    passed = all(row["pass"] for row in receipt["gates"].values())
    claim = "ADAPT06_CLIENT_AFFINITY_FORENSIC_REBOUND_R10" if passed else "ADAPT06_CLIENT_AFFINITY_REJECTED_R10"
    failed = [name for name, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review. Recomputed 36 baseline and 72 routed UTF-8 digests; "
        f"route-correct match `{metrics['route_correct_counterfactual_match_rate']:.4%}`; switch reduction "
        f"`{metrics['requested_route_switch_reduction']:.4%}`; failed gates: {', '.join(failed) if failed else 'none'}. "
        "Client ordering only; no speed, cache-isolation or server-native scheduling claim.\n",
        encoding="utf-8",
    )
    print(json.dumps({"claim": claim, "gates": receipt["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
