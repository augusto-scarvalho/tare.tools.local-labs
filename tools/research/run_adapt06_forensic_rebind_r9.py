#!/usr/bin/env python3
"""R9 wrapper correcting only R8 terminal file-count projection."""
from __future__ import annotations

import argparse
import json
import pathlib

from tools.research import run_adapt06_forensic_rebind_r8 as r8

ROOT = pathlib.Path(__file__).resolve().parents[2]
TASK_ID = "BACKLOG-ADAPT06-SLOP-LIVE-09"
ORIGINAL_VERIFY = r8.verify_run
EXPECTED = dict(r8.EXPECTED)
EXPECTED.update({
    ROOT / "config/research_backlog_admissions/BACKLOG-ADAPT06-SLOP-LIVE-09.json": "77bc0a61a7d94f815162acbbf65704ba928f561d06b68f4ab65a415dd129b9b8",
    ROOT / "runs/research/BACKLOG-ADAPT06-SLOP-LIVE-09/PRE_REGISTRATION.md": "87440f2ed267fff82fdd492b78b370cc02fcd958b875b52116681345ba86901a",
    ROOT / "runs/research/BACKLOG-ADAPT06-SLOP-LIVE-08/PIPELINE.json": "b9e9a2758ee6e65820de8a9b26ed88f9e0f446a4303c74c71bb7998ea500e903",
    ROOT / "tools/research/run_adapt06_forensic_rebind_r8.py": "c4efef978abfa35b0cbc384506ece6be5ed3c2cd2612f94925b17d5891c18607",
})


def verify_with_terminal_files(raw_dir: pathlib.Path) -> dict:
    projected = ORIGINAL_VERIFY(raw_dir)
    terminal = json.loads((pathlib.Path(raw_dir) / "run.terminal.json").read_text(encoding="utf-8"))
    files = terminal.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("sealed terminal files missing")
    projected["manifest"] = files
    return projected


def execute(outdir: pathlib.Path):
    r8.TASK_ID = TASK_ID
    r8.EXPECTED = EXPECTED
    r8.verify_run = verify_with_terminal_files
    r8.__file__ = __file__
    return r8.execute(outdir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt, metrics = execute(args.outdir.resolve())
    passed = all(row["pass"] for row in receipt["gates"].values())
    claim = "ADAPT06_CLIENT_AFFINITY_FORENSIC_REBOUND_R9" if passed else "ADAPT06_CLIENT_AFFINITY_REJECTED_R9"
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
