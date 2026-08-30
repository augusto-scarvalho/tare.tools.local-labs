#!/usr/bin/env python3
"""Repeat GDN02 retained-state R2 with a literal POSIX model path."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.research import run_gdn02_learned_state_r2 as r2

TASK_ID = "BACKLOG-GDN02-LEARNED-STATE-03"
MODEL = "/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe"
EXPECTED = {
    ROOT / "config/research_backlog_admissions/BACKLOG-GDN02-LEARNED-STATE-03.json": "d52a136798d147c0f28ec8e1511aa176ab41146d541bb37ead04c0053bfe3d3f",
    ROOT / "runs/research/BACKLOG-GDN02-LEARNED-STATE-03/PRE_REGISTRATION.md": "3f72a3d0f978dda17ce6ffbf290f10bf82f74e0fa7e6545eb1ff25fdbd5ecc9d",
    ROOT / "runs/research/BACKLOG-GDN02-LEARNED-STATE-02/raw/run.terminal.json": "4ab5f6a79bd1b6b73d895add25d0d1a5b282cc6e3bb0674230d77fbe50570e6b",
    ROOT / "tools/research/run_gdn02_learned_state_r2.py": "da6520358d880ac8b4fd94265face19e0324a28dd538051d7f5a16fc9c378dc9",
    ROOT / "tools/research/gdn02_learned_state_worker_r2.py": "c89d21fa26c26197e6facac92cc13f175502e898c3eceffcb6313f839521f9e1",
    ROOT / "tools/research/gdn02_retained_scorer.py": "a488bcd69196a3186a5ef2aeedb1427e80b409c77e8a9d753942c740f934f91e",
    ROOT / "runs/research/BACKLOG-GDN02-LEARNED-STATE-01/raw/receipt.json": "3222aceaa925b48fda0b9eb684e32f5dac917e4e86a80c6fc65279cddbb7f236",
    r2.CORPUS: "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
}


def execute(outdir: pathlib.Path):
    previous_task, previous_model, previous_expected = r2.TASK_ID, r2.MODEL, r2.EXPECTED
    r2.TASK_ID = TASK_ID
    r2.MODEL = MODEL
    r2.EXPECTED = EXPECTED
    try:
        return r2.execute(outdir)
    finally:
        r2.TASK_ID = previous_task
        r2.MODEL = previous_model
        r2.EXPECTED = previous_expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt, metrics = execute(args.outdir.resolve())
    passed = all(row["pass"] for row in receipt["gates"].values())
    claim = "GDN02_RETAINED_WSL_IDENTITY_QUALIFIED_R3" if passed else "GDN02_RETAINED_WSL_IDENTITY_REJECTED_R3"
    failed = [name for name, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Retained `{metrics['retained_decisive_layer_cells']}` cells and "
        f"`{metrics['retained_collateral_cosines']}` collateral cosines; scorer match "
        f"`{metrics['recomputed_metric_match_rate']:.4%}`; failed gates: "
        f"`{', '.join(failed) if failed else 'none'}`. Scope remains representation-level.\n",
        encoding="utf-8",
    )
    print(json.dumps({"claim": claim, "gates": receipt["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
