#!/usr/bin/env python3
"""Resume R9 with the complete delegated abort-receipt ledger."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import run_slx08_relevance_prefill_r9 as r9

TASK_ID = "BACKLOG-SLX08-RELEVANCE-PREFILL-10"
PRE_REG_SHA256 = "ade2a996154f13fcfbe74a6c5e201ea837c6fc815da3fced948dc70efc844070"
SOURCE_HASHES = {
    "runs/autonomous/SLX08-RELEVANCE-R9-2026-08-30/FINAL.json": "f065df7920bfb7173db5912e08be3698d8a4045c2978e09f23682419c78d311f",
    "runs/autonomous/SLX08-RELEVANCE-R9-2026-08-30/WORKER_EXIT.json": "18bc271555a742fe3f9198d08f1ecf7de07f515131c8856eacf43e3a27546b24",
    "tools/research/run_slx08_relevance_prefill_r9.py": "451f4df230289ef4b436266cf4c2952d44e53db2e174a742fd9c227a4c48e5b5",
    "runs/autonomous/SLX08-RELEVANCE-R6-2026-08-30/FINAL.json": "d4968708056ee0ed301aae5070d9514cd62834aa759738cfc54c061327e3aef0",
    "runs/autonomous/SLX08-RELEVANCE-R6-2026-08-30/WORKER_EXIT.json": "275e8883fdbe34459c96c11fb94a0fc59d982200ca63df1bcd78b0039d8ccc75",
}


def configure_delegate() -> None:
    r9.TASK_ID = TASK_ID
    r9.PRE_REG_SHA256 = PRE_REG_SHA256
    r9.SOURCE_HASHES = SOURCE_HASHES
    r9.__file__ = __file__


def execute(outdir: pathlib.Path) -> tuple[dict, dict]:
    configure_delegate()
    return r9.execute(outdir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt, metrics = execute(args.outdir.resolve())
    passed = all(gate["pass"] for gate in receipt["gates"].values())
    claim = "SLX08_RELEVANCE_SELECTED_PREFILL_QUALIFIED_R10" if passed else "SLX08_RELEVANCE_SELECTED_PREFILL_REJECTED_R10"
    failed = [gate for gate, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review. Dense/naive/relevance accuracy: "
        f"{metrics['dense_accuracy']:.4f}/{metrics['naive_accuracy']:.4f}/{metrics['relevance_accuracy']:.4f}; relevance p50 TTFT speedup "
        f"{metrics['relevance_vs_dense_p50_ttft_speedup']:.4f}x; failed gates: {', '.join(failed) if failed else 'none'}. "
        "Bounded to client-selected server token compaction on the frozen R10 panel.\n",
        encoding="utf-8",
    )
    print(json.dumps({"claim": claim, "metrics": metrics, "gates": receipt["gates"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
