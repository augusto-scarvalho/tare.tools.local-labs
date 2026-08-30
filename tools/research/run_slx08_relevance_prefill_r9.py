#!/usr/bin/env python3
"""Resume R8 with the exact source digest copied from disk."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import run_slx08_relevance_prefill_r8 as r8

TASK_ID = "BACKLOG-SLX08-RELEVANCE-PREFILL-09"
PRE_REG_SHA256 = "3a8a36330728d5101be56cdbbc50f9818404352b7b945fd152bfb0bcc76703de"
SOURCE_HASHES = {
    "runs/autonomous/SLX08-RELEVANCE-R8-2026-08-30/FINAL.json": "d0c4fe952604ebe0e8fef4a3a9649ac1b1aff10a54333b41b22e53a4c2e37358",
    "runs/autonomous/SLX08-RELEVANCE-R8-2026-08-30/WORKER_EXIT.json": "6c0b13eee349d2172310f96cdcac3251ddc4fd4b88b867686d29f1dc120cb3f6",
    "tools/research/run_slx08_relevance_prefill_r8.py": "4ec2628c52d182c11d796295e44fdfdcb5be4415cde29d79812e60243e8e216d",
}


def configure_delegate() -> None:
    r8.TASK_ID = TASK_ID
    r8.PRE_REG_SHA256 = PRE_REG_SHA256
    r8.SOURCE_HASHES = SOURCE_HASHES
    r8.__file__ = __file__


def execute(outdir: pathlib.Path) -> tuple[dict, dict]:
    configure_delegate()
    return r8.execute(outdir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt, metrics = execute(args.outdir.resolve())
    passed = all(gate["pass"] for gate in receipt["gates"].values())
    claim = "SLX08_RELEVANCE_SELECTED_PREFILL_QUALIFIED_R9" if passed else "SLX08_RELEVANCE_SELECTED_PREFILL_REJECTED_R9"
    failed = [gate for gate, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review. Dense/naive/relevance accuracy: "
        f"{metrics['dense_accuracy']:.4f}/{metrics['naive_accuracy']:.4f}/{metrics['relevance_accuracy']:.4f}; relevance p50 TTFT speedup "
        f"{metrics['relevance_vs_dense_p50_ttft_speedup']:.4f}x; failed gates: {', '.join(failed) if failed else 'none'}. "
        "Bounded to client-selected server token compaction on the frozen R9 panel.\n",
        encoding="utf-8",
    )
    print(json.dumps({"claim": claim, "metrics": metrics, "gates": receipt["gates"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
