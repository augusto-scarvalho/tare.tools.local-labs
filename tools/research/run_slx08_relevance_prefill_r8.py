#!/usr/bin/env python3
"""Resume R7 with direct-file repository import bootstrap."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import run_slx08_relevance_prefill_r7 as r7

TASK_ID = "BACKLOG-SLX08-RELEVANCE-PREFILL-08"
PRE_REG_SHA256 = "faa604fce631958f8b01a78fd5f4eb093c09b6d1e1daca737e35e3d56c9e4efa"
SOURCE_HASHES = {
    "runs/autonomous/SLX08-RELEVANCE-R7-2026-08-30/FINAL.json": "7e339315aa0feef47cc5b6971a4892494f170802d5b944b3be9cc0461fa4ba0",
    "runs/autonomous/SLX08-RELEVANCE-R7-2026-08-30/WORKER_EXIT.json": "eb7caa299abc71fd960777821c4d4aae4afeb0139312ab76e4a5e4a20946f2fe",
    "tools/research/run_slx08_relevance_prefill_r7.py": "21edf4276875cbc6b36faae5f3109fb88c18a3a6ade92444fe70076cc905a196",
}


def execute(outdir: pathlib.Path) -> tuple[dict, dict]:
    r7.TASK_ID = TASK_ID
    r7.PRE_REG_SHA256 = PRE_REG_SHA256
    r7.SOURCE_HASHES = SOURCE_HASHES
    r7.__file__ = __file__
    return r7.execute(outdir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt, metrics = execute(args.outdir.resolve())
    passed = all(gate["pass"] for gate in receipt["gates"].values())
    claim = "SLX08_RELEVANCE_SELECTED_PREFILL_QUALIFIED_R8" if passed else "SLX08_RELEVANCE_SELECTED_PREFILL_REJECTED_R8"
    failed = [gate for gate, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review. Dense/naive/relevance accuracy: "
        f"{metrics['dense_accuracy']:.4f}/{metrics['naive_accuracy']:.4f}/{metrics['relevance_accuracy']:.4f}; relevance p50 TTFT speedup "
        f"{metrics['relevance_vs_dense_p50_ttft_speedup']:.4f}x; failed gates: {', '.join(failed) if failed else 'none'}. "
        "Bounded to client-selected server token compaction on the frozen R8 panel.\n",
        encoding="utf-8",
    )
    print(json.dumps({"claim": claim, "metrics": metrics, "gates": receipt["gates"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
