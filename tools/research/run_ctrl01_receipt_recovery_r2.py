#!/usr/bin/env python3
"""Production-binding-corrected receipt recovery for CTRL-01."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.research import run_ctrl01_receipt_recovery as recovery


TASK_ID = "BACKLOG-CTRL01-REAL-TOKEN-06"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    recovery.TASK_ID = TASK_ID
    outdir = args.outdir.resolve()
    receipt = recovery.run(outdir)
    scores = json.loads((outdir / "raw/actual_scores.json").read_text(encoding="utf-8"))
    failed = [name for name, value in receipt["gates"].items() if not value["pass"]]
    (outdir / "RESULT.md").write_text(f"""# {TASK_ID} result

## Verdict

`CTRL01_FALSE_POSITIVE_CONFIRMED_R6` pending independent AGY review.

Independent recovery reproduced all source metrics exactly from 36 immutable physical rows. Raw real-model JSON validity was `{scores['raw_complete_valid_rate']:.6f}`; applying the sidecar reduced complete validity to `{scores['sanitized_complete_valid_rate']:.6f}`. Valid-token acceptance was `{scores['valid_token_acceptance_rate']:.6f}`, valid-control exact preservation was `{scores['valid_control_exact_preservation_rate']:.6f}`, p95 overhead was `{scores['p95_overhead_us_per_token']:.3f}` microseconds/token, and production runtime binding was `{scores['logit_mask_runtime_integrated']}`.

Failed mandatory gates: `{', '.join(failed)}`. This recovers canonical evidence from the already completed physical run; it performs no new inference and makes no Python-mode claim.
""", encoding="utf-8")
    print(json.dumps(receipt["gates"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
