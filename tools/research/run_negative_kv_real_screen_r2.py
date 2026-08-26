#!/usr/bin/env python3
"""Continuation of the frozen real-Qwen negative KV screen after binding abort."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.research import run_negative_kv_real_screen as r1

TASK_ID = "BACKLOG-NEGATIVE-KV-REAL-SCREEN-02"
ADMISSION = r1.ROOT / "config/research_backlog_admissions/BACKLOG-NEGATIVE-KV-REAL-SCREEN-02.json"
PREREG = r1.ROOT / "runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-02/PRE_REGISTRATION.md"
PREDECESSOR_INPUTS = {
    r1.ROOT / "runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-01/PRE_REGISTRATION.md":
        "dce466a9cdfc8a34b7baf1afe52f15752891bd063a32ac3d2f28450d7aebfe11",
    r1.ROOT / "runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-01/ABORTED.md":
        "c5cd9acb6070fb704ccf9f6b17230a2cfcad89b9bebb65daca9f988c8eb6590b",
    r1.ROOT / "runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-01/raw/worker.stderr.log":
        "61df470821dfd43c4c9a799c2b2a24e1fa5fa24f936a8eacff279c581d0ebbc4",
    r1.ROOT / "runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-01/raw/service_maintenance.json":
        "d4bf245ae8cf5a77665c62aeaba87118be977805a7a036fb1749b1fd3d2f90e7",
}
EXPECTED_INPUTS = {
    path: digest for path, digest in r1.EXPECTED_INPUTS.items()
    if path not in {r1.ADMISSION, r1.PREREG}
}
EXPECTED_INPUTS.update({
    ADMISSION: "903dad8498d0d54e9887349ca5477bdf8e531c025b2ed8abd6dd6261a9cf881a",
    PREREG: "8d4573e52651dd2aebc7ef369d9e8cf233bfa3b0257373f041659cf664896438",
    **PREDECESSOR_INPUTS,
})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path,
                        default=r1.ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt = r1.run_experiment(
        args.outdir.resolve(), task_id=TASK_ID, expected_inputs=EXPECTED_INPUTS,
        continuation_inputs=PREDECESSOR_INPUTS,
    )
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
