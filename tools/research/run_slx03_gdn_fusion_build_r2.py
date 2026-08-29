#!/usr/bin/env python3
"""R2 of the SLX-03 build qualification with explicit CUDA discovery."""
from __future__ import annotations

import argparse
import pathlib

from tools.research import run_slx03_gdn_fusion_build as base

ROOT = pathlib.Path(__file__).resolve().parents[2]
TASK_ID = "BACKLOG-SLX03-GDN-FUSION-BUILD-02"
CUDA_PATH = "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def configured_source_command(*args: str, timeout: int = 1800):
    return base.cmd(
        "--cd",
        base.SOURCE,
        "env",
        f"PATH={CUDA_PATH}",
        "CUDACXX=/usr/local/cuda/bin/nvcc",
        *args,
        timeout=timeout,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    base.TASK_ID = TASK_ID
    base.BUILD = "/home/augus/src/slop.cpp-main/build-slx03-gdn-audit-02"
    base.INPUTS = {
        "config/research_backlog_admissions/BACKLOG-SLX03-GDN-FUSION-BUILD-02.json": "ea7014fc48736ba767f7d05e7af39a61ae8effc1448b273e659a23c875f2ccce",
        "runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-02/PRE_REGISTRATION.md": "902fb53ce46889f31533f00721192018b1a4d15a59888cee5f1c7521d0c0db09",
        "runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-03/raw/receipt.json": "5ebe76094b02ed4557533fa6def8b64241a3b1ad4fc0124c7d10359df6e8589e",
        "runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-03/REVIEW.json": "c865e2f08d9bf36f7366b8ed256999babc1356c97067f6f9ac42d1cfdbabdea8",
    }
    base.at_source = configured_source_command
    base.run(args.outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
