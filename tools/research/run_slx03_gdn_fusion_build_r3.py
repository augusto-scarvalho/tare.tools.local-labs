#!/usr/bin/env python3
"""R3 of the SLX-03 build qualification with a preflighted import path."""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.research import run_slx03_gdn_fusion_build as base

TASK_ID = "BACKLOG-SLX03-GDN-FUSION-BUILD-03"
CUDA_PATH = "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def configured_source_command(*args: str, timeout: int = 1800):
    return base.cmd("--cd", base.SOURCE, "env", f"PATH={CUDA_PATH}", "CUDACXX=/usr/local/cuda/bin/nvcc", *args, timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    base.TASK_ID = TASK_ID
    base.BUILD = "/home/augus/src/slop.cpp-main/build-slx03-gdn-audit-03"
    base.INPUTS = {
        "config/research_backlog_admissions/BACKLOG-SLX03-GDN-FUSION-BUILD-03.json": "f33e6be5eeb062a227383ccb9d52b294f854a94038e5ef3e42572c6f53fff44c",
        "runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-03/PRE_REGISTRATION.md": "5c1eb4eccedb6537a54e9e07a0cea15e4ea113e6ded5b337bb8194dd68e4f1b4",
        "runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-03/raw/receipt.json": "5ebe76094b02ed4557533fa6def8b64241a3b1ad4fc0124c7d10359df6e8589e",
        "runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-03/REVIEW.json": "c865e2f08d9bf36f7366b8ed256999babc1356c97067f6f9ac42d1cfdbabdea8"
    }
    base.at_source = configured_source_command
    base.run(args.outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
