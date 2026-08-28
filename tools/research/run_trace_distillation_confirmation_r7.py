#!/usr/bin/env python3
"""Operational R7 wrapper for the unchanged R6 scientific continuation."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import canonical_json_sha256, sha256_file
from tools.research import run_trace_distillation_confirmation_r6 as r6
from tools.research import run_trace_distillation_training_r2 as r2

TASK_ID = "BACKLOG-ADAPT-TRACE-DISTILL-07"
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-ADAPT-TRACE-DISTILL-07.json"
PREREGISTRATION = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-07/PRE_REGISTRATION.md"
R6_PREREGISTRATION = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-06/PRE_REGISTRATION.md"
R6_TIMEOUT_LOG = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-06/runner.stderr.log"
R6_WATCH_FINAL = ROOT / "runs/autonomous/EXPERIMENT-WATCH-2026-08-27-R6/FINAL.json"


def long_systemctl(action: str) -> None:
    completed = subprocess.run(
        [
            "wsl", "-d", "Ubuntu-24.04", "-u", "root", "--",
            "systemctl", action, "llm-inference.service",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"systemctl {action} failed ({completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )


def configure_r6() -> None:
    r6.TASK_ID = TASK_ID
    r6.r2.systemctl = long_systemctl
    r6.EXPECTED_STATIC = {
        ADMISSION: "c4412e7ccea811ace51e91e92c3e285bc451b1a2f4b4474419a045bcde789f45",
        PREREGISTRATION: "39fbf88e3ad1683a3c268d1bf93dd7eb09511356103fc12664639c23c605cae7",
        R6_PREREGISTRATION: "bbf8cf1c5f7a952460da014c942cada1e74a7646d4101fcd30324e56b8e51a78",
        r6.LEDGER_PATH: "817d595739eff09e3b7d2a78f82b331f8a411d0874b54abe8d43055a2d3066fc",
        R6_TIMEOUT_LOG: "3eb3f1c1ce4d2ca2809df24133097bb9301cfa952bb9e8eb53247a7be5307836",
        R6_WATCH_FINAL: "6459899935b76ec1a17d6a3f0948f901a340a53412375f132a7aa9f577b4400b",
        r6.SOURCE / "raw/training_pairs.json": "5c3f0d5fd80d97351839bca1e38685e5e21b3357dfa56077f44f02b857bfe4cc",
        r6.SOURCE / "raw/service_maintenance.json": "0831e29cf2e138eb90ed663c07ab1252e0a82ae4a024c5db6df4649f0df49825",
        r2.DEFAULT_MATH_PATH: "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
        r2.TEACHER_PATH: "dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e",
        r2.DEFAULT_QA_PATH: "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
    }


def execute(outdir: pathlib.Path) -> dict:
    configure_r6()
    receipt = r6.execute(outdir)
    receipt["provenance"]["script"] = {
        "path": str(pathlib.Path(__file__).resolve()),
        "sha256": sha256_file(pathlib.Path(__file__).resolve()),
    }
    receipt.pop("receipt_fingerprint", None)
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    r6.write_json(outdir / "raw/receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
