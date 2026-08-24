#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone


ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "benchmarks"))

import optuna_runtime_tune as tune  # noqa: E402


OUTPUT = pathlib.Path(__file__).resolve().parent / "reserve-81920"
PROBES = pathlib.Path(__file__).resolve().parent / "baseline-probes.json"


def main() -> int:
    if tune.health(8080):
        raise SystemExit("canonical 8080 must be stopped before the reserve A/B")
    if tune.health(tune.PORT):
        raise SystemExit(f"candidate port {tune.PORT} is already occupied")
    tune.SERVER_BIN = "/home/augus/src/slop.cpp-candidate-71676e46c/build/bin/llama-server"
    reference_hashes = json.loads(PROBES.read_text(encoding="utf-8"))["hashes"]
    control = tune.measure_cell(
        3,
        512,
        round_name="reserve-control",
        reps=3,
        reference_hashes=reference_hashes,
        output=OUTPUT,
        ctx_size=81920,
    )
    challenger = None
    if control["feasible"]:
        challenger = tune.measure_cell(
            4,
            1024,
            round_name="reserve-challenger",
            reps=3,
            reference_hashes=reference_hashes,
            output=OUTPUT,
            ctx_size=81920,
        )
    decision = (
        tune.compare_pair(control, challenger)
        if challenger is not None
        else {"confirmed": False, "reason": "control failed; challenger not launched"}
    )
    report = {
        "campaign": "HOST-WSL-TUNE reserve-81920",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "binary": tune.SERVER_BIN,
        "binary_revision": "71676e46c",
        "context": 81920,
        "hard_vram_free_floor_mib": 4096,
        "control": control,
        "challenger": challenger,
        "decision": decision,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "control_feasible": control["feasible"],
        "challenger_feasible": challenger is not None and challenger["feasible"],
        "decision": decision,
    }, indent=2))
    return 0 if challenger is not None and challenger["feasible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
