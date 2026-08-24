#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone


ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "benchmarks"))

import optuna_runtime_tune as tune  # noqa: E402


HERE = pathlib.Path(__file__).resolve().parent
OUTPUT = HERE / "runtime-53248"


def main() -> int:
    if tune.health(8080) or tune.health(tune.PORT):
        raise SystemExit("runtime A/B requires free canonical and candidate ports")
    tune.SERVER_BIN = "/home/augus/src/slop.cpp-candidate-71676e46c/build/bin/llama-server"
    hashes = json.loads((HERE / "baseline-probes.json").read_text(encoding="utf-8"))["hashes"]
    control = tune.measure_cell(
        3, 512, round_name="control", reps=3, reference_hashes=hashes,
        output=OUTPUT, ctx_size=53248,
    )
    challenger = None
    if control["feasible"]:
        challenger = tune.measure_cell(
            4, 512, round_name="challenger", reps=3, reference_hashes=hashes,
            output=OUTPUT, ctx_size=53248,
        )
    decision = (
        tune.compare_pair(control, challenger)
        if challenger is not None
        else {"confirmed": False, "reason": "control failed; challenger not launched"}
    )
    report = {
        "campaign": "HOST-WSL-TUNE runtime-53248",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "binary_revision": "71676e46c",
        "context": 53248,
        "control": control,
        "challenger": challenger,
        "decision": decision,
        "deploy_changed": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "control_feasible": control["feasible"],
        "challenger_feasible": challenger is not None and challenger["feasible"],
        "control_metrics": control.get("metrics"),
        "challenger_metrics": challenger.get("metrics") if challenger else None,
        "decision": decision,
    }, indent=2))
    return 0 if challenger is not None and challenger["feasible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
