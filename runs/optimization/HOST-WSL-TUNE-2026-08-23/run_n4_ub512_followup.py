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
SOURCE = HERE / "runtime-57344" / "results.json"
OUTPUT = HERE / "runtime-57344-n4-ub512"


def main() -> int:
    if tune.health(8080) or tune.health(tune.PORT):
        raise SystemExit("follow-up requires free canonical and candidate ports")
    tune.SERVER_BIN = "/home/augus/src/slop.cpp-candidate-71676e46c/build/bin/llama-server"
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    control = source["control"]
    challenger = tune.measure_cell(
        4, 512, round_name="challenger", reps=3,
        reference_hashes=control["probe_hashes"], output=OUTPUT, ctx_size=57344,
    )
    decision = tune.compare_pair(control, challenger)
    report = {
        "campaign": "HOST-WSL-TUNE runtime-57344 n4-ub512 follow-up",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "binary_revision": "71676e46c",
        "context": 57344,
        "reused_control_receipt": str(SOURCE),
        "control": control,
        "challenger": challenger,
        "decision": decision,
        "deploy_changed": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "challenger_feasible": challenger["feasible"],
        "challenger_free_vram_mib": challenger.get("free_vram_after_load_mib"),
        "challenger_metrics": challenger.get("metrics"),
        "decision": decision,
    }, indent=2))
    return 0 if challenger["feasible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
