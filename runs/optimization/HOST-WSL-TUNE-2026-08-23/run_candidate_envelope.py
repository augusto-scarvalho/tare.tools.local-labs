#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone


ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "benchmarks"))

import optuna_runtime_tune as tune  # noqa: E402


CONTEXTS = (49152, 57344, 61440, 65536, 73728, 81920)
OUTPUT = pathlib.Path(__file__).resolve().parent / "candidate-envelope"


def main() -> int:
    if tune.health(8080) or tune.health(tune.PORT):
        raise SystemExit("envelope requires free canonical and candidate ports")
    tune.SERVER_BIN = "/home/augus/src/slop.cpp-candidate-71676e46c/build/bin/llama-server"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    for ctx_size in CONTEXTS:
        proc = None
        row = {"ctx_size": ctx_size, "loaded": False, "errors": []}
        try:
            proc, load_s = tune.start_candidate(
                3, 512, OUTPUT / "logs" / f"ctx-{ctx_size}.log", ctx_size
            )
            free_mib = tune.free_vram_mib()
            row.update({
                "loaded": True,
                "load_s": load_s,
                "free_vram_mib": free_mib,
                "passes_4gib_floor": free_mib >= 4096,
            })
            print(f"ctx={ctx_size}: free={free_mib:.0f} MiB pass={free_mib >= 4096}", flush=True)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            row["errors"].append(message)
            errors.append(f"ctx={ctx_size}: {message}")
        finally:
            if proc is not None:
                try:
                    tune.finish_candidate(proc)
                except Exception as exc:
                    message = f"cleanup {type(exc).__name__}: {exc}"
                    row["errors"].append(message)
                    errors.append(f"ctx={ctx_size}: {message}")
        rows.append(row)
        if errors:
            break
    passing = [row["ctx_size"] for row in rows if row.get("passes_4gib_floor")]
    report = {
        "campaign": "HOST-WSL-TUNE candidate context envelope",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "binary_revision": "71676e46c",
        "contexts": CONTEXTS,
        "vram_free_floor_mib": 4096,
        "rows": rows,
        "errors": errors,
        "largest_passing_ladder_context": max(passing) if passing else None,
        "deploy_changed": False,
    }
    (OUTPUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "largest_passing_ladder_context": report["largest_passing_ladder_context"],
        "errors": errors,
    }, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
