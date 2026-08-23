#!/usr/bin/env python3
"""LAB-OPS-003: startup-only context allocation envelope for the canonical model."""
from __future__ import annotations

import argparse
import json
import pathlib
from datetime import datetime, timezone

from optuna_runtime_tune import (ROOT, available_host_gib, finish_candidate, free_vram_mib,
                                 health, start_candidate)


CONTEXTS = (65536, 81920, 90112, 98304, 131072)
DEFAULT_OUTPUT = ROOT / "runs" / "ops" / "LAB-OPS-003-CONTEXT-VRAM-2026-08-22"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not health(8081) or health(8080) or health(8092):
        raise RuntimeError("preflight requires healthy 8081 and free 8080/8092")
    host_gib = available_host_gib()
    if host_gib < 16:
        raise RuntimeError(f"host RAM preflight failed: {host_gib:.1f} GiB")
    rows = []
    errors = []
    for ctx in CONTEXTS:
        proc = None
        row = {"ctx_size": ctx, "loaded": False, "errors": []}
        try:
            proc, load_s = start_candidate(3, 512, args.output / "logs" / f"ctx-{ctx}.log", ctx)
            row.update({"loaded": True, "load_s": load_s, "free_vram_mib": free_vram_mib()})
            row["passes_4gib_floor"] = row["free_vram_mib"] >= 4096
            if not health(8081):
                raise RuntimeError("embedding health lost")
            print(f"ctx={ctx}: free={row['free_vram_mib']:.0f} MiB "
                  f"pass={row['passes_4gib_floor']}", flush=True)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            row["errors"].append(message)
            errors.append(f"ctx={ctx}: {message}")
        finally:
            if proc is not None:
                try:
                    finish_candidate(proc)
                except Exception as exc:
                    message = f"cleanup {type(exc).__name__}: {exc}"
                    row["errors"].append(message)
                    errors.append(f"ctx={ctx}: {message}")
        rows.append(row)
        if errors:
            break
    passing = [row["ctx_size"] for row in rows if row.get("passes_4gib_floor")]
    report = {
        "campaign": "LAB-OPS-003", "timestamp": datetime.now(timezone.utc).isoformat(),
        "qualified": len(rows) == len(CONTEXTS) and not errors,
        "method": {"contexts": CONTEXTS, "vram_free_floor_mib": 4096,
                   "draft_n": 3, "ubatch": 512, "embedding_live": True},
        "host_available_gib_preflight": host_gib, "rows": rows, "errors": errors,
        "largest_passing_ladder_context": max(passing) if passing else None,
        "deploy_changed": False,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"qualified": report["qualified"],
                      "largest_passing_ladder_context": report["largest_passing_ladder_context"],
                      "errors": errors}, indent=2), flush=True)
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
