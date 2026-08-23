#!/usr/bin/env python3
"""LAB-ENERGY-002: counterbalanced RTX 3090 board-power Pareto curve."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess
import time
from datetime import datetime, timezone

from energy_phase_bench import run_rep


LIMIT_ORDERS = (
    (420, 294, 378, 336),
    (336, 378, 294, 420),
    (378, 420, 336, 294),
)
CELLS = (("short", 240), ("long", 1200))


def smi(*args: str) -> str:
    proc = subprocess.run(["nvidia-smi.exe", *args], capture_output=True,
                          text=True, timeout=15)
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi {' '.join(args)} failed rc={proc.returncode}: "
                           f"{proc.stderr.strip()[:500]}")
    return proc.stdout.strip()


def current_limit_w() -> float:
    raw = smi("--query-gpu=power.limit", "--format=csv,noheader,nounits")
    return float(raw.splitlines()[0].strip())


def set_limit_w(limit_w: int, settle_s: float) -> float:
    smi("--power-limit", str(limit_w))
    time.sleep(settle_s)
    observed = current_limit_w()
    if abs(observed - limit_w) > 1.0:
        raise RuntimeError(f"power-limit readback mismatch: requested={limit_w}, "
                           f"observed={observed}")
    return observed


def median_group(rows: list[dict], limit_w: int, cell: str) -> dict:
    group = [row for row in rows
             if row["power_limit_requested_w"] == limit_w and row["cell"] == cell]
    if not group:
        raise RuntimeError(f"missing group limit={limit_w} cell={cell}")
    metrics = {
        "prompt_tokens": statistics.median(row["prompt_tokens"] for row in group),
        "prompt_tps": statistics.median(
            row["prompt_tokens"] / row["prefill_ttft_s"] for row in group),
        "prefill_gross_j_per_prompt_token": statistics.median(
            row["prefill_gross_j_per_prompt_token"] for row in group),
        "decode_tps": statistics.median(
            row["decode_tokens_after_first"] / row["decode_s"] for row in group),
        "decode_gross_j_per_token": statistics.median(
            row["decode_gross_j_per_token"] for row in group),
        "power_mean_active_w": statistics.median(
            row["power_mean_active_w"] for row in group),
        "power_peak_active_w": max(row["power_peak_active_w"] for row in group),
        "temp_peak_c": max(row["temp_peak_c"] for row in group),
    }
    return {"power_limit_w": limit_w, "cell": cell, "reps": len(group), **metrics}


def dominates(left: dict, right: dict) -> bool:
    higher = ("prompt_tps", "decode_tps")
    lower = ("prefill_gross_j_per_prompt_token", "decode_gross_j_per_token")
    no_worse = (all(left[key] >= right[key] for key in higher)
                and all(left[key] <= right[key] for key in lower))
    strict = (any(left[key] > right[key] for key in higher)
              or any(left[key] < right[key] for key in lower))
    return no_worse and strict


def analyze(rows: list[dict], limits: list[int]) -> tuple[list[dict], dict]:
    aggregate = [median_group(rows, limit, cell)
                 for limit in limits for cell, _ in CELLS]
    for cell, _ in CELLS:
        group = [item for item in aggregate if item["cell"] == cell]
        for item in group:
            item["pareto_dominated_by_w"] = [
                candidate["power_limit_w"] for candidate in group
                if candidate is not item and dominates(candidate, item)
            ]
    long_rows = {item["power_limit_w"]: item for item in aggregate
                 if item["cell"] == "long"}
    baseline = long_rows[420]
    candidates = []
    for limit in sorted((limit for limit in limits if limit < 420)):
        item = long_rows[limit]
        prompt_retention = item["prompt_tps"] / baseline["prompt_tps"]
        decode_retention = item["decode_tps"] / baseline["decode_tps"]
        energy_no_worse = (
            item["prefill_gross_j_per_prompt_token"]
            <= baseline["prefill_gross_j_per_prompt_token"]
            and item["decode_gross_j_per_token"]
            <= baseline["decode_gross_j_per_token"]
        )
        qualifies = prompt_retention >= 0.95 and decode_retention >= 0.95 and energy_no_worse
        candidates.append({"power_limit_w": limit,
                           "prompt_throughput_retention": prompt_retention,
                           "decode_throughput_retention": decode_retention,
                           "energy_no_worse": energy_no_worse,
                           "qualifies": qualifies})
    qualified = [item["power_limit_w"] for item in candidates if item["qualifies"]]
    recommendation = min(qualified) if qualified else 420
    decision = {"rule": "lowest reduced limit with >=95% long-workload prompt and "
                         "decode throughput retention and no worse gross energy metrics",
                "comparisons": candidates,
                "recommended_power_limit_w": recommendation,
                "deployment_defaults_mutated": False}
    return aggregate, decision


def selfcheck() -> None:
    better = {"prompt_tps": 10, "decode_tps": 20,
              "prefill_gross_j_per_prompt_token": 1, "decode_gross_j_per_token": 2}
    worse = {"prompt_tps": 9, "decode_tps": 20,
             "prefill_gross_j_per_prompt_token": 1.1, "decode_gross_j_per_token": 2}
    assert dominates(better, worse)
    assert not dominates(worse, better)
    print("energy power curve self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path(
                            "runs/energy/LAB-ENERGY-002-POWER-CURVE-2026-08-22/results.json"))
    parser.add_argument("--sample-interval", type=float, default=0.08)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0

    initial_limit = current_limit_w()
    rows: list[dict] = []
    error: str | None = None
    restored_limit: float | None = None
    try:
        if abs(initial_limit - 420) > 1.0:
            raise RuntimeError(f"expected 420 W initial limit, observed {initial_limit}")
        for rep, order in enumerate(LIMIT_ORDERS):
            for order_index, limit_w in enumerate(order):
                observed_limit = set_limit_w(limit_w, args.settle_seconds)
                ordered_cells = CELLS if (rep + order_index) % 2 == 0 else tuple(reversed(CELLS))
                for cell, repeats in ordered_cells:
                    row = run_rep(args.base_url, cell, repeats, args.decode_tokens,
                                  args.sample_interval)
                    row.update({"rep": rep, "order_index": order_index,
                                "power_limit_requested_w": limit_w,
                                "power_limit_observed_w": observed_limit,
                                "undervolt": "none_stock_voltage_frequency_curve"})
                    rows.append(row)
                    print(f"{limit_w}W {cell} r{rep}: prompt={row['prompt_tokens']} "
                          f"prefill={row['prompt_tokens']/row['prefill_ttft_s']:.1f} tok/s "
                          f"{row['prefill_gross_j_per_prompt_token']:.3f} J/tok; "
                          f"decode={row['decode_tokens_after_first']/row['decode_s']:.1f} tok/s "
                          f"{row['decode_gross_j_per_token']:.3f} J/tok", flush=True)
    except Exception as exc:  # retain partial evidence before re-raising after restoration
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            restored_limit = set_limit_w(round(initial_limit), args.settle_seconds)
        except Exception as restore_exc:
            restore_error = f"{type(restore_exc).__name__}: {restore_exc}"
            error = f"{error}; restoration={restore_error}" if error else f"restoration={restore_error}"

        complete = len(rows) == len(LIMIT_ORDERS) * len(LIMIT_ORDERS[0]) * len(CELLS)
        valid_rows = all(row["boundaries_monotonic"] and not row["telemetry_errors"]
                         for row in rows)
        restored = restored_limit is not None and abs(restored_limit - initial_limit) <= 1.0
        aggregate, decision = analyze(rows, sorted(set(sum((list(x) for x in LIMIT_ORDERS), [])))) \
            if complete else ([], {})
        report = {
            "campaign": "LAB-ENERGY-002",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": args.base_url,
            "method": {"power_limits_w": [420, 378, 336, 294], "default_power_limit_w": 420,
                       "undervolt": "none_stock_voltage_frequency_curve",
                       "power_source": "nvidia-smi power.draw", "integration": "trapezoidal",
                       "sample_interval_s": args.sample_interval, "cache_prompt": False,
                       "decode_tokens": args.decode_tokens, "limit_orders": LIMIT_ORDERS},
            "initial_power_limit_w": initial_limit,
            "restored_power_limit_w": restored_limit,
            "restoration_verified": restored,
            "complete": complete,
            "qualified": complete and valid_rows and restored and error is None,
            "error": error,
            "aggregate": aggregate,
            "decision": decision,
            "runs": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
        print(json.dumps(aggregate, indent=2), flush=True)
        print(json.dumps(decision, indent=2), flush=True)
        print(f"restored={restored_limit}W qualified={report['qualified']} "
              f"evidence={args.output} sha256={digest}", flush=True)

    if error:
        raise RuntimeError(error)
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
