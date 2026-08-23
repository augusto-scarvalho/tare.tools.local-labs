#!/usr/bin/env python3
"""Parse LAB-CLOSE-001 raw log and apply its frozen paired decision rule."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re
import statistics


BLOCK = re.compile(
    r"BEGIN rep=(?P<rep>\d+) arm=(?P<arm>on|off) mmap=(?P<mmap>[01]) utc=(?P<utc>\S+)\n"
    r"(?P<temp>[\d.]+), (?P<clock>[\d.]+), (?P<power>[\d.]+), (?P<limit>[\d.]+)\n"
    r"host_mem_available_bytes=(?P<available>\d+)\n(?P<body>.*?)"
    r"END rep=(?P=rep) arm=(?P=arm) mmap=(?P=mmap) rc=(?P<rc>\d+) utc=(?P<end>\S+)",
    re.DOTALL,
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_median_ci(values: list[float], seed: int = 20260822,
                        samples: int = 100_000) -> list[float]:
    rng = random.Random(seed)
    estimates = [statistics.median(rng.choices(values, k=len(values)))
                 for _ in range(samples)]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def field(body: str, label: str, cast=float):
    match = re.search(rf"^\s*{re.escape(label)}:\s*(.+)$", body, re.MULTILINE)
    if not match:
        raise ValueError(f"missing time field: {label}")
    return cast(match.group(1).strip())


def parse(raw: pathlib.Path) -> list[dict]:
    text = raw.read_text(encoding="utf-8")
    rows = []
    for match in BLOCK.finditer(text):
        body = match.group("body")
        json_line = next(line for line in body.splitlines() if line.startswith("{"))
        bench = json.loads(json_line)
        elapsed_text = field(body, "Elapsed (wall clock) time (h:mm:ss or m:ss)", str)
        parts = [float(piece) for piece in elapsed_text.split(":")]
        elapsed_s = parts[-1] + (parts[-2] * 60 if len(parts) >= 2 else 0)
        if len(parts) == 3:
            elapsed_s += parts[0] * 3600
        rows.append({
            "rep": int(match.group("rep")), "arm": match.group("arm"),
            "use_mmap": bool(int(match.group("mmap"))), "start_utc": match.group("utc"),
            "end_utc": match.group("end"), "return_code": int(match.group("rc")),
            "start_temp_c": float(match.group("temp")),
            "start_sm_clock_mhz": float(match.group("clock")),
            "start_power_w": float(match.group("power")),
            "power_limit_w": float(match.group("limit")),
            "host_mem_available_bytes": int(match.group("available")),
            "decode_tps": float(bench["avg_ts"]), "bench": bench,
            "process_elapsed_s": elapsed_s,
            "max_rss_kib": field(body, "Maximum resident set size (kbytes)", int),
            "major_page_faults": field(body, "Major (requiring I/O) page faults", int),
            "minor_page_faults": field(body, "Minor (reclaiming a frame) page faults", int),
            "exit_status": field(body, "Exit status", int),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    by_rep: dict[int, dict[str, dict]] = {}
    for row in rows:
        by_rep.setdefault(row["rep"], {})[row["arm"]] = row
    paired = []
    for rep in sorted(by_rep):
        on, off = by_rep[rep]["on"], by_rep[rep]["off"]
        paired.append({"rep": rep, "mmap_on_tps": on["decode_tps"],
                       "mmap_off_tps": off["decode_tps"],
                       "off_relative_delta_pct": (off["decode_tps"] / on["decode_tps"] - 1) * 100,
                       "off_process_elapsed_reduction_pct":
                           (1 - off["process_elapsed_s"] / on["process_elapsed_s"]) * 100})
    deltas = [item["off_relative_delta_pct"] for item in paired]
    warm_deltas = [item["off_relative_delta_pct"] for item in paired if item["rep"] > 0]
    elapsed_reductions = [item["off_process_elapsed_reduction_pct"] for item in paired]
    arms = {}
    for arm in ("on", "off"):
        group = [row for row in rows if row["arm"] == arm]
        arms[arm] = {
            "n": len(group),
            "decode_tps_median": statistics.median(row["decode_tps"] for row in group),
            "decode_tps_min": min(row["decode_tps"] for row in group),
            "decode_tps_max": max(row["decode_tps"] for row in group),
            "process_elapsed_s_median": statistics.median(row["process_elapsed_s"] for row in group),
            "max_rss_kib_median": statistics.median(row["max_rss_kib"] for row in group),
            "major_page_faults_total": sum(row["major_page_faults"] for row in group),
            "minor_page_faults_median": statistics.median(row["minor_page_faults"] for row in group),
        }
    paired_median = statistics.median(deltas)
    paired_ci = bootstrap_median_ci(deltas)
    warm_median = statistics.median(warm_deltas)
    warm_ci = bootstrap_median_ci(warm_deltas, seed=20260823)
    elapsed_median = statistics.median(elapsed_reductions)
    elapsed_ci = bootstrap_median_ci(elapsed_reductions, seed=20260824)
    effect_real = abs(paired_median) > 2.3 and not (paired_ci[0] <= 0 <= paired_ci[1])
    operational_advantage = elapsed_median > 2.3 and elapsed_ci[0] > 0
    return {
        "arms": arms, "paired": paired,
        "paired_off_relative_delta_pct_median": paired_median,
        "paired_off_relative_delta_pct_bootstrap_95ci": paired_ci,
        "warm_cache_off_relative_delta_pct_median": warm_median,
        "warm_cache_off_relative_delta_pct_bootstrap_95ci": warm_ci,
        "off_process_elapsed_reduction_pct_median": elapsed_median,
        "off_process_elapsed_reduction_pct_bootstrap_95ci": elapsed_ci,
        "standing_noise_floor_pct": 2.3,
        "decode_classification": "REAL" if effect_real else "NOISE / HISTORICAL RESIDUAL CONFOUNDED",
        "fresh_process_operational_advantage": operational_advantage,
        "recommendation_for_tested_qwen36_ncmoe6":
            "no-mmap" if operational_advantage or (effect_real and paired_median > 0) else "mmap",
        "current_qwen38_service_mutation": "none; different model/placement",
    }


def selfcheck() -> None:
    assert percentile([1, 2, 3], 0.5) == 2
    assert bootstrap_median_ci([1, 1, 1], samples=100) == [1.0, 1.0]
    print("mmap A/B summarizer self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=pathlib.Path, nargs="?", default=pathlib.Path(
        "runs/close-outs/LAB-CLOSE-001-MMAP-2026-08-22/raw.log"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path(
        "runs/close-outs/LAB-CLOSE-001-MMAP-2026-08-22/summary.json"))
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    rows = parse(args.raw)
    valid = (len(rows) == 12 and all(row["return_code"] == row["exit_status"] == 0 for row in rows)
             and all(abs(row["power_limit_w"] - 420) <= 1 for row in rows)
             and {(row["bench"]["build_commit"], row["bench"]["n_cpu_moe"],
                   row["bench"]["n_depth"]) for row in rows} == {("5e7f6271c", 6, 8192)})
    report = {"campaign": "LAB-CLOSE-001", "qualified": valid,
              "raw_log_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
              "summary": summarize(rows) if valid else {}, "runs": rows}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"qualified={valid} evidence={args.output}")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
