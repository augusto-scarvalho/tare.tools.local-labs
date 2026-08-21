#!/usr/bin/env python3
"""LAB-REL-001: resumable low-duty-cycle endpoint reliability soak.

Rotates the qualified LAB-AGENT-001-v2 cases, periodically adds a longer known-answer
request, and records host/GPU telemetry around every operation.  JSONL is append-only so
an interrupted 24-hour run remains useful evidence; summary.json is atomically replaced.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from model_lifecycle.collectors.host import sample as host_sample  # noqa: E402
from tools.benchmarks.agent_suite_v2 import CASES, run_case  # noqa: E402
from tools.benchmarks.energy_phase_bench import gpu_sample  # noqa: E402
from tools.probes.cache_correctness_v2 import completion, normalize  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def health(base_url: str) -> dict:
    started = time.monotonic()
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=5) as response:
            payload = json.loads(response.read().decode())
        return {"ok": payload.get("status") == "ok",
                "latency_s": time.monotonic() - started, "response": payload}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "latency_s": time.monotonic() - started,
                "error": f"{type(exc).__name__}: {exc}"}


def telemetry() -> dict:
    gpu = gpu_sample()
    host = host_sample()
    return {"power_w": gpu["power_w"], "temperature_c": gpu["temp_c"],
            "gpu_util_pct": gpu["util_pct"], "vram_used_mb": gpu["vram_mb"],
            "vram_total_mb": host.vram_total_mb,
            "ram_available_mb": host.ram_available_mb}


def long_control(base_url: str, iteration: int) -> dict:
    marker = f"SOAK-{iteration:06d}"
    shared = (("Routine archival telemetry is nominal and contains no alert. " * 550) +
              f"The validation word for {marker} is SEQUOIA.")
    prompt = (shared + f"\nWhat is the validation word for {marker}? Return ONLY the exact word.")
    result = completion(base_url, prompt, False, n_predict=64)
    return {"case": "long_known_answer", "pass": "sequoia" in normalize(result["content"]),
            "content": result["content"], "timings": result["timings"],
            "error": result["error"]}


def write_atomic(path: pathlib.Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def summarize(records_path: pathlib.Path, started_at: str, duration_s: float,
              interval_s: float, status: str) -> dict:
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if records_path.exists() else []
    operations = [row for row in rows if row.get("kind") == "operation"]
    telemetry_rows = [row for row in rows if row.get("telemetry_after")]
    return {"campaign": "LAB-REL-001", "status": status, "started_at": started_at,
            "updated_at": utc_now(), "requested_duration_s": duration_s,
            "interval_s": interval_s, "iterations": len(operations),
            "operation_pass": sum(bool(row.get("result", {}).get("pass")) for row in operations),
            "operation_fail": sum(not bool(row.get("result", {}).get("pass")) for row in operations),
            "health_fail": sum(not bool(row.get("health", {}).get("ok")) for row in operations),
            "temperature_peak_c": max((row["telemetry_after"]["temperature_c"]
                                       for row in telemetry_rows), default=None),
            "vram_peak_mb": max((row["telemetry_after"]["vram_used_mb"]
                                 for row in telemetry_rows), default=None),
            "power_peak_w": max((row["telemetry_after"]["power_w"]
                                 for row in telemetry_rows), default=None),
            "records": str(records_path)}


def selfcheck() -> None:
    assert len(CASES) == 8
    assert normalize(" SEQUOIA. ") == "sequoia"
    print("reliability soak self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--duration-hours", type=float, default=24.0)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--long-every", type=int, default=10)
    parser.add_argument("--output-dir", type=pathlib.Path, required=False,
                        default=pathlib.Path("runs/reliability/LAB-REL-001-24h"))
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    if records_path.exists() and records_path.stat().st_size:
        raise FileExistsError(f"refusing to append a new run to existing {records_path}")
    (args.output_dir / "soak.pid").write_text(str(os.getpid()), encoding="ascii")
    started_at = utc_now()
    started = time.monotonic()
    duration_s = args.duration_hours * 3600
    iteration = 0
    status = "RUNNING"
    try:
        with records_path.open("a", encoding="utf-8", buffering=1) as handle:
            while time.monotonic() - started < duration_s:
                cycle_started = time.monotonic()
                before = telemetry()
                health_result = health(args.base_url)
                if args.long_every > 0 and iteration % args.long_every == args.long_every - 1:
                    result = long_control(args.base_url, iteration)
                else:
                    case = CASES[iteration % len(CASES)]
                    result = run_case(args.base_url, case, timeout_s=300.0)
                after = telemetry()
                row = {"kind": "operation", "iteration": iteration, "timestamp": utc_now(),
                       "elapsed_s": time.monotonic() - started, "health": health_result,
                       "telemetry_before": before, "telemetry_after": after, "result": result}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                iteration += 1
                write_atomic(args.output_dir / "summary.json", summarize(
                    records_path, started_at, duration_s, args.interval_seconds, status))
                remaining = args.interval_seconds - (time.monotonic() - cycle_started)
                if remaining > 0:
                    time.sleep(remaining)
        status = "COMPLETE"
    except KeyboardInterrupt:
        status = "INTERRUPTED"
    except Exception as exc:  # noqa: BLE001
        status = "CRASHED"
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": "fatal", "timestamp": utc_now(),
                                     "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        raise
    finally:
        write_atomic(args.output_dir / "summary.json", summarize(
            records_path, started_at, duration_s, args.interval_seconds, status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
