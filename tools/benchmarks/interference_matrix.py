#!/usr/bin/env python3
"""LAB-OPS-002 controlled interference matrix around the canonical endpoint."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import queue
import statistics
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone

from energy_phase_bench import run_rep


ROOT = pathlib.Path(__file__).resolve().parents[2]
WSL_SCRIPT = "/mnt/c/projects/tare.tools.local-labs/tools/contenders/interference_load.py"
CONDITIONS = ("baseline", "cpu", "ram", "disk", "gpu")
ORDERS = (
    ("baseline", "cpu", "ram", "disk", "gpu"),
    ("gpu", "disk", "ram", "cpu", "baseline"),
    ("ram", "baseline", "gpu", "cpu", "disk"),
)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def host_available_bytes() -> int:
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "--",
                           "cat", "/proc/meminfo"],
                          capture_output=True, text=True, timeout=10,
                          creationflags=NO_WINDOW)
    for line in proc.stdout.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError(f"MemAvailable missing from /proc/meminfo: {proc.stdout[:200]!r}")


def gpu_free_mib() -> float:
    proc = subprocess.run(["nvidia-smi.exe", "--query-gpu=memory.free",
                           "--format=csv,noheader,nounits"], capture_output=True,
                          text=True, timeout=10, creationflags=NO_WINDOW)
    return float(proc.stdout.splitlines()[0].strip())


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def start_contender(kind: str) -> tuple[subprocess.Popen, str]:
    interpreter = ("/home/augus/sglang-venv/bin/python" if kind == "gpu"
                   else "/usr/bin/python3")
    proc = subprocess.Popen(
        ["wsl.exe", "-d", "Ubuntu-24.04", "--", interpreter, WSL_SCRIPT, kind],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        creationflags=NO_WINDOW,
    )
    output: queue.Queue[str] = queue.Queue()

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            output.put(line.rstrip())

    threading.Thread(target=reader, daemon=True).start()
    lines = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            while not output.empty():
                lines.append(output.get_nowait())
            raise RuntimeError(f"{kind} contender exited before READY rc={proc.returncode}: {lines}")
        try:
            line = output.get(timeout=0.25)
            lines.append(line)
            if line.startswith("READY "):
                return proc, "\n".join(lines)
        except queue.Empty:
            pass
    raise RuntimeError(f"{kind} contender readiness timeout: {lines}")


def stop_contender(proc: subprocess.Popen) -> str:
    try:
        assert proc.stdin is not None
        proc.stdin.write("STOP\n")
        proc.stdin.flush()
        proc.stdin.close()
        proc.wait(timeout=20)
    except (BrokenPipeError, subprocess.TimeoutExpired):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    tail = proc.stdout.read() if proc.stdout else ""
    if proc.returncode != 0:
        raise RuntimeError(f"contender cleanup rc={proc.returncode}: {tail[-500:]}")
    return tail


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for condition in CONDITIONS:
        group = [row for row in rows if row["condition"] == condition]
        output.append({
            "condition": condition, "reps": len(group),
            "prompt_tps_median": statistics.median(
                row["prompt_tokens"] / row["prefill_ttft_s"] for row in group),
            "decode_tps_median": statistics.median(
                row["decode_tokens_after_first"] / row["decode_s"] for row in group),
            "prefill_gross_j_per_prompt_token_median": statistics.median(
                row["prefill_gross_j_per_prompt_token"] for row in group),
            "decode_gross_j_per_token_median": statistics.median(
                row["decode_gross_j_per_token"] for row in group),
            "power_peak_w": max(row["power_peak_active_w"] for row in group),
            "temp_peak_c": max(row["temp_peak_c"] for row in group),
        })
    baseline = next(item for item in output if item["condition"] == "baseline")
    for item in output:
        item["prompt_throughput_degradation_pct"] = (
            1 - item["prompt_tps_median"] / baseline["prompt_tps_median"]) * 100
        item["decode_throughput_degradation_pct"] = (
            1 - item["decode_tps_median"] / baseline["decode_tps_median"]) * 100
        item["prefill_energy_increase_pct"] = (
            item["prefill_gross_j_per_prompt_token_median"]
            / baseline["prefill_gross_j_per_prompt_token_median"] - 1) * 100
        item["decode_energy_increase_pct"] = (
            item["decode_gross_j_per_token_median"]
            / baseline["decode_gross_j_per_token_median"] - 1) * 100
        item["material"] = item["condition"] != "baseline" and any(value > 10 for value in (
            item["prompt_throughput_degradation_pct"], item["decode_throughput_degradation_pct"],
            item["prefill_energy_increase_pct"], item["decode_energy_increase_pct"]))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "runs" / "ops" /
                        "LAB-OPS-002-INTERFERENCE-2026-08-22" / "results.json")
    args = parser.parse_args()
    if host_available_bytes() < 16 * 1024**3:
        raise RuntimeError("RAM preflight failed: less than 16 GiB available")
    if not healthy(f"{args.base_url}/health") or not healthy("http://127.0.0.1:8081/health"):
        raise RuntimeError("endpoint preflight failed")
    rows = []
    errors = []
    for rep, order in enumerate(ORDERS):
        for order_index, condition in enumerate(order):
            proc = None
            ready = "baseline"
            free_after_ready = gpu_free_mib()
            try:
                if condition != "baseline":
                    proc, ready = start_contender(condition)
                    free_after_ready = gpu_free_mib()
                    if condition == "gpu" and free_after_ready < 1024:
                        raise RuntimeError(f"GPU reserve gate failed: {free_after_ready} MiB")
                row = run_rep(args.base_url, "short", 240, 128, 0.08)
                row.update({"rep": rep, "order_index": order_index,
                            "condition": condition, "contender_ready": ready,
                            "gpu_free_after_ready_mib": free_after_ready})
                rows.append(row)
                print(f"{condition} r{rep}: prompt={row['prompt_tokens']/row['prefill_ttft_s']:.1f} "
                      f"tok/s decode={row['decode_tokens_after_first']/row['decode_s']:.1f} tok/s "
                      f"free={free_after_ready:.0f} MiB", flush=True)
            except Exception as exc:  # retain partial receipts and ensure cleanup
                errors.append(f"{condition} r{rep}: {type(exc).__name__}: {exc}")
            finally:
                if proc is not None:
                    try:
                        stop_contender(proc)
                    except Exception as exc:
                        errors.append(f"{condition} r{rep} cleanup: {type(exc).__name__}: {exc}")
                time.sleep(1.0)
            if errors:
                break
        if errors:
            break
    report = {"campaign": "LAB-OPS-002", "timestamp": datetime.now(timezone.utc).isoformat(),
              "method": {"conditions": CONDITIONS, "orders": ORDERS, "reps": 3,
                         "cpu_workers": 12, "ram_gib": 8, "gpu_matmul": "2048x2048 fp16",
                         "disk": "read-only direct I/O", "material_threshold_pct": 10},
              "qualified": len(rows) == 15 and not errors
                           and all(not row["telemetry_errors"] for row in rows),
              "errors": errors, "aggregate": aggregate(rows) if len(rows) == 15 else [],
              "runs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2), flush=True)
    print(f"qualified={report['qualified']} evidence={args.output}", flush=True)
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
