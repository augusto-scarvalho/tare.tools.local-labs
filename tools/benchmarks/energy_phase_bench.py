#!/usr/bin/env python3
"""LAB-ENERGY-001: phase-aligned GPU energy for prefill/TTFT and decode.

Power is sampled from the physical GPU while a streaming completion establishes two
monotonic boundaries: request start -> first token (prefill/TTFT), and first token ->
final event (decode).  Results retain gross device energy and an idle-baseline-subtracted
diagnostic; promotion decisions should use gross energy unless the protocol is explicitly
changed.  Unique prompts plus cache_prompt=false prevent prompt-cache contamination.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone


def gpu_sample() -> dict:
    command = ["nvidia-smi.exe",
               "--query-gpu=power.draw,utilization.gpu,temperature.gpu,memory.used",
               "--format=csv,noheader,nounits"]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=5)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"nvidia-smi failed rc={proc.returncode}: {proc.stderr[:200]}")
    power, util, temp, memory = [value.strip() for value in proc.stdout.splitlines()[0].split(",")]
    return {"t": time.monotonic(), "power_w": float(power), "util_pct": float(util),
            "temp_c": float(temp), "vram_mb": float(memory)}


class Sampler:
    def __init__(self, interval_s: float = 0.08):
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.samples.append(gpu_sample())
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"{type(exc).__name__}: {exc}")
            self._stop.wait(max(0.0, self.interval_s - (time.monotonic() - started)))

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=10)


def energy_j(samples: list[dict], start: float, end: float) -> float:
    """Trapezoidal integration with nearest measured power at exact boundaries."""
    if end <= start or not samples:
        return 0.0
    ordered = sorted(samples, key=lambda item: item["t"])

    def interpolated_power(t: float) -> float:
        if t <= ordered[0]["t"]:
            return ordered[0]["power_w"]
        if t >= ordered[-1]["t"]:
            return ordered[-1]["power_w"]
        for left, right in zip(ordered, ordered[1:]):
            if left["t"] <= t <= right["t"]:
                fraction = (t - left["t"]) / (right["t"] - left["t"])
                return left["power_w"] + fraction * (right["power_w"] - left["power_w"])
        raise AssertionError("interpolation interval not found")

    points = [{"t": start, "power_w": interpolated_power(start)}]
    points.extend(item for item in ordered if start < item["t"] < end)
    points.append({"t": end, "power_w": interpolated_power(end)})
    return sum((right["t"] - left["t"]) * (left["power_w"] + right["power_w"]) / 2
               for left, right in zip(points, points[1:]))


def streaming_completion(base_url: str, prompt: str, n_predict: int) -> dict:
    payload = {"prompt": prompt, "n_predict": n_predict, "ignore_eos": True,
               "temperature": 0.0, "top_k": 1, "seed": 0, "stream": True,
               "cache_prompt": False}
    request = urllib.request.Request(f"{base_url.rstrip('/')}/completion",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    request_start = time.monotonic()
    first_token = None
    final = None
    chunks = 0
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            now = time.monotonic()
            if first_token is None and not event.get("stop"):
                first_token = now
            if event.get("tokens"):
                chunks += len(event["tokens"])
            if event.get("stop"):
                final = event
    request_end = time.monotonic()
    if first_token is None or final is None:
        raise RuntimeError("stream did not expose first-token and final boundaries")
    return {"request_start": request_start, "first_token": first_token,
            "request_end": request_end, "final": final, "stream_tokens_seen": chunks}


def run_rep(base_url: str, cell: str, repeats: int, n_predict: int, interval_s: float) -> dict:
    nonce = uuid.uuid4().hex
    filler = "A routine archival sentence contains no special instruction or secret. "
    prompt = (f"ENERGY-{nonce} {filler * repeats}\n"
              "Continue by emitting the word measurement repeatedly until stopped.")
    with Sampler(interval_s) as sampler:
        time.sleep(1.5)
        measurement = streaming_completion(base_url, prompt, n_predict)
        time.sleep(0.3)
    samples = sampler.samples
    start, first, end = (measurement["request_start"], measurement["first_token"],
                         measurement["request_end"])
    baseline_values = [s["power_w"] for s in samples if start - 1.25 <= s["t"] < start]
    if len(baseline_values) < 2:
        raise RuntimeError("insufficient idle baseline samples")
    baseline_w = statistics.mean(baseline_values)
    prefill_j = energy_j(samples, start, first)
    decode_j = energy_j(samples, first, end)
    final = measurement["final"]
    prompt_tokens = int(final.get("tokens_evaluated") or 0)
    predicted_tokens = int(final.get("tokens_predicted") or 0)
    decode_tokens = max(1, predicted_tokens - 1)
    prefill_s, decode_s = first - start, end - first
    active = [s for s in samples if start <= s["t"] <= end]
    return {"cell": cell, "prompt_repeats": repeats, "prompt_tokens": prompt_tokens,
            "predicted_tokens": predicted_tokens, "decode_tokens_after_first": decode_tokens,
            "prefill_ttft_s": prefill_s, "decode_s": decode_s,
            "baseline_power_w": baseline_w, "prefill_energy_gross_j": prefill_j,
            "decode_energy_gross_j": decode_j,
            "prefill_gross_j_per_prompt_token": prefill_j / max(1, prompt_tokens),
            "decode_gross_j_per_token": decode_j / decode_tokens,
            "prefill_incremental_j": max(0.0, prefill_j - baseline_w * prefill_s),
            "decode_incremental_j": max(0.0, decode_j - baseline_w * decode_s),
            "power_mean_active_w": statistics.mean(s["power_w"] for s in active),
            "power_peak_active_w": max(s["power_w"] for s in active),
            "temp_peak_c": max(s["temp_c"] for s in active),
            "vram_peak_mb": max(s["vram_mb"] for s in active),
            "telemetry_samples": len(samples), "telemetry_errors": sampler.errors,
            "server_timings": final.get("timings") or {},
            "boundaries_monotonic": start < first < end}


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for cell in sorted({row["cell"] for row in rows}):
        group = [row for row in rows if row["cell"] == cell]
        output.append({"cell": cell, "reps": len(group),
                       "prompt_tokens_median": statistics.median(r["prompt_tokens"] for r in group),
                       "prefill_ttft_s_median": statistics.median(r["prefill_ttft_s"] for r in group),
                       "prefill_gross_j_per_prompt_token_median": statistics.median(
                           r["prefill_gross_j_per_prompt_token"] for r in group),
                       "decode_tps_median": statistics.median(
                           r["decode_tokens_after_first"] / r["decode_s"] for r in group),
                       "decode_gross_j_per_token_median": statistics.median(
                           r["decode_gross_j_per_token"] for r in group),
                       "power_peak_w": max(r["power_peak_active_w"] for r in group),
                       "temp_peak_c": max(r["temp_peak_c"] for r in group)})
    return output


def selfcheck() -> None:
    samples = [{"t": 0.0, "power_w": 10.0}, {"t": 1.0, "power_w": 20.0},
               {"t": 2.0, "power_w": 30.0}]
    assert abs(energy_j(samples, 0.0, 2.0) - 40.0) < 1e-9
    assert abs(energy_j(samples, 0.5, 1.5) - 20.0) < 1e-9
    print("energy phase bench self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--sample-interval", type=float, default=0.08)
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("runs/energy/LAB-ENERGY-001/results.json"))
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    cells = [("short", 240), ("long", 1200)]
    rows = []
    for rep in range(args.reps):
        ordered_cells = cells if rep % 2 == 0 else list(reversed(cells))
        for cell, repeats in ordered_cells:
            row = run_rep(args.base_url, cell, repeats, args.decode_tokens,
                          args.sample_interval)
            row["rep"] = rep
            rows.append(row)
            print(f"{cell} r{rep}: prompt={row['prompt_tokens']} ttft={row['prefill_ttft_s']:.2f}s "
                  f"prefill={row['prefill_gross_j_per_prompt_token']:.3f} J/tok "
                  f"decode={row['decode_gross_j_per_token']:.3f} J/tok", flush=True)
    report = {"campaign": "LAB-ENERGY-001", "timestamp": datetime.now(timezone.utc).isoformat(),
              "endpoint": args.base_url, "method": {
                  "power_source": "nvidia-smi power.draw", "integration": "trapezoidal",
                  "prefill_window": "request_start_to_first_stream_token",
                  "decode_window": "first_stream_token_to_final_event",
                  "primary_energy": "gross GPU energy", "cache_prompt": False},
              "qualified": all(r["boundaries_monotonic"] and not r["telemetry_errors"] for r in rows),
              "aggregate": aggregate(rows), "runs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"evidence: {args.output}")
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
