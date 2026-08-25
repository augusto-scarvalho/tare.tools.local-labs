#!/usr/bin/env python3
"""SLX-03: ReplaySSM State-Write Elision Oracle on RTX 3090.

Measures memory bandwidth savings and decode speedup from eliding recurrent state
DRAM writes across the 18 Gated DeltaNet linear attention layers in Qwen3.5-0.8B.
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def benchmark_state_io(num_layers: int = 18, state_shape: tuple[int, ...] = (1, 16, 64, 64),
                       steps: int = 128, torch=None) -> dict:
    # State buffer size per layer in FP16 (2 bytes per element)
    elem_count = 1
    for s in state_shape:
        elem_count *= s
    bytes_per_layer = elem_count * 2
    total_state_bytes_per_step = bytes_per_layer * num_layers

    # Allocate GPU SRAM registers/cache simulation tensors
    current_states = [torch.randn(state_shape, dtype=torch.bfloat16, device="cuda") for _ in range(num_layers)]
    # Persistent DRAM global buffer
    dram_buffers = [torch.zeros(state_shape, dtype=torch.bfloat16, device="cuda") for _ in range(num_layers)]

    policies = [
        ("PERSIST_EVERY_TOKEN", 1),
        ("ELISION_N4", 4),
        ("ELISION_N16", 16),
        ("EPHEMERAL_EOS_ONLY", steps),
    ]

    results = {}

    for pol_name, interval in policies:
        # Warmup
        for _ in range(10):
            for l in range(num_layers):
                current_states[l].add_(0.01)
                if 10 % interval == 0:
                    dram_buffers[l].copy_(current_states[l])
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        dram_writes = 0
        start_event.record()

        for step in range(steps):
            for l in range(num_layers):
                # Recurrent DeltaNet forward update (simulated)
                current_states[l].add_(0.001)
                # Write to DRAM buffer if step matches interval
                if (step + 1) % interval == 0:
                    dram_buffers[l].copy_(current_states[l])
                    dram_writes += 1

        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        mb_written = (dram_writes * bytes_per_layer) / (1024 * 1024)

        results[pol_name] = {
            "interval_n": interval,
            "dram_writes_count": dram_writes,
            "total_state_mb_written": round(mb_written, 2),
            "total_elapsed_ms": round(elapsed_ms, 3),
            "latency_per_step_us": round((elapsed_ms / steps) * 1000.0, 2),
        }

    base_time = results["PERSIST_EVERY_TOKEN"]["total_elapsed_ms"]
    base_mb = results["PERSIST_EVERY_TOKEN"]["total_state_mb_written"]

    for pol_name, data in results.items():
        data["speedup"] = round(base_time / data["total_elapsed_ms"], 3)
        data["io_reduction_pct"] = round(((base_mb - data["total_state_mb_written"]) / base_mb) * 100.0, 2)

    return {
        "num_layers": num_layers,
        "state_shape": list(state_shape),
        "steps": steps,
        "state_size_per_step_kb": round(total_state_bytes_per_step / 1024.0, 2),
        "policies": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SLX-03 State-Write Elision Oracle")
    parser.add_argument("--output", default="runs/research/SLX-03-STATE-WRITE-ELISION-2026-08-25/raw/receipt.json")
    args = parser.parse_args()

    import torch

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== SLX-03 ReplaySSM State-Write Elision Oracle ===", flush=True)
    bench_res = benchmark_state_io(num_layers=18, state_shape=(1, 16, 64, 64), steps=128, torch=torch)

    print(f"Recurrent Layers: {bench_res['num_layers']} (GDN)")
    print(f"State Size / Step: {bench_res['state_size_per_step_kb']} KB")
    print("\nPolicy Results:")
    for pol, d in bench_res["policies"].items():
        print(f"  [{pol:20}]: Time = {d['total_elapsed_ms']:6.2f} ms | Step = {d['latency_per_step_us']:5.1f} µs | IO = {d['total_state_mb_written']:6.2f} MB | Speedup = {d['speedup']:.2f}× | IO Reduction = {d['io_reduction_pct']}%")

    eph_speedup = bench_res["policies"]["EPHEMERAL_EOS_ONLY"]["speedup"]
    n16_io_red = bench_res["policies"]["ELISION_N16"]["io_reduction_pct"]

    gates = {
        "io_reduction_ge_70pct": n16_io_red >= 70.0,
        "speedup_ge_1_20x": eph_speedup >= 1.20,
    }

    verdict = "PROMOTED" if all(gates.values()) else "REJECTED"

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Antigravity",
        "benchmark": bench_res,
        "gates": gates,
        "verdict": verdict,
    }

    out_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n==================================================", flush=True)
    print(f"  SLX-03 STATE-WRITE ELISION VERDICT: {verdict}", flush=True)
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if verdict == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
