#!/usr/bin/env python3
"""HYPER-01: Hypernetworks for Contextual Adapters (Dynamic Capsule Generator).

Evaluates on-the-fly synthesis of LoRA adapters (A and B matrices) from task metadata
embeddings, measuring generation latency, weight reconstruction fidelity, and VRAM overhead.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run_hypernetwork_benchmark(d_in: int = 1024, d_out: int = 1024, r: int = 8,
                               z_dim: int = 64, num_tasks: int = 4, torch=None) -> dict:
    torch.manual_seed(20260824)

    # 1. Target Task Adapters (Ground Truth for 4 domains)
    target_tasks = []
    task_embeddings = []

    for t_idx in range(num_tasks):
        # Task embedding z
        z = torch.randn(1, z_dim, device="cuda")
        z = torch.nn.functional.normalize(z, dim=-1)
        task_embeddings.append(z)

        # Target LoRA weights
        target_A = torch.randn(d_in, r, device="cuda") * 0.05
        target_B = torch.randn(r, d_out, device="cuda") * 0.05
        target_tasks.append((target_A, target_B))

    # 2. HyperNetwork Generator Definition
    # Generates flattened A (1024*8) and B (8*1024) = 16,384 weights
    out_features = (d_in * r) + (r * d_out)

    class HyperLoRAMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(z_dim, 256),
                torch.nn.GELU(),
                torch.nn.Linear(256, 512),
                torch.nn.GELU(),
                torch.nn.Linear(512, out_features),
            )

        def forward(self, z):
            out = self.net(z)
            a_flat = out[:, :d_in * r].view(d_in, r)
            b_flat = out[:, d_in * r:].view(r, d_out)
            return a_flat, b_flat

    hypernet = HyperLoRAMLP().to("cuda")
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=5e-3)

    # 3. Fast Calibration Training (150 steps)
    t0 = time.monotonic()
    for step in range(150):
        t_idx = step % num_tasks
        z = task_embeddings[t_idx]
        target_A, target_B = target_tasks[t_idx]

        optimizer.zero_grad()
        gen_A, gen_B = hypernet(z)
        loss = torch.nn.functional.mse_loss(gen_A, target_A) + torch.nn.functional.mse_loss(gen_B, target_B)
        loss.backward()
        optimizer.step()

    train_time = time.monotonic() - t0

    # 4. Measure Inference / Synthesis Latency on RTX 3090
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    iters = 100
    start_ev.record()
    with torch.inference_mode():
        for _ in range(iters):
            for z in task_embeddings:
                _ = hypernet(z)
    end_ev.record()
    torch.cuda.synchronize()

    synthesis_latency_ms = (start_ev.elapsed_time(end_ev) / (iters * num_tasks))

    # 5. Evaluate Reconstruction Fidelity for each task
    task_similarities = []
    hypernet.eval()
    with torch.inference_mode():
        for t_idx in range(num_tasks):
            z = task_embeddings[t_idx]
            target_A, target_B = target_tasks[t_idx]
            gen_A, gen_B = hypernet(z)

            # Weight Cosine Similarity
            delta_target = torch.matmul(target_A, target_B).flatten()
            delta_gen = torch.matmul(gen_A, gen_B).flatten()
            cos_sim = torch.nn.functional.cosine_similarity(delta_gen, delta_target, dim=0).item()
            task_similarities.append(cos_sim)

    avg_sim = sum(task_similarities) / len(task_similarities)
    param_count = sum(p.numel() for p in hypernet.parameters())
    vram_mb = (param_count * 4) / (1024 * 1024)

    return {
        "num_tasks_modeled": num_tasks,
        "hypernet_parameters": param_count,
        "vram_overhead_mb": round(vram_mb, 2),
        "synthesis_latency_ms": round(synthesis_latency_ms, 3),
        "task_cosine_similarities": [round(s, 5) for s in task_similarities],
        "average_weight_cosine_sim": round(avg_sim, 5),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="HYPER-01 Hypernetworks for Capsules")
    parser.add_argument("--output", default="runs/research/HYPER-01-CAPSULES-2026-08-25/raw/receipt.json")
    args = parser.parse_args()

    import torch

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== HYPER-01 Hypernetworks for Capsules Probe on RTX 3090 ===", flush=True)
    res = run_hypernetwork_benchmark(d_in=1024, d_out=1024, r=8, z_dim=64, num_tasks=4, torch=torch)

    print(f"Tasks Modeled:             {res['num_tasks_modeled']}")
    print(f"HyperNetwork Parameters:   {res['hypernet_parameters']:,} ({res['vram_overhead_mb']} MB)")
    print(f"Synthesis Latency / Task:  {res['synthesis_latency_ms']} ms")
    print(f"Average Weight Cosine Sim: {res['average_weight_cosine_sim']}")

    gates = {
        "synthesis_latency_le_5ms": res["synthesis_latency_ms"] <= 5.0,
        "weight_cosine_sim_ge_0_950": res["average_weight_cosine_sim"] >= 0.950,
        "vram_overhead_le_20mb": res["vram_overhead_mb"] <= 20.0,
    }

    verdict = "PROMOTED" if all(gates.values()) else "REJECTED"

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Antigravity",
        "results": res,
        "gates": gates,
        "verdict": verdict,
    }

    out_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n==================================================", flush=True)
    print(f"  HYPER-01 HYPERNETWORK VERDICT: {verdict}", flush=True)
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if verdict == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
