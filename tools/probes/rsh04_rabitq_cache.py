#!/usr/bin/env python3
"""RSH-04: RaBitQCache Sparse Retrieval Benchmark on RTX 3090.

Evaluates 1-bit rotated binary sketching for fast KV cache block indexing,
measuring Top-K block retrieval recall and DRAM bandwidth reduction.
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


def run_rabitq_benchmark(L: int = 4096, block_size: int = 32, num_heads: int = 16,
                         head_dim: int = 128, topk_blocks: int = 32,
                         num_queries: int = 100, torch=None) -> dict:
    torch.manual_seed(20260824)
    num_blocks = L // block_size  # 128 blocks

    # 1. Allocate realistic KV Cache with sharp attention blocks
    k_cache = torch.randn(1, num_heads, L, head_dim, dtype=torch.float32, device="cuda")
    # Insert concentrated semantic blocks
    hot_blocks = [10, 42, 75, 110]
    for b in hot_blocks:
        k_cache[:, :, b * block_size:(b + 1) * block_size, :] += 15.0

    # 2. Construct Orthogonal Rotation Matrix R (128x128)
    R_raw = torch.randn(head_dim, head_dim, device="cuda")
    Q, _ = torch.linalg.qr(R_raw)
    R = Q.to(torch.float32)

    # 3. Precompute Block Centroids and 1-Bit Binary Sketches
    # Block Key Centroids: (1, num_heads, num_blocks, head_dim)
    k_blocks = k_cache.view(1, num_heads, num_blocks, block_size, head_dim).mean(dim=3)
    # Rotated: (1, num_heads, num_blocks, head_dim)
    k_blocks_rot = torch.matmul(k_blocks, R)
    # Binary Sketch: +1 / -1
    k_sketches = torch.sign(k_blocks_rot)

    # 4. Generate Queries and Evaluate Retrieval Recall
    recalls = []
    latencies_us = []

    # Warmup
    q_dummy = torch.randn(1, num_heads, 1, head_dim, device="cuda")
    for _ in range(5):
        _ = torch.matmul(torch.sign(torch.matmul(q_dummy, R)), k_sketches.transpose(-1, -2))
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for q_idx in range(num_queries):
        if q_idx % 4 == 0:
            # Query targeting one of the hot blocks
            target_b = hot_blocks[(q_idx // 4) % len(hot_blocks)]
            q = k_cache[:, :, target_b * block_size:target_b * block_size + 1, :].clone()
        else:
            q = torch.randn(1, num_heads, 1, head_dim, device="cuda")

        # Ground Truth Exact Dense Top-K Blocks
        # Compute exact max attention score per block
        scores_all = torch.matmul(q, k_cache.transpose(-1, -2)).squeeze(2)  # (1, num_heads, L)
        scores_per_block = scores_all.view(1, num_heads, num_blocks, block_size).max(dim=-1)[0]  # (1, num_heads, num_blocks)
        avg_exact_scores = scores_per_block.mean(dim=1).squeeze(0)  # (num_blocks,)
        _, true_topk_indices = torch.topk(avg_exact_scores, topk_blocks)
        true_topk_set = set(true_topk_indices.cpu().tolist())

        # RaBitQ Binary Sketch Matching
        start_ev.record()
        q_rot = torch.matmul(q, R)
        q_sketch = torch.sign(q_rot)  # (1, num_heads, 1, head_dim)
        # Fast binary dot product: (1, num_heads, 1, num_blocks)
        sketch_scores = torch.matmul(q_sketch, k_sketches.transpose(-1, -2)).squeeze(2)
        avg_sketch_scores = sketch_scores.mean(dim=1).squeeze(0)
        _, pred_topk_indices = torch.topk(avg_sketch_scores, topk_blocks)
        end_ev.record()
        torch.cuda.synchronize()

        latencies_us.append(start_ev.elapsed_time(end_ev) * 1000.0)

        pred_topk_set = set(pred_topk_indices.cpu().tolist())
        overlap = len(true_topk_set & pred_topk_set)
        recall_pct = (overlap / topk_blocks) * 100.0
        recalls.append(recall_pct)

    mean_recall = sum(recalls) / len(recalls)
    mean_latency = sum(latencies_us) / len(latencies_us)
    dram_saved_pct = ((num_blocks - topk_blocks) / num_blocks) * 100.0

    return {
        "context_length": L,
        "total_blocks": num_blocks,
        "retained_topk_blocks": topk_blocks,
        "mean_topk_recall_pct": round(mean_recall, 2),
        "mean_filter_latency_us": round(mean_latency, 2),
        "dram_bandwidth_savings_pct": round(dram_saved_pct, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="RSH-04 RaBitQCache Benchmark")
    parser.add_argument("--output", default="runs/research/RSH-04-RABITQ-CACHE-2026-08-25/raw/receipt.json")
    args = parser.parse_args()

    import torch

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== RSH-04 RaBitQCache Sparse Retrieval Probe on RTX 3090 ===", flush=True)
    res = run_rabitq_benchmark(L=4096, block_size=32, num_heads=16, head_dim=128, topk_blocks=32, torch=torch)

    print(f"Context Length:        {res['context_length']} ({res['total_blocks']} blocks)")
    print(f"Top-K Blocks Kept:     {res['retained_topk_blocks']} (DRAM Savings = {res['dram_bandwidth_savings_pct']}%)")
    print(f"Mean Top-K Recall:     {res['mean_topk_recall_pct']}%")
    print(f"Filter Latency:        {res['mean_filter_latency_us']} µs")

    gates = {
        "recall_ge_90pct": res["mean_topk_recall_pct"] >= 90.0,
        "dram_savings_ge_70pct": res["dram_bandwidth_savings_pct"] >= 70.0,
        "filter_latency_le_50us": res["mean_filter_latency_us"] <= 50.0,
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
    print(f"  RSH-04 RABITQ CACHE VERDICT: {verdict}", flush=True)
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if verdict == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
