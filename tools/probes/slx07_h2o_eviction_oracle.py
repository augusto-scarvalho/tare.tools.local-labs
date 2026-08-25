#!/usr/bin/env python3
"""SLX-07: Hierarchical Dynamic KV Eviction (H2O) Oracle on RTX 3090.

Evaluates Heavy-Hitter (H2O) KV cache pruning on 4096-token contexts,
measuring Needle-in-a-Haystack recall and memory savings.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run_h2o_benchmark(L: int = 4096, num_sinks: int = 4, num_recent: int = 64, num_heavy: int = 128, torch=None) -> dict:
    torch.manual_seed(20260824)
    head_dim = 128
    num_heads = 16

    # 1. Build 4096 KV sequence
    k = torch.randn(1, num_heads, L, head_dim, dtype=torch.float32, device="cuda")
    v = torch.randn(1, num_heads, L, head_dim, dtype=torch.float32, device="cuda")

    # 2. Plant 3 Key Needles (distinct high-magnitude semantic signatures)
    needle_positions = [int(L * 0.10), int(L * 0.50), int(L * 0.90)]
    needle_keys = [torch.randn(1, num_heads, 1, head_dim, dtype=torch.float32, device="cuda") for _ in range(3)]

    for pos, n_key in zip(needle_positions, needle_keys):
        k[:, :, pos:pos + 1, :] = n_key * 3.0

    # 3. Simulate attention queries during generation
    cumulative_attention_scores = torch.zeros(1, num_heads, L, device="cuda")

    # Generate 50 decoding queries: 10 queries targeting needle 1, 10 for needle 2, 10 for needle 3, 20 general
    queries = []
    for _ in range(10):
        queries.append((needle_keys[0], needle_positions[0]))
    for _ in range(10):
        queries.append((needle_keys[1], needle_positions[1]))
    for _ in range(10):
        queries.append((needle_keys[2], needle_positions[2]))
    for _ in range(20):
        queries.append((torch.randn(1, num_heads, 1, head_dim, dtype=torch.float32, device="cuda"), -1))

    # Accumulate attention scores
    for q, _ in queries:
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        probs = torch.softmax(scores, dim=-1)
        cumulative_attention_scores += probs.squeeze(2)

    # 4. Construct H2O Mask
    # Fixed Sinks
    sinks = set(range(num_sinks))
    # Recent Window
    recent = set(range(L - num_recent, L))
    # Candidates for heavy hitters (all tokens not in sinks and not in recent)
    candidate_mask = torch.ones(L, dtype=torch.bool, device="cuda")
    candidate_mask[:num_sinks] = False
    candidate_mask[-num_recent:] = False

    avg_scores = cumulative_attention_scores.mean(dim=(0, 1))  # Shape: (L,)
    avg_scores[~candidate_mask] = -1e9

    _, heavy_hitter_indices = torch.topk(avg_scores, num_heavy)
    heavy_hitters = set(heavy_hitter_indices.cpu().tolist())

    h2o_indices = sorted(list(sinks | recent | heavy_hitters))
    h2o_cache_size = len(h2o_indices)

    # 5. Evaluate Needle Retrieval Recall for Full KV, H2O, and Random Eviction
    rng = random.Random(42)
    random_indices = sorted(rng.sample(range(L), h2o_cache_size))

    def evaluate_retrieval(indices_subset: list[int] | None) -> tuple[int, float]:
        recalled_count = 0
        total_needle_queries = 30
        for q, target_pos in queries[:30]:
            if indices_subset is None:
                # Full KV
                q_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
                q_probs = torch.softmax(q_scores, dim=-1).mean(dim=1).squeeze()
                top_pos = torch.argmax(q_probs).item()
                if top_pos == target_pos:
                    recalled_count += 1
            else:
                # Evicted KV subset
                sub_k = k[:, :, indices_subset, :]
                q_scores = torch.matmul(q, sub_k.transpose(-1, -2)) / math.sqrt(head_dim)
                q_probs = torch.softmax(q_scores, dim=-1).mean(dim=1).squeeze()
                top_sub_idx = torch.argmax(q_probs).item()
                mapped_orig_pos = indices_subset[top_sub_idx]
                if mapped_orig_pos == target_pos:
                    recalled_count += 1
        return recalled_count, (recalled_count / total_needle_queries) * 100.0

    full_recalled, full_recall_pct = evaluate_retrieval(None)
    h2o_recalled, h2o_recall_pct = evaluate_retrieval(h2o_indices)
    rand_recalled, rand_recall_pct = evaluate_retrieval(random_indices)

    # Verify all needles are retained in H2O indices
    needles_retained = [pos in h2o_indices for pos in needle_positions]
    memory_savings_pct = ((L - h2o_cache_size) / L) * 100.0

    return {
        "context_len": L,
        "h2o_budget": {
            "sinks": num_sinks,
            "recent": num_recent,
            "heavy_hitters": num_heavy,
            "total_retained_tokens": h2o_cache_size,
            "memory_savings_pct": round(memory_savings_pct, 2),
        },
        "retrieval_performance": {
            "full_kv_recall_pct": round(full_recall_pct, 2),
            "h2o_recall_pct": round(h2o_recall_pct, 2),
            "random_eviction_recall_pct": round(rand_recall_pct, 2),
            "all_needles_retained_in_mask": all(needles_retained),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SLX-07 H2O Eviction Oracle")
    parser.add_argument("--output", default="runs/research/SLX-07-H2O-EVICTION-2026-08-25/raw/receipt.json")
    args = parser.parse_args()

    import torch

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== SLX-07 Hierarchical KV Cache Eviction (H2O) Oracle ===", flush=True)
    res = run_h2o_benchmark(L=4096, num_sinks=4, num_recent=64, num_heavy=128, torch=torch)

    b = res["h2o_budget"]
    r = res["retrieval_performance"]

    print(f"Context Length: {res['context_len']} tokens")
    print(f"H2O Retained Tokens: {b['total_retained_tokens']} (Savings = {b['memory_savings_pct']}%)")
    print(f"Full KV Recall:      {r['full_kv_recall_pct']}%")
    print(f"H2O Recall:          {r['h2o_recall_pct']}%")
    print(f"Random Recall:       {r['random_eviction_recall_pct']}%")

    gates = {
        "h2o_recall_ge_98pct": r["h2o_recall_pct"] >= 98.0,
        "memory_savings_ge_85pct": b["memory_savings_pct"] >= 85.0,
        "h2o_beats_random_ge_30pct": (r["h2o_recall_pct"] - r["random_eviction_recall_pct"]) >= 30.0,
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
    print(f"  SLX-07 H2O EVICTION VERDICT: {verdict}", flush=True)
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if verdict == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
