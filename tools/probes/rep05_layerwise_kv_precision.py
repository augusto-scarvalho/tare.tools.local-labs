#!/usr/bin/env python3
"""REP-05: Layer-Wise Mixed Precision KV Cache Allocation on RTX 3090.

Evaluates asymmetric bit allocation across 24 transformer layers (FP16 on sensitive
early/late layers + INT4 on robust middle layers), measuring memory savings and output fidelity.
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


def quantize_block_symmetric(tensor: "torch.Tensor", bits: int = 4, block_size: int = 32, torch=None) -> "torch.Tensor":
    orig_shape = tensor.shape
    flat = tensor.view(-1, block_size)
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax

    scale = torch.max(torch.abs(flat), dim=-1, keepdim=True)[0] / qmax
    scale = torch.clamp(scale, min=1e-8)

    q = torch.clamp(torch.round(flat / scale), qmin, qmax)
    dequant = (q * scale).view(orig_shape)
    return dequant


def run_layerwise_experiment(L: int = 4096, num_layers: int = 24, num_heads: int = 16,
                             head_dim: int = 128, torch=None) -> dict:
    torch.manual_seed(20260824)

    # Memory per layer calculation
    # FP16 layer: 2 * num_heads * L * head_dim * 2 bytes = 32 MB
    fp16_bytes_per_layer = 2 * num_heads * L * head_dim * 2
    # INT4 layer: 2 * num_heads * L * head_dim * 0.5 bytes + scales = 8.5 MB
    int4_bytes_per_layer = int(fp16_bytes_per_layer * 0.265)

    # Generate synthetic 24-layer sequence
    # Early layers (0..3) have extreme syntactic attention sinks / outliers
    # Middle layers (4..19) have diffuse attention
    # Late layers (20..23) have sharp categorical heads
    k_caches = []
    v_caches = []

    for l in range(num_layers):
        k = torch.randn(1, num_heads, L, head_dim, dtype=torch.float32, device="cuda")
        v = torch.randn(1, num_heads, L, head_dim, dtype=torch.float32, device="cuda")

        if l in (0, 1, 2, 3):
            # Sharp attention sinks
            k[:, :, 0:4, :] += 20.0
        elif l in (20, 21, 22, 23):
            # Late categorical spikes
            k[:, :, -16:, :] += 15.0

        k_caches.append(k)
        v_caches.append(v)

    # Query token embedding
    q_init = torch.randn(1, num_heads, 1, head_dim, dtype=torch.float32, device="cuda")

    policies = ["HOMOGENEOUS_FP16", "HOMOGENEOUS_INT4", "LAYERWISE_MIXED"]
    results = {}

    for pol in policies:
        h = q_init.clone()
        total_bytes = 0

        for l in range(num_layers):
            k = k_caches[l]
            v = v_caches[l]

            if pol == "HOMOGENEOUS_FP16":
                use_fp16 = True
            elif pol == "HOMOGENEOUS_INT4":
                use_fp16 = False
            elif pol == "LAYERWISE_MIXED":
                # Layers 0..3 and 20..23 are FP16; 4..19 are INT4
                use_fp16 = (l in (0, 1, 2, 3, 20, 21, 22, 23))

            if use_fp16:
                k_eval = k
                v_eval = v
                total_bytes += fp16_bytes_per_layer
            else:
                k_eval = quantize_block_symmetric(k, bits=4, block_size=32, torch=torch)
                v_eval = quantize_block_symmetric(v, bits=4, block_size=32, torch=torch)
                total_bytes += int4_bytes_per_layer

            # Attention step: h_next = h + Attn(h, k_eval, v_eval)
            scores = torch.matmul(h, k_eval.transpose(-1, -2)) / math.sqrt(head_dim)
            probs = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(probs, v_eval)
            h = h + attn_out

        results[pol] = {
            "final_embedding": h.squeeze(),
            "total_memory_mb": round(total_bytes / (1024 * 1024), 2),
        }

    ref_embed = results["HOMOGENEOUS_FP16"]["final_embedding"]
    fp16_mem = results["HOMOGENEOUS_FP16"]["total_memory_mb"]

    summary = {}
    for pol, d in results.items():
        cos_sim = torch.nn.functional.cosine_similarity(d["final_embedding"], ref_embed, dim=-1).mean().item()
        mse = torch.nn.functional.mse_loss(d["final_embedding"], ref_embed).item()
        savings_pct = ((fp16_mem - d["total_memory_mb"]) / fp16_mem) * 100.0

        summary[pol] = {
            "kv_cache_mb": d["total_memory_mb"],
            "memory_savings_pct": round(savings_pct, 2),
            "end_to_end_cosine_sim": round(cos_sim, 5),
            "reconstruction_mse": round(mse, 6),
        }

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="REP-05 Layer-Wise Mixed Precision KV")
    parser.add_argument("--output", default="runs/research/REP-05-LAYERWISE-PRECISION-2026-08-25/raw/receipt.json")
    args = parser.parse_args()

    import torch

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== REP-05 Layer-Wise Mixed Precision KV Cache Probe ===", flush=True)
    res = run_layerwise_experiment(L=4096, num_layers=24, torch=torch)

    print("\nLayer-Wise Precision Policy Comparison (24 Layers, L=4096):")
    for name, d in res.items():
        print(f"  [{name:20}]: Memory = {d['kv_cache_mb']:6.1f} MB (-{d['memory_savings_pct']}%) | Cosine Sim = {d['end_to_end_cosine_sim']} | MSE = {d['reconstruction_mse']:.6f}")

    mixed_s = res["LAYERWISE_MIXED"]
    int4_s = res["HOMOGENEOUS_INT4"]

    gates = {
        "mixed_savings_ge_45pct": mixed_s["memory_savings_pct"] >= 45.0,
        "mixed_cosine_sim_ge_0_990": mixed_s["end_to_end_cosine_sim"] >= 0.990,
        "mixed_beats_homogeneous_int4": (mixed_s["end_to_end_cosine_sim"] - int4_s["end_to_end_cosine_sim"]) >= 0.05,
    }

    verdict = "PROMOTED" if all(gates.values()) else "REJECTED"

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Antigravity",
        "context_length": 4096,
        "total_layers": 24,
        "results": res,
        "gates": gates,
        "verdict": verdict,
    }

    out_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n==================================================", flush=True)
    print(f"  REP-05 LAYER-WISE PRECISION VERDICT: {verdict}", flush=True)
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if verdict == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
