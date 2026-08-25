#!/usr/bin/env python3
"""RSH-01: FibQuant Non-Linear Vector Quantization Codebook Simulation.

Evaluates non-uniform Fibonacci and logarithmic codebooks vs uniform linear grids
on heavy-tailed LLM activation distributions in 4-bit representation.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def build_codebooks(torch) -> dict[str, "torch.Tensor"]:
    # 16-level (4-bit) codebooks
    # Uniform Linear
    cb_uniform = torch.linspace(-1.0, 1.0, 16, device="cuda")

    # Fibonacci Geometric
    fib_seq = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0]
    fib_pos = [f / 21.0 for f in fib_seq]
    fib_neg = [-f for f in reversed(fib_pos[1:])]
    cb_fib = torch.tensor(sorted(fib_neg + fib_pos), dtype=torch.float32, device="cuda")

    # Logarithmic Power-of-2
    log_pos = [0.0] + [2.0 ** (-i) for i in reversed(range(7))]
    log_neg = [-x for x in reversed(log_pos[1:])]
    cb_log = torch.tensor(sorted(log_neg + log_pos), dtype=torch.float32, device="cuda")

    return {
        "UNIFORM_LINEAR_4BIT": cb_uniform,
        "FIBONACCI_NONLINEAR_4BIT": cb_fib,
        "LOGARITHMIC_4BIT": cb_log,
    }


def quantize_with_codebook(tensor: "torch.Tensor", codebook: "torch.Tensor",
                           block_size: int = 32, torch=None) -> "torch.Tensor":
    orig_shape = tensor.shape
    flat = tensor.view(-1, block_size)
    scales = torch.max(torch.abs(flat), dim=-1, keepdim=True)[0].clamp(min=1e-8)

    norm_x = flat / scales  # Shape: (B, 32)
    # Distance to each centroid: (B, 32, 16)
    dist = torch.abs(norm_x.unsqueeze(-1) - codebook.view(1, 1, -1))
    idx = torch.argmin(dist, dim=-1)  # Shape: (B, 32)

    dequant = codebook[idx] * scales
    return dequant.view(orig_shape)


def evaluate_fibquant(dim: int = 1024, torch=None) -> dict:
    torch.manual_seed(20260824)
    # Generate realistic Gaussian + Laplacian heavy-tailed LLM tensor
    x_gauss = torch.randn(dim, dim, dtype=torch.float32, device="cuda")
    x_laplace = torch.from_numpy(
        __import__("numpy").random.laplace(0.0, 0.5, (dim, dim))
    ).float().to("cuda")
    x = (0.7 * x_gauss + 0.3 * x_laplace)

    variance = torch.var(x).item()
    codebooks = build_codebooks(torch)
    results = {}

    for name, cb in codebooks.items():
        x_rec = quantize_with_codebook(x, cb, block_size=32, torch=torch)
        mse = torch.nn.functional.mse_loss(x_rec, x).item()
        sqnr_db = 10.0 * math.log10(variance / mse) if mse > 0 else 99.0
        cos_sim = torch.nn.functional.cosine_similarity(x.flatten(), x_rec.flatten(), dim=0).item()

        results[name] = {
            "codebook_levels": len(cb),
            "reconstruction_mse": round(mse, 6),
            "sqnr_db": round(sqnr_db, 2),
            "cosine_similarity": round(cos_sim, 5),
        }

    linear_mse = results["UNIFORM_LINEAR_4BIT"]["reconstruction_mse"]
    linear_sqnr = results["UNIFORM_LINEAR_4BIT"]["sqnr_db"]

    for name, d in results.items():
        d["mse_reduction_pct"] = round(((linear_mse - d["reconstruction_mse"]) / linear_mse) * 100.0, 2)
        d["sqnr_gain_db"] = round(d["sqnr_db"] - linear_sqnr, 2)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="RSH-01 FibQuant Codebook Simulation")
    parser.add_argument("--output", default="runs/research/RSH-01-FIBQUANT-2026-08-25/raw/receipt.json")
    args = parser.parse_args()

    import torch

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== RSH-01 FibQuant Non-Linear Codebook Simulation ===", flush=True)
    res = evaluate_fibquant(dim=1024, torch=torch)

    for name, d in res.items():
        print(f"  [{name:24}]: MSE = {d['reconstruction_mse']:.6f} (-{d['mse_reduction_pct']}%) | SQNR = {d['sqnr_db']:5.2f} dB (+{d['sqnr_gain_db']} dB) | Cosine = {d['cosine_similarity']}")

    fib_s = res["FIBONACCI_NONLINEAR_4BIT"]
    gates = {
        "fib_mse_reduction_ge_30pct": fib_s["mse_reduction_pct"] >= 30.0,
        "fib_sqnr_gain_ge_2_5db": fib_s["sqnr_gain_db"] >= 2.5,
        "fib_cosine_sim_ge_0_995": fib_s["cosine_similarity"] >= 0.995,
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
    print(f"  RSH-01 FIBQUANT VERDICT: {verdict}", flush=True)
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if verdict == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
