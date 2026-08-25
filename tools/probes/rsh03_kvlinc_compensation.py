#!/usr/bin/env python3
"""RSH-03: KVLinC Residual Compensation Probe on RTX 3090.

Evaluates low-rank residual error compensation (U @ V^T, rank=4) on INT4 quantized
linear layers to restore activation fidelity and suppress quantization noise.
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


def run_kvlinc_benchmark(dim_m: int = 1024, dim_n: int = 1024, rank: int = 4,
                         num_samples: int = 500, torch=None) -> dict:
    torch.manual_seed(20260824)

    # 1. Weight matrix with realistic spectrum
    w_orig = torch.randn(dim_m, dim_n, dtype=torch.float32, device="cuda")
    # Add structured low-rank components (mimicking trained weight structure)
    w_orig += 2.0 * torch.matmul(torch.randn(dim_m, 16, device="cuda"), torch.randn(16, dim_n, device="cuda")) / 4.0

    # 2. Direct INT4 Quantization
    w_quant = quantize_block_symmetric(w_orig, bits=4, block_size=32, torch=torch)
    e_residual = w_orig - w_quant

    # 3. Fit KVLinC Rank-4 Residual Adapter via SVD
    U, S, Vh = torch.linalg.svd(e_residual, full_matrices=False)
    u_4 = U[:, :rank] * torch.sqrt(S[:rank]).unsqueeze(0)
    v_4 = Vh[:rank, :].t() * torch.sqrt(S[:rank]).unsqueeze(0)

    # Residual matrix: C = U_4 @ V_4^T
    w_comp = w_quant + torch.matmul(u_4, v_4.t())

    # 4. Evaluate Forward Activations on Test Inputs
    x_test = torch.randn(num_samples, dim_n, dtype=torch.float32, device="cuda")
    y_orig = torch.matmul(x_test, w_orig.t())
    y_quant = torch.matmul(x_test, w_quant.t())
    y_comp = torch.matmul(x_test, w_comp.t())

    mse_quant = torch.nn.functional.mse_loss(y_quant, y_orig).item()
    mse_comp = torch.nn.functional.mse_loss(y_comp, y_orig).item()

    cos_quant = torch.nn.functional.cosine_similarity(y_quant.flatten(), y_orig.flatten(), dim=0).item()
    cos_comp = torch.nn.functional.cosine_similarity(y_comp.flatten(), y_orig.flatten(), dim=0).item()

    mse_recovery_pct = ((mse_quant - mse_comp) / mse_quant) * 100.0 if mse_quant > 0 else 0.0
    param_overhead_pct = ((dim_m * rank + dim_n * rank) / (dim_m * dim_n)) * 100.0

    return {
        "matrix_shape": f"{dim_m}x{dim_n}",
        "compensation_rank": rank,
        "param_overhead_pct": round(param_overhead_pct, 2),
        "uncompensated_int4": {
            "output_mse": round(mse_quant, 6),
            "output_cosine_sim": round(cos_quant, 5),
        },
        "kvlinc_compensated_int4": {
            "output_mse": round(mse_comp, 6),
            "output_cosine_sim": round(cos_comp, 5),
            "mse_recovery_pct": round(mse_recovery_pct, 2),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="RSH-03 KVLinC Residual Compensation")
    parser.add_argument("--output", default="runs/research/RSH-03-KVLINC-COMPENSATION-2026-08-25/raw/receipt.json")
    args = parser.parse_args()

    import torch

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== RSH-03 KVLinC Residual Compensation Probe ===", flush=True)
    res = run_kvlinc_benchmark(dim_m=1024, dim_n=1024, rank=4, num_samples=500, torch=torch)

    uncomp = res["uncompensated_int4"]
    comp = res["kvlinc_compensated_int4"]

    print(f"Matrix: {res['matrix_shape']} | Compensation Rank: {res['compensation_rank']} (Overhead = {res['param_overhead_pct']}%)")
    print(f"Uncompensated INT4:  MSE = {uncomp['output_mse']:.6f} | Cosine Sim = {uncomp['output_cosine_sim']}")
    print(f"KVLinC Compensated:  MSE = {comp['output_mse']:.6f} | Cosine Sim = {comp['output_cosine_sim']} | Recovery = +{comp['mse_recovery_pct']}%")

    gates = {
        "mse_recovery_ge_50pct": comp["mse_recovery_pct"] >= 50.0,
        "output_cosine_ge_0_998": comp["output_cosine_sim"] >= 0.998,
        "param_overhead_le_1pct": res["param_overhead_pct"] <= 1.0,
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
    print(f"  RSH-03 KVLINC VERDICT: {verdict}", flush=True)
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if verdict == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
