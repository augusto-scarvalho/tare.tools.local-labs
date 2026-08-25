#!/usr/bin/env python3
"""SLX-09: 2:4 Structured Sparsity Oracle on RTX 3090.

Evaluates 2:4 Ampere semi-structured pruning (Magnitude vs Wanda activation-weighted)
on Qwen3.5-0.8B Base, measuring logits cosine similarity, MSE, and structural conformity.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (  # noqa: E402
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
)


def enforce_2_to_4_sparsity(matrix: "torch.Tensor", scores: "torch.Tensor") -> "torch.Tensor":
    """Applies strict 2:4 pruning along columns (input dimension) guided by scores tensor."""
    import torch
    rows, cols = matrix.shape
    pad_cols = (4 - (cols % 4)) % 4
    if pad_cols > 0:
        matrix = torch.nn.functional.pad(matrix, (0, pad_cols))
        scores = torch.nn.functional.pad(scores, (0, pad_cols))

    orig_cols = cols
    cols = matrix.shape[1]

    # Reshape to (-1, 4)
    reshaped_w = matrix.view(-1, 4)
    reshaped_s = scores.view(-1, 4)

    # Find top 2 indices in each group of 4
    _, top_indices = torch.topk(reshaped_s, 2, dim=-1)
    mask = torch.zeros_like(reshaped_w, dtype=torch.bool)
    mask.scatter_(1, top_indices, True)

    sparse_w = reshaped_w * mask
    sparse_w = sparse_w.view(rows, cols)

    if pad_cols > 0:
        sparse_w = sparse_w[:, :orig_cols]

    return sparse_w


def prune_model(model, method: str, activation_norms: dict[str, "torch.Tensor"] | None, torch):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(p in name for p in ("proj", "dense")):
            w = module.weight.data
            if method == "STRUCTURED_2_4_MAGNITUDE":
                scores = torch.abs(w)
                module.weight.data = enforce_2_to_4_sparsity(w, scores)
            elif method == "STRUCTURED_2_4_WANDA":
                act_norm = activation_norms.get(name) if activation_norms else None
                if act_norm is not None:
                    scores = torch.abs(w) * act_norm.unsqueeze(0)
                else:
                    scores = torch.abs(w)
                module.weight.data = enforce_2_to_4_sparsity(w, scores)


def verify_2_to_4_conformity(model, torch) -> tuple[bool, float]:
    total_quartets = 0
    valid_quartets = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(p in name for p in ("proj", "dense")):
            w = module.weight.data
            cols = (w.shape[1] // 4) * 4
            reshaped = w[:, :cols].reshape(-1, 4)
            nonzeros = torch.count_nonzero(reshaped, dim=-1)
            total_quartets += reshaped.shape[0]
            valid_quartets += (nonzeros == 2).sum().item()

    conformity = valid_quartets / total_quartets if total_quartets > 0 else 1.0
    return conformity >= 0.9999, conformity


def main() -> int:
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_monotonic = time.monotonic()
    parser = argparse.ArgumentParser(description="SLX-09 2:4 Structured Sparsity Oracle")
    parser.add_argument("--model-path", default="/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe")
    parser.add_argument("--model-revision", default="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
    parser.add_argument("--output", default="runs/research/SLX-09B-SPARSITY-24-2026-08-25/raw/receipt.json")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== SLX-09 2:4 Structured Sparsity Oracle ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 1. Calibration Pass for Wanda (collecting activation L2 norms)
    calib_text = "Empirical verification requires measuring hardware-level speedup and weight sparsity. " * 30
    calib_tokens = tokenizer(calib_text, return_tensors="pt").input_ids.to("cuda")

    print(f"Loading dense baseline model from {args.model_path}...", flush=True)
    dense_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map={"": "cuda"},
        attn_implementation="sdpa")
    dense_model.eval()

    # Capture activation norms
    activation_norms = {}
    hooks = []

    def make_hook(mod_name):
        def hook_fn(module, inp, out):
            # inp[0] is (batch, seq, in_features)
            x = inp[0].detach()
            norm = torch.norm(x, p=2, dim=(0, 1))
            if mod_name not in activation_norms:
                activation_norms[mod_name] = norm
            else:
                activation_norms[mod_name] += norm
        return hook_fn

    for name, module in dense_model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(p in name for p in ("proj", "dense")):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.inference_mode():
        dense_out = dense_model(input_ids=calib_tokens)
        dense_logits = dense_out.logits[:, -1, :]

    for h in hooks:
        h.remove()

    print(f"Captured activation norms for {len(activation_norms)} linear modules.", flush=True)

    # 2. Evaluate Magnitude 2:4 Pruning
    print("\n--- Evaluating Magnitude 2:4 Pruning ---", flush=True)
    mag_model = copy.deepcopy(dense_model)
    prune_model(mag_model, "STRUCTURED_2_4_MAGNITUDE", None, torch)
    mag_valid, mag_conf = verify_2_to_4_conformity(mag_model, torch)

    with torch.inference_mode():
        mag_out = mag_model(input_ids=calib_tokens)
        mag_logits = mag_out.logits[:, -1, :]

    mag_cos = torch.nn.functional.cosine_similarity(mag_logits, dense_logits, dim=-1).item()
    mag_mse = torch.nn.functional.mse_loss(mag_logits, dense_logits).item()
    print(f"  Magnitude 2:4: Cosine Sim = {mag_cos:.5f} | Logits MSE = {mag_mse:.5f} | Conformity = {mag_conf * 100:.1f}%")

    del mag_model
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Evaluate Wanda 2:4 Pruning
    print("\n--- Evaluating Wanda (Activation-Weighted) 2:4 Pruning ---", flush=True)
    wanda_model = copy.deepcopy(dense_model)
    prune_model(wanda_model, "STRUCTURED_2_4_WANDA", activation_norms, torch)
    wanda_valid, wanda_conf = verify_2_to_4_conformity(wanda_model, torch)

    with torch.inference_mode():
        wanda_out = wanda_model(input_ids=calib_tokens)
        wanda_logits = wanda_out.logits[:, -1, :]

    wanda_cos = torch.nn.functional.cosine_similarity(wanda_logits, dense_logits, dim=-1).item()
    wanda_mse = torch.nn.functional.mse_loss(wanda_logits, dense_logits).item()
    print(f"  Wanda 2:4:     Cosine Sim = {wanda_cos:.5f} | Logits MSE = {wanda_mse:.5f} | Conformity = {wanda_conf * 100:.1f}%")

    mse_gain_pct = ((mag_mse - wanda_mse) / mag_mse) * 100.0 if mag_mse > 0 else 0.0

    gates = {
        "wanda_cosine_sim_ge_0_90": wanda_cos >= 0.90,
        "structural_conformity_100pct": wanda_valid,
        "wanda_beats_magnitude_ge_20pct": mse_gain_pct >= 20.0,
    }

    verdict = "PROMOTED" if all(gates.values()) else "REJECTED"

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Codex",
        "model": args.model_path,
        "dense_baseline": {
            "parameters": sum(p.numel() for p in dense_model.parameters()),
            "linear_layers_pruned": len(activation_norms),
        },
        "results": {
            "magnitude_2_4": {
                "cosine_similarity": round(mag_cos, 5),
                "logits_mse": round(mag_mse, 5),
                "conformity_pct": round(mag_conf * 100.0, 2),
            },
            "wanda_2_4": {
                "cosine_similarity": round(wanda_cos, 5),
                "logits_mse": round(wanda_mse, 5),
                "conformity_pct": round(wanda_conf * 100.0, 2),
                "mse_reduction_over_magnitude_pct": round(mse_gain_pct, 2),
            },
        },
        "gates": gates,
        "verdict": verdict,
    }

    model_root = pathlib.Path(args.model_path)
    provenance = build_provenance(
        script_path=pathlib.Path(__file__),
        started_at_utc=started_at_utc,
        started_monotonic=started_monotonic,
        input_paths=[
            model_root / "config.json",
            model_root / "model.safetensors-00001-of-00001.safetensors",
            model_root / "tokenizer.json",
        ],
        packages=["torch", "transformers"],
        runtime={
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "model_revision": args.model_revision,
            "sparsity_representation": "dense_tensors_with_2_of_4_weights_zeroed",
            "throughput_measured": False,
            "calibration_corpus": "single_repeated_sentence",
        },
    )
    provenance_ok, provenance_errors = provenance_complete(provenance)
    final_payload["provenance"] = provenance
    final_payload["provenance_complete"] = provenance_ok
    final_payload["provenance_errors"] = provenance_errors
    if not provenance_ok:
        final_payload["verdict"] = "UNVERIFIED_PROVENANCE"
    final_payload["receipt_fingerprint"] = canonical_json_sha256(final_payload)

    out_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n==================================================", flush=True)
    print(f"  SLX-09 2:4 SPARSITY VERDICT: {verdict}", flush=True)
    print(f"  Wanda Cosine Sim:        {wanda_cos:.5f} (Gate >=0.90: {gates['wanda_cosine_sim_ge_0_90']})")
    print(f"  MSE Gain over Magnitude: {mse_gain_pct:.1f}% (Gate >=20%: {gates['wanda_beats_magnitude_ge_20pct']})")
    print(f"  Structural Conformity:   {wanda_conf * 100.0:.1f}%")
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if final_payload["verdict"] == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
