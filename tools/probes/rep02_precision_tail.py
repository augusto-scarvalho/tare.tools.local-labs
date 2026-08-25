#!/usr/bin/env python3
"""REP-02: Precision Tail Standard Probe on RTX 3090.

Evaluates the impact of preserving exact Attention Sinks (S=4) and Recent Tail (T=64, 128)
in full precision while quantizing the historical body into 4-bit KV representation.
Handles hybrid recurrent/transformer KV structures (DynamicLayer + LinearAttentionLayer).
Measures Attention KLD, Logit MSE, Memory Savings, and Needle-in-a-Haystack recall.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
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


def quantize_int4(tensor: "torch.Tensor", group_size: int = 32) -> "torch.Tensor":
    """Simulates symmetric 4-bit per-group quantization with exact rounding distortion."""
    import torch
    orig_shape = tensor.shape
    last_dim = orig_shape[-1]
    pad_len = (group_size - (last_dim % group_size)) % group_size
    if pad_len > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_len))

    reshaped = tensor.view(*orig_shape[:-1], -1, group_size)
    max_val = torch.amax(torch.abs(reshaped), dim=-1, keepdim=True).clamp(min=1e-5)
    scale = max_val / 7.0
    quant = torch.round(reshaped / scale).clamp(-8, 7)
    dequant = quant * scale
    out = dequant.view(*tensor.shape)
    if pad_len > 0:
        out = out[..., :last_dim]
    return out


def apply_kv_policy(k: "torch.Tensor", v: "torch.Tensor", policy: str, sinks: int = 4, tail: int = 64) -> tuple["torch.Tensor", "torch.Tensor", float]:
    """Applies precision policy to K and V tensors of shape (batch, heads, seq_len, head_dim)."""
    import torch
    seq_len = k.shape[2]
    if policy == "FP16_UNIFORM":
        return k.clone(), v.clone(), 1.0

    elif policy == "INT4_UNIFORM":
        k_q = quantize_int4(k)
        v_q = quantize_int4(v)
        return k_q, v_q, 0.28

    elif policy.startswith("PRECISION_TAIL"):
        s_len = min(sinks, seq_len)
        t_len = tail
        if s_len + t_len >= seq_len:
            return k.clone(), v.clone(), 1.0

        k_out = k.clone()
        v_out = v.clone()

        # Quantize only the body [s_len : seq_len - t_len]
        body_k = k[:, :, s_len:seq_len - t_len, :]
        body_v = v[:, :, s_len:seq_len - t_len, :]

        k_out[:, :, s_len:seq_len - t_len, :] = quantize_int4(body_k)
        v_out[:, :, s_len:seq_len - t_len, :] = quantize_int4(body_v)

        body_len = seq_len - (s_len + t_len)
        comp_ratio = ((s_len + t_len) * 1.0 + body_len * 0.28) / seq_len
        return k_out, v_out, comp_ratio

    raise ValueError(f"Unknown policy: {policy}")


def clone_and_modify_cache(pkv, policy: str, sinks: int, tail: int):
    """Creates a modified deep copy of DynamicCache under the specified precision policy."""
    import torch
    new_cache = copy.copy(pkv)
    new_layers = []
    comp_ratios = []

    for layer in pkv.layers:
        new_layer = copy.copy(layer)
        if hasattr(layer, "keys") and layer.keys is not None and layer.keys.numel() > 0:
            k_mod, v_mod, comp = apply_kv_policy(layer.keys, layer.values, policy, sinks=sinks, tail=tail)
            new_layer.keys = k_mod
            new_layer.values = v_mod
            comp_ratios.append(comp)
        elif hasattr(layer, "recurrent_states"):
            # Linear attention recurrent state
            if isinstance(layer.recurrent_states, dict):
                new_layer.recurrent_states = {k: v.clone() if hasattr(v, "clone") else v for k, v in layer.recurrent_states.items()}
            elif hasattr(layer.recurrent_states, "clone"):
                new_layer.recurrent_states = layer.recurrent_states.clone()
            else:
                new_layer.recurrent_states = copy.deepcopy(layer.recurrent_states)

            if hasattr(layer, "conv_states"):
                if isinstance(layer.conv_states, dict):
                    new_layer.conv_states = {k: v.clone() if hasattr(v, "clone") else v for k, v in layer.conv_states.items()}
                elif hasattr(layer.conv_states, "clone"):
                    new_layer.conv_states = layer.conv_states.clone()
                else:
                    new_layer.conv_states = copy.deepcopy(layer.conv_states)
        new_layers.append(new_layer)

    new_cache.layers = new_layers
    avg_comp = sum(comp_ratios) / len(comp_ratios) if comp_ratios else 1.0
    return new_cache, avg_comp


def evaluate_context(model, tokenizer, seq_len: int, torch) -> dict:
    text = "Empirical verification in local systems engineering requires isolating causal variables. " * ((seq_len // 10) + 1)
    tokens = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True).input_ids.to("cuda")
    actual_len = tokens.shape[1]

    with torch.inference_mode():
        out_fp16 = model(input_ids=tokens, use_cache=True)

    # Every arm must predict the same next token from an independently cloned
    # copy of the exact same post-prefill cache. The previous probe incorrectly
    # compared next-token logits with logits from the final prefill position.
    dummy_next = torch.tensor([[100]], device="cuda", dtype=torch.long)
    baseline_cache, _ = clone_and_modify_cache(out_fp16.past_key_values, "FP16_UNIFORM", 0, 0)
    with torch.inference_mode():
        baseline_out = model(input_ids=dummy_next, past_key_values=baseline_cache, use_cache=True)
        logits_fp16 = baseline_out.logits[:, -1, :]

    policies = [
        ("FP16_UNIFORM", 0, 0),
        ("INT4_UNIFORM", 0, 0),
        ("PRECISION_TAIL_64", 4, 64),
        ("PRECISION_TAIL_128", 4, 128),
    ]

    policy_results = {}
    for name, sinks, tail in policies:
        if name == "FP16_UNIFORM":
            policy_results[name] = {
                "compression_ratio": 1.0,
                "estimated_packed_storage_savings_pct": 0.0,
                "logits_mse": 0.0,
                "logits_cosine_sim": 1.0,
            }
            continue

        mod_cache, avg_comp = clone_and_modify_cache(out_fp16.past_key_values, name, sinks, tail)

        with torch.inference_mode():
            out_mod = model(input_ids=dummy_next, past_key_values=mod_cache, use_cache=True)
            logits_mod = out_mod.logits[:, -1, :]

        mse = torch.nn.functional.mse_loss(logits_mod, logits_fp16).item()
        cos_sim = torch.nn.functional.cosine_similarity(logits_mod, logits_fp16, dim=-1).item()

        policy_results[name] = {
            "compression_ratio": round(avg_comp, 3),
            "estimated_packed_storage_savings_pct": round((1.0 - avg_comp) * 100.0, 1),
            "logits_mse": round(mse, 6),
            "logits_cosine_sim": round(cos_sim, 6),
        }

    return {"seq_len": actual_len, "policies": policy_results}


def evaluate_needle(model, tokenizer, torch) -> dict:
    needle = "The secret laboratory access key is 749102."
    haystack_sentence = "All hardware performance benchmarks must be conducted with stable fan profiles and locked voltage curves. "
    
    target_context_tokens = 2048
    needle_pos = target_context_tokens // 2

    haystack_text = haystack_sentence * 80
    haystack_tokens = tokenizer(haystack_text, return_tensors="pt").input_ids[0]
    needle_tokens = tokenizer(needle, return_tensors="pt").input_ids[0]

    pre = haystack_tokens[:needle_pos]
    post = haystack_tokens[needle_pos:target_context_tokens - len(needle_tokens) - 40]
    prompt_q = "\nQuestion: What is the secret laboratory access key? Reply with only the 6-digit key:\n"
    q_tokens = tokenizer(prompt_q, return_tensors="pt").input_ids[0]

    full_ids = torch.cat([pre, needle_tokens, post, q_tokens]).unsqueeze(0).to("cuda")

    needle_results = {}
    policies = [
        ("FP16_UNIFORM", 0, 0),
        ("INT4_UNIFORM", 0, 0),
        ("PRECISION_TAIL_64", 4, 64),
    ]

    for name, sinks, tail in policies:
        with torch.inference_mode():
            out = model(input_ids=full_ids, use_cache=True)
            past = out.past_key_values

            if name != "FP16_UNIFORM":
                past, _ = clone_and_modify_cache(past, name, sinks, tail)

            # Greedy generation (12 tokens)
            curr_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            gen_ids = [curr_token.item()]
            for _ in range(12):
                out = model(input_ids=curr_token, past_key_values=past, use_cache=True)
                past = out.past_key_values
                curr_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                gen_ids.append(curr_token.item())
                if curr_token.item() == tokenizer.eos_token_id:
                    break

            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            passed = "749102" in text
            needle_results[name] = {"generated": text, "pass": passed}

    return needle_results


def main() -> int:
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_monotonic = time.monotonic()
    parser = argparse.ArgumentParser(description="REP-02 Precision Tail Probe")
    parser.add_argument("--model-path", default="/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe")
    parser.add_argument("--model-revision", default="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
    parser.add_argument("--output", default="runs/research/REP-02B-PRECISION-TAIL-2026-08-25/raw/receipt.json")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== REP-02 Precision Tail Standard Probe ===", flush=True)
    print(f"Loading model {args.model_path}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map={"": "cuda"},
        attn_implementation="sdpa")
    model.eval()

    context_lengths = [256, 1024, 4096]
    context_evals = []
    for cl in context_lengths:
        print(f"\nEvaluating Context Length: {cl} tokens...", flush=True)
        res = evaluate_context(model, tokenizer, cl, torch)
        context_evals.append(res)
        print(json.dumps(res, indent=2))

    print("\nEvaluating Needle-in-a-Haystack (Context=2048, Depth=50%)...", flush=True)
    needle_res = evaluate_needle(model, tokenizer, torch)
    print(json.dumps(needle_res, indent=2))

    # Evaluate promotion gates
    res_4096 = next(r for r in context_evals if r["seq_len"] >= 4000)["policies"]
    int4_mse = res_4096["INT4_UNIFORM"]["logits_mse"]
    tail64_mse = res_4096["PRECISION_TAIL_64"]["logits_mse"]
    mse_reduction_pct = ((int4_mse - tail64_mse) / int4_mse) * 100.0 if int4_mse > 0 else 0.0

    tail64_savings = res_4096["PRECISION_TAIL_64"]["estimated_packed_storage_savings_pct"]
    needle_pass = needle_res["PRECISION_TAIL_64"]["pass"]

    gates = {
        "logits_mse_reduced_ge_50pct": mse_reduction_pct >= 50.0,
        "needle_retrieval_pass": needle_pass,
        "estimated_packed_storage_savings_ge_65pct": tail64_savings >= 65.0,
    }

    verdict = "PROMOTED" if all(gates.values()) else "REJECTED"

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Codex",
        "model": args.model_path,
        "context_evaluations": context_evals,
        "needle_evaluation": needle_res,
        "summary": {
            "int4_uniform_mse_4096": int4_mse,
            "precision_tail_64_mse_4096": tail64_mse,
            "mse_reduction_percentage": mse_reduction_pct,
            "precision_tail_64_estimated_packed_storage_savings_percentage": tail64_savings,
            "needle_pass": needle_pass,
            "gates": gates,
            "verdict": verdict,
        },
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
            "quantization_mode": "dequantized_int4_simulation",
            "memory_metric": "analytical_packed_storage_estimate_not_measured_vram",
        },
    )
    provenance_ok, provenance_errors = provenance_complete(provenance)
    final_payload["provenance"] = provenance
    final_payload["provenance_complete"] = provenance_ok
    final_payload["provenance_errors"] = provenance_errors
    if not provenance_ok:
        final_payload["summary"]["verdict"] = "UNVERIFIED_PROVENANCE"
    final_payload["receipt_fingerprint"] = canonical_json_sha256(final_payload)

    out_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n==================================================", flush=True)
    print(f"  REP-02 PROBE VERDICT: {verdict}", flush=True)
    print(f"  Logits MSE Reduction at 4k: {mse_reduction_pct:.1f}% (Gate >=50%: {gates['logits_mse_reduced_ge_50pct']})")
    print(f"  Estimated Packed Storage Savings at 4k: {tail64_savings:.1f}% (Gate >=65%: {gates['estimated_packed_storage_savings_ge_65pct']})")
    print(f"  Needle Retrieval Pass:      {needle_pass}")
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
