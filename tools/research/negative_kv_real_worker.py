#!/usr/bin/env python3
"""GPU worker for BACKLOG-NEGATIVE-KV-REAL-SCREEN-01.

All decisive tensors are extracted from a frozen Qwen forward pass or its
frozen weights. Randomness is used only by the preregistered RaBitQ treatment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import time

LAYERS = (3, 7, 11, 15, 19, 23)
CONTEXT_LENGTH = 4096
REP_LENGTH = 2048
BLOCK_SIZE = 32
PROJECTION_SEEDS = (20260824, 20260825, 20260826)


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def tensor_sha256(tensor, torch) -> str:
    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.view(torch.int16)
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def quantize_symmetric(tensor, bits: int, torch):
    shape = tensor.shape
    flat = tensor.reshape(-1, BLOCK_SIZE)
    qmax = (1 << (bits - 1)) - 1
    scale = flat.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    quantized = (flat / scale).round().clamp(-qmax, qmax) * scale
    return quantized.reshape(shape)


def quantize_fibonacci(tensor, torch):
    shape = tensor.shape
    flat = tensor.reshape(-1, BLOCK_SIZE)
    scale = flat.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    levels = torch.tensor(
        [-1.0, -13 / 21, -8 / 21, -5 / 21, -3 / 21, -2 / 21, -1 / 21,
         0.0, 1 / 21, 2 / 21, 3 / 21, 5 / 21, 8 / 21, 13 / 21, 1.0],
        device=tensor.device, dtype=torch.float32,
    )
    normalized = flat / scale
    indexes = (normalized.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    return (levels[indexes] * scale).reshape(shape)


def normalized_hadamard(n: int, torch, device):
    h = torch.ones((1, 1), dtype=torch.float32, device=device)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h / math.sqrt(n)


def cosine(left, right, torch) -> float:
    return float(torch.nn.functional.cosine_similarity(left.reshape(-1), right.reshape(-1), dim=0).item())


def attention_probs(keys, torch):
    query = keys[:, -1:, :]
    scores = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(keys.shape[-1])
    return torch.softmax(scores, dim=-1)


def build_contexts(tokenizer, corpus: pathlib.Path) -> tuple[list[list[int]], list[list[str]]]:
    token_stream: list[int] = []
    token_tasks: list[str] = []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        text = f"Problem {row['task_id']}: {row['prompt']}\nAnswer: {row['answer']}\n\n"
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_stream.extend(ids)
        token_tasks.extend([row["task_id"]] * len(ids))
        if len(token_stream) >= 3 * CONTEXT_LENGTH:
            break
    if len(token_stream) < 3 * CONTEXT_LENGTH:
        raise ValueError("frozen corpus did not produce three complete contexts")
    contexts, task_ids = [], []
    for index in range(3):
        start = index * CONTEXT_LENGTH
        end = start + CONTEXT_LENGTH
        contexts.append(token_stream[start:end])
        task_ids.append(list(dict.fromkeys(token_tasks[start:end])))
    return contexts, task_ids


def logits_entropy(hidden, lm_head, torch):
    values = []
    for start in range(0, REP_LENGTH, 32):
        logits = lm_head(hidden[:, start:start + 32, :]).float().squeeze(0)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        values.append((-(probs * log_probs).sum(dim=-1)).detach())
        del logits, log_probs, probs
    return torch.cat(values)


def evaluate_rsh01(weight, torch) -> dict:
    source = weight.float()
    uniform = quantize_symmetric(source, 4, torch)
    fib = quantize_fibonacci(source, torch)
    uniform_mse = float(torch.mean((uniform - source) ** 2).item())
    fib_mse = float(torch.mean((fib - source) ** 2).item())
    variance = float(torch.var(source).item())
    uniform_sqnr = 10.0 * math.log10(variance / uniform_mse)
    fib_sqnr = 10.0 * math.log10(variance / fib_mse)
    return {
        "uniform_mse": uniform_mse,
        "fib_mse": fib_mse,
        "fib_mse_ratio_vs_uniform": fib_mse / uniform_mse,
        "fib_sqnr_gain_db": fib_sqnr - uniform_sqnr,
        "fib_cosine_similarity": cosine(source, fib, torch),
    }


def evaluate_rsh03(weight, inputs, torch) -> dict:
    source = weight.float()
    quantized = quantize_symmetric(source, 4, torch)
    residual = source - quantized
    u, singular, vh = torch.linalg.svd(residual, full_matrices=False)
    rank = 4
    correction = (u[:, :rank] * singular[:rank]) @ vh[:rank, :]
    compensated = quantized + correction
    x = inputs.float()
    exact = x @ source.t()
    baseline = x @ quantized.t()
    treated = x @ compensated.t()
    baseline_mse = float(torch.mean((baseline - exact) ** 2).item())
    treated_mse = float(torch.mean((treated - exact) ** 2).item())
    overhead = rank * (source.shape[0] + source.shape[1]) / source.numel()
    return {
        "baseline_output_mse": baseline_mse,
        "treated_output_mse": treated_mse,
        "rank4_mse_recovery": (baseline_mse - treated_mse) / baseline_mse,
        "rank4_output_cosine": cosine(exact, treated, torch),
        "rank4_parameter_overhead": overhead,
    }


def evaluate_rep03(keys, hadamard, torch) -> dict:
    source = keys.float()
    direct = quantize_symmetric(source, 4, torch)
    rotated = source @ hadamard
    restored = quantize_symmetric(rotated, 4, torch) @ hadamard.t()
    direct_mse = float(torch.mean((direct - source) ** 2).item())
    treated_mse = float(torch.mean((restored - source) ** 2).item())
    exact_attention = attention_probs(source, torch)
    treated_attention = attention_probs(restored, torch)
    return {
        "direct_mse": direct_mse,
        "hadamard_mse": treated_mse,
        "hadamard_mse_reduction": (direct_mse - treated_mse) / direct_mse,
        "hadamard_attention_cosine": cosine(exact_attention, treated_attention, torch),
    }


def evaluate_rsh04(keys, torch) -> dict:
    source = keys.float()
    heads, length, dim = source.shape
    blocks = source.reshape(heads, length // BLOCK_SIZE, BLOCK_SIZE, dim).mean(dim=2)
    query = source[:, -1, :]
    exact_scores = torch.einsum("hd,hbd->hb", query, blocks)
    keep = max(1, math.ceil(blocks.shape[1] * 0.25))
    exact_top = exact_scores.topk(keep, dim=-1).indices
    recalls = []
    for seed in PROJECTION_SEEDS:
        generator = torch.Generator(device=source.device).manual_seed(seed)
        projection = torch.randn((dim, 128), generator=generator, device=source.device)
        block_bits = (blocks @ projection) >= 0
        query_bits = (query @ projection) >= 0
        similarities = (block_bits == query_bits[:, None, :]).float().mean(dim=-1)
        approx_top = similarities.topk(keep, dim=-1).indices
        per_head = []
        for head in range(heads):
            overlap = len(set(exact_top[head].tolist()) & set(approx_top[head].tolist()))
            per_head.append(overlap / keep)
        recalls.append(statistics.mean(per_head))
    return {
        "binary_top_block_recall": statistics.median(recalls),
        "retained_fraction": keep / blocks.shape[1],
        "projection_seed_recalls": recalls,
        "total_blocks": blocks.shape[1],
    }


def mixed_precision(keys, entropy, torch):
    source = keys.float()
    treated = source.clone()
    low = entropy < 0.8
    medium = (entropy >= 0.8) & (entropy < 2.0)
    high = entropy >= 2.0
    if low.any():
        treated[:, low, :] = quantize_symmetric(source[:, low, :], 2, torch)
    if medium.any():
        treated[:, medium, :] = quantize_symmetric(source[:, medium, :], 4, torch)
    counts = {"low": int(low.sum().item()), "medium": int(medium.sum().item()), "high": int(high.sum().item())}
    average_bits = (2 * counts["low"] + 4 * counts["medium"] + 16 * counts["high"]) / len(entropy)
    return treated, average_bits, counts


def evaluate_rep06(keys, entropy, torch) -> dict:
    source = keys.float()
    static = quantize_symmetric(source, 4, torch)
    dynamic, average_bits, counts = mixed_precision(source, entropy, torch)
    exact_attention = attention_probs(source, torch)
    static_cos = cosine(exact_attention, attention_probs(static, torch), torch)
    dynamic_cos = cosine(exact_attention, attention_probs(dynamic, torch), torch)
    return {
        "average_bits_per_element": average_bits,
        "static_int4_attention_cosine": static_cos,
        "dynamic_attention_cosine": dynamic_cos,
        "dynamic_beats_static_int4": dynamic_cos > static_cos,
        "entropy_counts": counts,
        "entropy_min": float(entropy.min().item()),
        "entropy_median": float(entropy.median().item()),
        "entropy_max": float(entropy.max().item()),
    }


def aggregate(samples: list[dict]) -> dict:
    def median(candidate: str, metric: str) -> float:
        return statistics.median(row[metric] for row in samples if row["candidate"] == candidate)

    return {
        "actual_model_activation_cells": len({(row["context"], row["layer"]) for row in samples if row["candidate"] == "REP-03"}),
        "actual_model_weight_matrices": len([row for row in samples if row["candidate"] == "RSH-01"]),
        "candidate_hypotheses_evaluated": len({row["candidate"] for row in samples}),
        "all_decisive_tensors_from_frozen_model": all(row["tensor_source"] == "frozen_qwen" for row in samples),
        "rsh01_fib_mse_ratio_vs_uniform": median("RSH-01", "fib_mse_ratio_vs_uniform"),
        "rsh01_fib_sqnr_gain_db": median("RSH-01", "fib_sqnr_gain_db"),
        "rsh01_fib_cosine_similarity": median("RSH-01", "fib_cosine_similarity"),
        "rep03_hadamard_mse_reduction": median("REP-03", "hadamard_mse_reduction"),
        "rep03_hadamard_attention_cosine": median("REP-03", "hadamard_attention_cosine"),
        "rsh03_rank4_mse_recovery": median("RSH-03", "rank4_mse_recovery"),
        "rsh03_rank4_output_cosine": median("RSH-03", "rank4_output_cosine"),
        "rsh03_rank4_parameter_overhead": median("RSH-03", "rank4_parameter_overhead"),
        "rsh04_binary_top_block_recall": median("RSH-04", "binary_top_block_recall"),
        "rsh04_retained_fraction": median("RSH-04", "retained_fraction"),
        "rep06_average_bits_per_element": median("REP-06", "average_bits_per_element"),
        "rep06_dynamic_attention_cosine": median("REP-06", "dynamic_attention_cosine"),
        "rep06_dynamic_beats_static_int4": all(row["dynamic_beats_static_int4"] for row in samples if row["candidate"] == "REP-06"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(20260824)
    torch.cuda.manual_seed_all(20260824)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    contexts, task_ids = build_contexts(tokenizer, args.corpus)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": "cuda"}, trust_remote_code=True,
    ).eval()
    language = model.model
    lm_head = model.lm_head
    hadamard = normalized_hadamard(256, torch, "cuda")
    samples: list[dict] = []
    context_ledger = []
    weight_done = False

    for context_index, ids in enumerate(contexts):
        captured_k = {}
        captured_inputs = {}
        handles = []
        for layer_index in LAYERS:
            attention = language.layers[layer_index].self_attn
            handles.append(attention.k_proj.register_forward_hook(
                lambda _m, _i, output, layer=layer_index: captured_k.__setitem__(layer, output.detach())
            ))
            handles.append(attention.q_proj.register_forward_pre_hook(
                lambda _m, inputs, layer=layer_index: captured_inputs.__setitem__(layer, inputs[0].detach())
            ))
        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = language(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
            entropy = logits_entropy(output.last_hidden_state, lm_head, torch)
        for handle in handles:
            handle.remove()
        if set(captured_k) != set(LAYERS) or set(captured_inputs) != set(LAYERS):
            raise RuntimeError("incomplete attention hook capture")
        context_ledger.append({
            "context": context_index,
            "token_count": len(ids),
            "token_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            "task_ids": task_ids[context_index],
            "entropy_sha256": tensor_sha256(entropy, torch),
        })

        for layer_index in LAYERS:
            raw_k = captured_k[layer_index].squeeze(0)
            keys = raw_k[:REP_LENGTH].reshape(REP_LENGTH, 2, 256).permute(1, 0, 2).contiguous()
            keys_long = raw_k.reshape(CONTEXT_LENGTH, 2, 256).permute(1, 0, 2).contiguous()
            identity = {
                "tensor_source": "frozen_qwen", "context": context_index, "layer": layer_index,
                "tensor_sha256": tensor_sha256(raw_k, torch), "tensor_shape": list(raw_k.shape),
            }
            samples.append({"candidate": "REP-03", **identity, **evaluate_rep03(keys, hadamard, torch)})
            samples.append({"candidate": "RSH-04", **identity, **evaluate_rsh04(keys_long, torch)})
            samples.append({"candidate": "REP-06", **identity, **evaluate_rep06(keys, entropy, torch)})

            if not weight_done:
                weight = language.layers[layer_index].self_attn.q_proj.weight.detach()
                inputs = captured_inputs[layer_index].squeeze(0)[::16][:256]
                for slice_index in range(2):
                    start = slice_index * 1024
                    matrix = weight[start:start + 1024, :].contiguous()
                    matrix_identity = {
                        "tensor_source": "frozen_qwen", "context": 0, "layer": layer_index,
                        "slice": slice_index, "tensor_sha256": tensor_sha256(matrix, torch),
                        "tensor_shape": list(matrix.shape),
                    }
                    samples.append({"candidate": "RSH-01", **matrix_identity, **evaluate_rsh01(matrix, torch)})
                    samples.append({"candidate": "RSH-03", **matrix_identity, **evaluate_rsh03(matrix, inputs, torch)})
        weight_done = True
        del output, entropy, input_ids, attention_mask, captured_k, captured_inputs
        torch.cuda.empty_cache()

    scores = aggregate(samples)
    payload = {
        "schema": "negative-kv-real-worker-v1",
        "pid": os.getpid(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_model_class": type(model).__name__,
        "layers": list(LAYERS),
        "context_ledger": context_ledger,
        "samples": samples,
        "scores": scores,
    }
    write_json(args.output, payload)
    print(json.dumps(scores, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
