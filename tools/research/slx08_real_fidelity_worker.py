#!/usr/bin/env python3
"""Real-Qwen fidelity worker for the corrected SLX-08 block gather."""
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
CONTEXT_LENGTH = 8192
BLOCK_SIZE = 256


def tensor_sha256(tensor, torch) -> str:
    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.view(torch.int16)
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def build_contexts(tokenizer, corpus: pathlib.Path):
    stream, tasks = [], []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        ids = tokenizer.encode(
            f"Problem {row['task_id']}: {row['prompt']}\nAnswer: {row['answer']}\n\n",
            add_special_tokens=False,
        )
        stream.extend(ids)
        tasks.extend([row["task_id"]] * len(ids))
        if len(stream) >= 2 * CONTEXT_LENGTH:
            break
    if len(stream) < 2 * CONTEXT_LENGTH:
        raise ValueError("frozen corpus did not yield two contexts")
    return (
        [stream[i * CONTEXT_LENGTH:(i + 1) * CONTEXT_LENGTH] for i in range(2)],
        [list(dict.fromkeys(tasks[i * CONTEXT_LENGTH:(i + 1) * CONTEXT_LENGTH])) for i in range(2)],
    )


def cosine(left, right, torch) -> float:
    return float(torch.nn.functional.cosine_similarity(left.flatten(), right.flatten(), dim=0).item())


def attention_context(query, keys, values, torch):
    scores = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(keys.shape[-1])
    return torch.matmul(torch.softmax(scores.float(), dim=-1), values.float())


def evaluate_cell(q_raw, k_raw, v_raw, attention, torch) -> dict:
    length = q_raw.shape[1]
    query, _gate = torch.chunk(q_raw.view(1, length, -1, 512), 2, dim=-1)
    query = attention.q_norm(query).transpose(1, 2).squeeze(0)
    keys = attention.k_norm(k_raw.view(1, length, -1, 256)).transpose(1, 2).squeeze(0)
    values = v_raw.view(1, length, -1, 256).transpose(1, 2).squeeze(0)
    keys = keys.repeat_interleave(4, dim=0).float()
    values = values.repeat_interleave(4, dim=0).float()
    query = query[:, -1:, :].float()

    dense = attention_context(query, keys, values, torch)
    blocks = keys.view(8, length // BLOCK_SIZE, BLOCK_SIZE, 256).mean(dim=2)
    block_scores = torch.matmul(query, blocks.transpose(-1, -2)).squeeze(1)
    selected = block_scores.topk(blocks.shape[1] // 2, dim=-1).indices
    offsets = torch.arange(BLOCK_SIZE, device=keys.device)
    token_indices = (selected.unsqueeze(-1) * BLOCK_SIZE + offsets).reshape(8, -1)
    gather_index = token_indices.unsqueeze(-1).expand(-1, -1, 256)
    selected_keys = torch.gather(keys, 1, gather_index)
    selected_values = torch.gather(values, 1, gather_index)
    corrected = attention_context(query, selected_keys, selected_values, torch)
    legacy = attention_context(query, keys[:, :length // 2, :], values[:, :length // 2, :], torch)
    first_half = set(range(blocks.shape[1] // 2))
    differs = all(set(selected[head].tolist()) != first_half for head in range(8))
    return {
        "selected_block_context_cosine": cosine(dense, corrected, torch),
        "legacy_first_half_context_cosine": cosine(dense, legacy, torch),
        "computed_indices_materially_used": differs,
        "selected_block_indices": selected.tolist(),
        "retained_fraction": 0.5,
    }


def aggregate(samples: list[dict]) -> dict:
    return {
        "actual_qkv_cells": len(samples),
        "all_decisive_tensors_from_frozen_model": all(row["tensor_source"] == "frozen_qwen" for row in samples),
        "computed_top_block_indices_materially_used": all(row["computed_indices_materially_used"] for row in samples),
        "median_selected_block_context_cosine": statistics.median(row["selected_block_context_cosine"] for row in samples),
        "median_legacy_first_half_context_cosine": statistics.median(row["legacy_first_half_context_cosine"] for row in samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    contexts, task_ids = build_contexts(tokenizer, args.corpus)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda"}, trust_remote_code=True,
    ).eval()
    language = model.model
    samples = []
    context_ledger = []
    for context_index, ids in enumerate(contexts):
        q_capture, k_capture, v_capture, handles = {}, {}, {}, []
        for layer_index in LAYERS:
            attention = language.layers[layer_index].self_attn
            handles.extend([
                attention.q_proj.register_forward_hook(lambda _m, _i, out, layer=layer_index: q_capture.__setitem__(layer, out.detach())),
                attention.k_proj.register_forward_hook(lambda _m, _i, out, layer=layer_index: k_capture.__setitem__(layer, out.detach())),
                attention.v_proj.register_forward_hook(lambda _m, _i, out, layer=layer_index: v_capture.__setitem__(layer, out.detach())),
            ])
        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            output = language(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False, return_dict=True)
        for handle in handles:
            handle.remove()
        if set(q_capture) != set(LAYERS) or set(k_capture) != set(LAYERS) or set(v_capture) != set(LAYERS):
            raise RuntimeError("incomplete QKV capture")
        context_ledger.append({
            "context": context_index, "token_count": len(ids), "task_ids": task_ids[context_index],
            "token_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        })
        for layer_index in LAYERS:
            identity = {
                "tensor_source": "frozen_qwen", "context": context_index, "layer": layer_index,
                "q_sha256": tensor_sha256(q_capture[layer_index], torch),
                "k_sha256": tensor_sha256(k_capture[layer_index], torch),
                "v_sha256": tensor_sha256(v_capture[layer_index], torch),
            }
            samples.append({
                **identity,
                **evaluate_cell(q_capture[layer_index], k_capture[layer_index], v_capture[layer_index],
                                language.layers[layer_index].self_attn, torch),
            })
        del output, input_ids, q_capture, k_capture, v_capture
        torch.cuda.empty_cache()

    payload = {
        "schema": "slx08-real-fidelity-worker-v1", "pid": os.getpid(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": torch.cuda.get_device_name(0), "model_class": type(model).__name__,
        "context_ledger": context_ledger, "samples": samples, "scores": aggregate(samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["scores"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
