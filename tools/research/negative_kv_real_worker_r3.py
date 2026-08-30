#!/usr/bin/env python3
"""R3 GPU worker retaining every decisive tensor for independent replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research.negative_kv_real_worker import (
    CONTEXT_LENGTH,
    LAYERS,
    PROJECTION_SEEDS,
    REP_LENGTH,
    aggregate,
    build_contexts,
    evaluate_rep03,
    evaluate_rep06,
    evaluate_rsh01,
    evaluate_rsh03,
    evaluate_rsh04,
    logits_entropy,
    normalized_hadamard,
    tensor_sha256,
    write_json,
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retain(tensor, path: pathlib.Path, torch, ledger: list[dict], kind: str, **identity) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().contiguous().cpu(), path)
    ledger.append({
        "file": path.name,
        "kind": kind,
        **identity,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tensor_dir = args.output.parent / "tensors"
    if tensor_dir.exists() and any(tensor_dir.iterdir()):
        raise RuntimeError(f"tensor output directory is not empty: {tensor_dir}")
    tensor_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260824)
    torch.cuda.manual_seed_all(20260824)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    contexts, task_ids = build_contexts(tokenizer, args.corpus)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": "cuda"}, trust_remote_code=True,
    ).eval()
    language, lm_head = model.model, model.lm_head
    hadamard = normalized_hadamard(256, torch, "cuda")
    samples: list[dict] = []
    context_ledger: list[dict] = []
    tensor_ledger: list[dict] = []
    weight_done = False

    for context_index, ids in enumerate(contexts):
        captured_k, captured_inputs, handles = {}, {}, []
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
        retain(entropy, tensor_dir / f"context_{context_index:02d}_entropy.pt", torch, tensor_ledger,
               "entropy", context=context_index)
        context_ledger.append({
            "context": context_index,
            "token_count": len(ids),
            "token_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            "task_ids": task_ids[context_index],
            "entropy_sha256": tensor_sha256(entropy, torch),
        })

        for layer_index in LAYERS:
            raw_k = captured_k[layer_index].squeeze(0).contiguous()
            retain(raw_k, tensor_dir / f"context_{context_index:02d}_layer_{layer_index:02d}_k.pt",
                   torch, tensor_ledger, "k_projection", context=context_index, layer=layer_index)
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
                inputs = captured_inputs[layer_index].squeeze(0)[::16][:256].contiguous()
                retain(inputs, tensor_dir / f"layer_{layer_index:02d}_inputs.pt", torch, tensor_ledger,
                       "q_inputs", layer=layer_index)
                weight = language.layers[layer_index].self_attn.q_proj.weight.detach()
                for slice_index in range(2):
                    start = slice_index * 1024
                    matrix = weight[start:start + 1024, :].contiguous()
                    retain(matrix, tensor_dir / f"layer_{layer_index:02d}_slice_{slice_index:02d}_weight.pt",
                           torch, tensor_ledger, "q_weight", layer=layer_index, slice=slice_index)
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

    if len(tensor_ledger) != 39 or len(samples) != 78:
        raise RuntimeError(f"retention mismatch: {len(tensor_ledger)} tensors, {len(samples)} samples")
    payload = {
        "schema": "negative-kv-real-worker-r3-v1",
        "pid": os.getpid(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_model_class": type(model).__name__,
        "layers": list(LAYERS),
        "projection_seeds": list(PROJECTION_SEEDS),
        "context_ledger": context_ledger,
        "retained_tensors": tensor_ledger,
        "samples": samples,
        "scores": aggregate(samples),
    }
    write_json(args.output, payload)
    print(json.dumps(payload["scores"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
