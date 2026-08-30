#!/usr/bin/env python3
"""Retain decisive learned GDN output vectors and collateral cosines."""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.research import gdn02_learned_state_worker as prior


def run(args) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    records = prior.load_records(args.corpus)
    conditions = prior.build_conditions(records)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    lengths = {
        name: [len(tokenizer.encode(text, add_special_tokens=False)) for text in texts]
        for name, texts in conditions.items()
    }
    if lengths["baseline"] != lengths["treatment"]:
        raise ValueError("baseline/treatment token lengths differ")
    if lengths["baseline"][prior.TARGET_INDEX] != lengths["oracle"][0]:
        raise ValueError("target/oracle token lengths differ")

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    if not all(hasattr(model.model.layers[index], "linear_attn") for index in prior.LAYERS):
        raise ValueError("frozen layers are not learned GatedDeltaNet modules")

    original_rule = modeling.torch_chunk_gated_delta_rule
    captured: list[Any] = []

    def capturing_rule(*rule_args, **rule_kwargs):
        rule_kwargs["output_final_state"] = True
        output, state = original_rule(*rule_args, **rule_kwargs)
        captured.append(state)
        return output, state

    modeling.torch_chunk_gated_delta_rule = capturing_rule

    def evaluate(layer, texts: list[str]):
        outputs, state_hashes = [], []
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start:start + args.batch_size]
            encoded = tokenizer(batch, padding=True, add_special_tokens=False, return_tensors="pt")
            input_ids = encoded["input_ids"].to("cuda")
            mask = encoded["attention_mask"].to("cuda")
            row_lengths = mask.sum(dim=1).tolist()
            captured.clear()
            with torch.inference_mode():
                hidden = model.model.embed_tokens(input_ids)
                normalized = layer.input_layernorm(hidden)
                result = layer.linear_attn(normalized, attention_mask=mask)
            if len(captured) != 1 or captured[0] is None:
                raise RuntimeError("official recurrent state was not captured")
            state = captured[0]
            for row_index, length in enumerate(row_lengths):
                outputs.append(result[row_index, int(length) - 1].float().cpu())
                state_hashes.append(prior.tensor_hash(state[row_index]))
            del input_ids, mask, hidden, normalized, result, state
        return torch.stack(outputs), state_hashes

    cells, retained = [], {}
    try:
        for layer_index in prior.LAYERS:
            layer = model.model.layers[layer_index]
            baseline, baseline_states = evaluate(layer, conditions["baseline"])
            treatment, treatment_states = evaluate(layer, conditions["treatment"])
            oracle, oracle_states = evaluate(layer, conditions["oracle"])
            b = baseline[prior.TARGET_INDEX]
            c = treatment[prior.TARGET_INDEX]
            o = oracle[0]
            retained[f"layer_{layer_index}.baseline"] = b.contiguous()
            retained[f"layer_{layer_index}.treatment"] = c.contiguous()
            retained[f"layer_{layer_index}.oracle"] = o.contiguous()
            d_co = float(torch.linalg.vector_norm(c - o).item())
            d_cb = float(torch.linalg.vector_norm(c - b).item())
            d_bo = float(torch.linalg.vector_norm(b - o).item())
            if d_bo < 1e-4:
                raise RuntimeError(f"oracle target not material at layer {layer_index}: {d_bo}")
            leakage = 100.0 * d_co / max(d_co + d_cb, 1e-12)
            fidelity = 100.0 * max(0.0, 1.0 - d_co / d_bo)
            indexes = [index for index in range(50) if index != prior.TARGET_INDEX]
            cosine_tensor = torch.nn.functional.cosine_similarity(
                baseline[indexes], treatment[indexes], dim=1
            ).float().cpu()
            cosines = [float(value) for value in cosine_tensor.tolist()]
            collateral = statistics.fmean(cosines) * 100.0
            target_hashes = [
                baseline_states[prior.TARGET_INDEX],
                treatment_states[prior.TARGET_INDEX],
                oracle_states[0],
            ]
            cells.append({
                "layer": layer_index,
                "module_class": type(layer.linear_attn).__name__,
                "target_key": records[prior.TARGET_INDEX]["key"],
                "old_value": prior.OLD_VALUE,
                "new_value": prior.NEW_VALUE,
                "baseline_oracle_distance": d_bo,
                "correction_oracle_distance": d_co,
                "correction_baseline_distance": d_cb,
                "old_fact_leakage_pct": leakage,
                "updated_fact_fidelity_pct": fidelity,
                "collateral_retention_pct": collateral,
                "collateral_cosines": cosines,
                "target_output_hashes": {
                    "baseline": prior.tensor_hash(b),
                    "treatment": prior.tensor_hash(c),
                    "oracle": prior.tensor_hash(o),
                },
                "target_state_hashes": {
                    "baseline": target_hashes[0],
                    "treatment": target_hashes[1],
                    "oracle": target_hashes[2],
                },
                "distinct_recurrent_state_conditions": len(set(target_hashes)),
            })
    finally:
        modeling.torch_chunk_gated_delta_rule = original_rule

    save_file(retained, str(args.bundle))
    metrics = {
        "learned_gdn_layer_cells": len(cells),
        "retained_decisive_layer_cells": len(cells),
        "retained_collateral_cosines": sum(len(cell["collateral_cosines"]) for cell in cells),
        "median_old_fact_leakage_pct": statistics.median(cell["old_fact_leakage_pct"] for cell in cells),
        "median_collateral_retention_pct": statistics.median(cell["collateral_retention_pct"] for cell in cells),
        "median_updated_fact_fidelity_pct": statistics.median(cell["updated_fact_fidelity_pct"] for cell in cells),
        "distinct_recurrent_state_conditions": min(cell["distinct_recurrent_state_conditions"] for cell in cells),
    }
    return {
        "schema": "gdn02-learned-state-worker-v2",
        "model_path": str(args.model),
        "model_file_sha256": prior.sha256_file(args.model / "model.safetensors-00001-of-00001.safetensors"),
        "corpus_sha256": prior.sha256_file(args.corpus),
        "records": records,
        "target_index": prior.TARGET_INDEX,
        "token_lengths": lengths,
        "bundle": str(args.bundle),
        "bundle_sha256": prior.sha256_file(args.bundle),
        "bundle_keys": sorted(retained),
        "cells": cells,
        "metrics": metrics,
        "hardware": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()
    started = time.time()
    result = run(args)
    result["elapsed_seconds"] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": result["metrics"], "bundle_sha256": result["bundle_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
