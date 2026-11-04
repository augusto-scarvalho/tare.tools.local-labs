#!/usr/bin/env python3
"""ADAPT-00A: deterministic LoRA mechanics/retention smoke on a small causal LM.

Heavy dependencies are imported only in ``run`` so ``--selfcheck`` and CI
compile validation remain host-independent.
"""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import math
import pathlib
import random
import sys
import time
from dataclasses import dataclass


PROMPT_TEMPLATE = (
    "Solve the problem. Show your reasoning, then on the final line write only:\n"
    "#### <answer>\nwhere <answer> is the final number.\n\n{prompt}"
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pairs(teacher_path: pathlib.Path, prompt_path: pathlib.Path,
               seed: int) -> list[dict[str, str]]:
    prompts: dict[str, str] = {}
    with prompt_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            prompts[row["task_id"]] = row["prompt"]

    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    pairs = []
    for row in teacher:
        task_id = row.get("task_id")
        completion = (row.get("completion") or "").strip()
        if row.get("ok") and task_id in prompts and completion:
            pairs.append({
                "task_id": task_id,
                "prompt": PROMPT_TEMPLATE.format(prompt=prompts[task_id]),
                "completion": completion,
            })
    pairs.sort(key=lambda row: row["task_id"])
    random.Random(seed).shuffle(pairs)
    return pairs


def protected_blocks(path: pathlib.Path, tokenizer, max_length: int,
                     limit: int = 16) -> list[list[int]]:
    text = path.read_text(encoding="utf-8")
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    width = max_length
    return [ids[i:i + width] for i in range(0, len(ids) - width + 1, width)][:limit]


@dataclass
class Batch:
    input_ids: object
    attention_mask: object
    labels: object


def target_batch(row: dict[str, str], tokenizer, max_length: int, torch) -> Batch:
    prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    # Preserve conditioning even for a long teacher response, and preserve the
    # response tail because it contains GSM8K's machine-checkable final line.
    prompt_ids = prompt_ids[-(max_length // 2):]
    completion_room = max(1, max_length - len(prompt_ids) - len(eos))
    completion_ids = completion_ids[-completion_room:]
    ids = prompt_ids + completion_ids + eos
    completion_n = len(completion_ids) + len(eos)
    labels = [-100] * (len(ids) - completion_n) + ids[-completion_n:]
    return Batch(
        input_ids=torch.tensor([ids], dtype=torch.long, device="cuda"),
        attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device="cuda"),
        labels=torch.tensor([labels], dtype=torch.long, device="cuda"),
    )


def protected_batch(ids: list[int], torch) -> Batch:
    tensor = torch.tensor([ids], dtype=torch.long, device="cuda")
    return Batch(tensor, torch.ones_like(tensor), tensor.clone())


def mean_loss(model, batches: list[Batch], torch) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for batch in batches:
            value = model(input_ids=batch.input_ids,
                          attention_mask=batch.attention_mask,
                          labels=batch.labels).loss.float().item()
            values.append(value)
    return sum(values) / len(values)


def adapter_config(method: str, tokenizer, train_rows: list[dict[str, str]],
                   args: argparse.Namespace):
    from peft import (BOFTConfig, IA3Config, LoHaConfig, LoKrConfig,
                      LoraConfig, TrainableTokensConfig)

    common = {"task_type": "CAUSAL_LM", "target_modules": "all-linear"}
    if method == "lora":
        return LoraConfig(r=args.rank, lora_alpha=args.alpha,
                          lora_dropout=0.0, **common), {
                              "rank": args.rank, "alpha": args.alpha,
                              "target_modules": "all-linear"}
    if method == "dora":
        return LoraConfig(r=args.rank, lora_alpha=args.alpha,
                          lora_dropout=0.0, use_dora=True, **common), {
                              "rank": args.rank, "alpha": args.alpha,
                              "target_modules": "all-linear", "use_dora": True}
    if method == "loha":
        return LoHaConfig(r=args.rank, alpha=args.alpha, **common), {
            "rank": args.rank, "alpha": args.alpha, "target_modules": "all-linear"}
    if method == "lokr":
        return LoKrConfig(r=args.rank, alpha=args.alpha, **common), {
            "rank": args.rank, "alpha": args.alpha, "target_modules": "all-linear"}
    if method == "boft":
        # PEFT 0.20 repeatedly retries its optional FBD CUDA JIT once per
        # wrapped layer after a build failure. The frozen toolchain preflight
        # already failed that extension against torch 2.5.1/CUDA 12.4, so bind
        # the documented torch fallback once instead of recompiling 187 times.
        from peft.tuners.boft import layer as boft_layer
        boft_layer._FBD_CUDA = False
        return BOFTConfig(boft_block_size=4, boft_n_butterfly_factor=1,
                          **common), {
                              "block_size": 4, "butterfly_factors": 1,
                              "target_modules": "all-linear",
                              "fbd_cuda_extension": "preflight_failed; torch fallback"}
    if method == "ia3":
        targets = ["k_proj", "v_proj", "down_proj", "in_proj_qkv", "out_proj"]
        feedforward = ["down_proj", "out_proj"]
        return IA3Config(task_type="CAUSAL_LM", target_modules=targets,
                         feedforward_modules=feedforward), {
                             "target_modules": targets,
                             "feedforward_modules": feedforward}
    if method == "trainable_tokens":
        counts: collections.Counter[int] = collections.Counter()
        for row in train_rows:
            counts.update(tokenizer(row["prompt"] + row["completion"],
                                    add_special_tokens=False)["input_ids"])
        indices = [token for token, _ in counts.most_common(args.trainable_token_count)]
        if not indices:
            raise ValueError("no token indices found in training corpus")
        return TrainableTokensConfig(task_type="CAUSAL_LM", token_indices=indices), {
            "selection": "most-frequent tokens in frozen train split",
            "requested_token_count": args.trainable_token_count,
            "realized_token_count": len(indices),
        }
    raise ValueError(method)


def run(args: argparse.Namespace) -> int:
    import peft
    import torch
    import transformers
    from peft import PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    teacher_path = pathlib.Path(args.teacher).resolve()
    prompt_path = pathlib.Path(args.prompts).resolve()
    protected_path = pathlib.Path(args.protected_file).resolve()
    out_dir = pathlib.Path(args.output).resolve()
    adapter_dir = out_dir / "adapter"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(teacher_path, prompt_path, args.seed)
    required = args.train_n + args.target_n
    if len(pairs) < required:
        raise SystemExit(f"need {required} valid pairs, found {len(pairs)}")
    train_rows = pairs[:args.train_n]
    target_rows = pairs[args.train_n:required]

    load_from = args.model_path or args.model
    revision_kw = {} if args.model_path else {"revision": args.revision}
    tokenizer = AutoTokenizer.from_pretrained(load_from, **revision_kw)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        load_from, torch_dtype=torch.bfloat16, **revision_kw,
        device_map={"": "cuda"}, attn_implementation="sdpa")
    model.config.use_cache = False
    peft_config, method_detail = adapter_config(
        args.method, tokenizer, train_rows, args)
    model = get_peft_model(model, peft_config)

    target_batches = [target_batch(row, tokenizer, args.max_length, torch)
                      for row in target_rows]
    protected_ids = protected_blocks(protected_path, tokenizer, args.max_length)
    if not protected_ids:
        raise SystemExit("protected file is too short for one evaluation block")
    protected_batches = [protected_batch(ids, torch) for ids in protected_ids]

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    initial_target = mean_loss(model, target_batches, torch)
    initial_protected = mean_loss(model, protected_batches, torch)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    losses: list[float] = []
    nonzero_gradient = False
    started = time.monotonic()
    for step in range(args.steps):
        row = train_rows[step % len(train_rows)]
        batch = target_batch(row, tokenizer, args.max_length, torch)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=batch.input_ids,
                     attention_mask=batch.attention_mask,
                     labels=batch.labels).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        nonzero_gradient = nonzero_gradient or any(
            p.grad is not None and bool(torch.count_nonzero(p.grad).item())
            for p in model.parameters() if p.requires_grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.float().item())

    final_target = mean_loss(model, target_batches, torch)
    final_protected = mean_loss(model, protected_batches, torch)
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    del optimizer, model, target_batches, protected_batches
    gc.collect()
    torch.cuda.empty_cache()

    fresh = AutoModelForCausalLM.from_pretrained(
        load_from, torch_dtype=torch.bfloat16, **revision_kw,
        device_map={"": "cuda"}, attn_implementation="sdpa")
    fresh.config.use_cache = False
    reloaded = PeftModel.from_pretrained(fresh, adapter_dir)
    reload_batches = [target_batch(row, tokenizer, args.max_length, torch)
                      for row in target_rows]
    reload_target = mean_loss(reloaded, reload_batches, torch)

    target_improvement = (initial_target - final_target) / initial_target
    protected_regression = (final_protected - initial_protected) / initial_protected
    reload_delta = abs(reload_target - final_target) / final_target
    finite = all(math.isfinite(v) for v in [
        *losses, initial_target, final_target, initial_protected,
        final_protected, reload_target])
    gates = {
        "finite_losses": finite,
        "nonzero_gradient": nonzero_gradient,
        "target_improvement_ge_1pct": target_improvement >= 0.01,
        "protected_regression_le_15pct": protected_regression <= 0.15,
        "reload_delta_le_0_5pct": reload_delta <= 0.005,
        "peak_vram_lt_23_gib": peak_gib < 23.0,
    }
    result = {
        "schema_version": 1,
        "verdict": "PASS" if all(gates.values()) else "FAIL",
        "model": args.model,
        "model_path": args.model_path,
        "revision": args.revision,
        "seed": args.seed,
        "versions": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__,
        },
        "inputs": {
            "teacher": str(teacher_path), "teacher_sha256": sha256(teacher_path),
            "prompts": str(prompt_path), "prompts_sha256": sha256(prompt_path),
            "protected_file": str(protected_path),
            "protected_sha256": sha256(protected_path),
        },
        "split": {
            "valid_pairs": len(pairs), "train_n": len(train_rows),
            "target_n": len(target_rows),
            "train_task_ids": [row["task_id"] for row in train_rows],
            "target_task_ids": [row["task_id"] for row in target_rows],
        },
        "configuration": {
            "method": args.method, **method_detail, "steps": args.steps,
            "learning_rate": args.learning_rate, "max_length": args.max_length,
            "precision": "bfloat16", "batch_size": 1,
        },
        "parameters": {"trainable": trainable, "total_with_adapter": total},
        "metrics": {
            "train_loss_first": losses[0], "train_loss_last": losses[-1],
            "initial_target_loss": initial_target, "final_target_loss": final_target,
            "target_improvement_fraction": target_improvement,
            "initial_protected_loss": initial_protected,
            "final_protected_loss": final_protected,
            "protected_regression_fraction": protected_regression,
            "reload_target_loss": reload_target,
            "reload_delta_fraction": reload_delta,
            "peak_allocated_vram_gib": peak_gib,
            "elapsed_seconds": time.monotonic() - started,
        },
        "gates": gates,
        "adapter_dir": str(adapter_dir),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["verdict"] == "PASS" else 2


def selfcheck() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        prompts = root / "prompts.jsonl"
        teacher = root / "teacher.json"
        prompts.write_text("\n".join(json.dumps({
            "task_id": f"gsm8k/{i}", "prompt": f"question {i}"})
            for i in range(4)) + "\n", encoding="utf-8")
        teacher.write_text(json.dumps([{
            "task_id": f"gsm8k/{i}", "completion": f"#### {i}", "ok": True}
            for i in range(4)]), encoding="utf-8")
        a = load_pairs(teacher, prompts, 7)
        b = load_pairs(teacher, prompts, 7)
        assert a == b and len(a) == 4
        assert {row["task_id"] for row in a} == {f"gsm8k/{i}" for i in range(4)}
        assert all("#### <answer>" in row["prompt"] for row in a)
        assert sha256(prompts) == hashlib.sha256(prompts.read_bytes()).hexdigest()
    print("adapt00_lora_smoke selfcheck: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--model-path")
    parser.add_argument("--revision")
    parser.add_argument("--teacher", default="runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json")
    parser.add_argument("--prompts", default="workloads/gsm8k.jsonl")
    parser.add_argument("--protected-file", default="README.md")
    parser.add_argument("--output", default="runs/research/ADAPT-00A-MECHANICS-2026-08-24/raw")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--method", choices=(
        "lora", "dora", "loha", "lokr", "boft", "ia3", "trainable_tokens"),
        default="lora")
    parser.add_argument("--train-n", type=int, default=128)
    parser.add_argument("--target-n", type=int, default=32)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--trainable-token-count", type=int, default=4096)
    args = parser.parse_args()
    if args.selfcheck:
        return selfcheck()
    if not args.revision:
        parser.error("--revision is required for a real run")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
