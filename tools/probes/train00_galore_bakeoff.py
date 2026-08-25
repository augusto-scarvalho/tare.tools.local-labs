#!/usr/bin/env python3
"""TRAIN-00: 3090 Fine-Tuning Memory Budget Bakeoff (GaLore vs LoKr vs Full AdamW).

Evaluates Gradient Low-Rank Projection (GaLore) memory savings and throughput on RTX 3090
by projecting optimizer states into rank-16 subspaces.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools/benchmarks"))

from tools.probes.adapt00_lora_smoke import load_pairs, target_batch  # noqa: E402
from tools.analysis.experiment_provenance import (  # noqa: E402
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
)


class GaLoreSubspaceOptimizer:
    """Implements Gradient Low-Rank Projection for 2D weight matrices."""
    def __init__(self, model, lr: float = 1e-4, rank: int = 16, update_proj_gap: int = 20, torch=None):
        self.model = model
        self.lr = lr
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.step_num = 0
        self.torch = torch

        # Initialize projection state and low-rank Adam moments for Linear layers
        self.params = []
        self.state = {}

        for p in model.parameters():
            if p.requires_grad:
                self.params.append(p)
                if p.ndim == 2 and min(p.shape) > rank:
                    m, n = p.shape
                    self.state[p] = {
                        "is_galore": True,
                        "type": "row" if m >= n else "col",
                        "proj": torch.randn((min(m, n), rank), dtype=torch.float32, device="cuda"),
                        "exp_avg": torch.zeros((max(m, n), rank), dtype=torch.float32, device="cuda"),
                        "exp_avg_sq": torch.zeros((max(m, n), rank), dtype=torch.float32, device="cuda"),
                    }
                else:
                    self.state[p] = {
                        "is_galore": False,
                        "exp_avg": torch.zeros_like(p.data, dtype=torch.float32),
                        "exp_avg_sq": torch.zeros_like(p.data, dtype=torch.float32),
                    }

    def _update_projection(self, p):
        grad = p.grad.data.float()
        # SVD or QR for subspace
        try:
            if grad.shape[0] >= grad.shape[1]:
                # Col subspace
                _, _, V = self.torch.linalg.svd(grad, full_matrices=False)
                self.state[p]["proj"] = V[:self.rank, :].t()
            else:
                U, _, _ = self.torch.linalg.svd(grad, full_matrices=False)
                self.state[p]["proj"] = U[:, :self.rank]
        except Exception:
            pass

    def step(self):
        self.step_num += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for p in self.params:
            if p.grad is None:
                continue
            st = self.state[p]

            if st["is_galore"]:
                if self.step_num % self.update_proj_gap == 1 or st["proj"] is None:
                    self._update_projection(p)

                g = p.grad.data.float()
                proj = st["proj"]

                # Project gradient: g_proj = g @ proj
                if g.shape[1] == proj.shape[0]:
                    g_proj = self.torch.matmul(g, proj)
                else:
                    g_proj = self.torch.matmul(g.t(), proj)

                exp_avg = st["exp_avg"]
                exp_avg_sq = st["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(g_proj, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_proj, g_proj, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** self.step_num
                bias_correction2 = 1 - beta2 ** self.step_num

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = self.lr / bias_correction1
                norm_step = (exp_avg / denom)

                # Project back to full space: delta = norm_step @ proj.t()
                if g.shape[1] == proj.shape[0]:
                    update = self.torch.matmul(norm_step, proj.t())
                else:
                    update = self.torch.matmul(proj, norm_step.t())

                p.data.add_(update.to(p.dtype), alpha=-1.0)
            else:
                g = p.grad.data.float()
                exp_avg = st["exp_avg"]
                exp_avg_sq = st["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg.to(p.dtype), denom.to(p.dtype), value=-self.lr)


def run_training_arm(arm_type: str, steps: int, args, repo, torch) -> dict:
    from peft import LoKrConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n--- Running Training Arm: {arm_type} ({steps} steps) ---", flush=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    pairs = load_pairs(repo / args.teacher, repo / args.prompts, args.seed)
    train_rows = pairs[:128]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map={"": "cuda"},
        attn_implementation="sdpa")
    model.config.use_cache = False

    if arm_type == "LOKR_PEFT":
        peft_config = LoKrConfig(task_type="CAUSAL_LM", r=8, alpha=16, target_modules=["q_proj", "v_proj", "gate_proj"])
        model = get_peft_model(model, peft_config)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    elif arm_type == "GALORE_R16":
        optimizer = GaLoreSubspaceOptimizer(model, lr=1e-4, rank=16, update_proj_gap=20, torch=torch)
    elif arm_type == "FULL_ADAMW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    else:
        raise ValueError(arm_type)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable Parameters: {trainable_params:,}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    model.train()

    losses = []
    t0 = time.monotonic()

    for step in range(steps):
        target_row = train_rows[step % len(train_rows)]
        batch_target = target_batch(target_row, tokenizer, 256, torch)

        if hasattr(optimizer, "zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)

        out = model(
            input_ids=batch_target.input_ids,
            attention_mask=batch_target.attention_mask,
            labels=batch_target.labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.float().item())

    elapsed = time.monotonic() - t0
    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    steps_per_sec = steps / elapsed if elapsed > 0 else 0.0

    print(f"  [{arm_type}] Peak VRAM = {peak_vram_gib:.2f} GiB | Steps/s = {steps_per_sec:.2f} | Final Loss = {losses[-1]:.4f}")

    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "arm": arm_type,
        "trainable_params": trainable_params,
        "peak_vram_gib": round(peak_vram_gib, 2),
        "steps_per_sec": round(steps_per_sec, 2),
        "elapsed_seconds": round(elapsed, 2),
        "initial_loss": round(losses[0], 4),
        "final_loss": round(losses[-1], 4),
    }


def main() -> int:
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_monotonic = time.monotonic()
    parser = argparse.ArgumentParser(description="TRAIN-00 GaLore Fine-Tuning Bakeoff")
    parser.add_argument("--model-path", default="/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe")
    parser.add_argument("--teacher", default="runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json")
    parser.add_argument("--prompts", default="workloads/gsm8k.jsonl")
    parser.add_argument("--model-revision", default="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
    parser.add_argument("--output", default="runs/research/TRAIN-00B-GALORE-3090-2026-08-25/raw/receipt.json")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    import torch

    repo = ROOT
    out_path = (repo / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== TRAIN-00 GaLore 3090 Fine-Tuning Bakeoff ===", flush=True)

    arms = ["LOKR_PEFT", "GALORE_R16", "FULL_ADAMW"]
    results = {}

    for arm in arms:
        results[arm] = run_training_arm(arm, args.steps, args, repo, torch)

    full_vram = results["FULL_ADAMW"]["peak_vram_gib"]
    galore_vram = results["GALORE_R16"]["peak_vram_gib"]
    vram_savings_pct = ((full_vram - galore_vram) / full_vram) * 100.0

    gates = {
        "galore_vram_savings_ge_30pct": vram_savings_pct >= 30.0,
        "galore_steps_per_sec_ge_2": results["GALORE_R16"]["steps_per_sec"] >= 2.0,
        "galore_converged": results["GALORE_R16"]["final_loss"] < results["GALORE_R16"]["initial_loss"],
    }

    verdict = "PROMOTED" if all(gates.values()) else "REJECTED"

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Codex",
        "results": results,
        "analysis": {
            "vram_savings_pct": round(vram_savings_pct, 2),
            "full_adamw_vram_gib": full_vram,
            "galore_r16_vram_gib": galore_vram,
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
            repo / args.teacher,
            repo / args.prompts,
        ],
        packages=["torch", "transformers", "peft"],
        runtime={
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "model_revision": args.model_revision,
            "seed": args.seed,
            "steps_per_arm": args.steps,
            "peak_metric": "torch.cuda.max_memory_allocated",
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
    print(f"  TRAIN-00 GALORE BAKEOFF VERDICT: {verdict}", flush=True)
    print(f"  GaLore VRAM Savings: {vram_savings_pct:.1f}% (Gate >=30%: {gates['galore_vram_savings_ge_30pct']})")
    print(f"  GaLore Throughput:   {results['GALORE_R16']['steps_per_sec']} steps/s (Gate >=2.0: {gates['galore_steps_per_sec_ge_2']})")
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if final_payload["verdict"] == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
