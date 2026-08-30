#!/usr/bin/env python3
"""Train a compact factorized HYPER-01 generator on physical LoRA targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import time


TARGET_KEY_A = "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight"
TARGET_KEY_B = "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight"


def tensor_hash(tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def factorized_parameter_count(z_dim: int, hidden: int, projection: int, rank: int, output: int) -> int:
    """Exact parameter count, including biases, for the frozen architecture."""
    return (
        z_dim * hidden + hidden
        + hidden * projection + projection
        + projection * rank + rank
        + rank * output + output
    )


def summarize_seed_cosines(rows: list[dict]) -> dict:
    means = [statistics.mean(row["target_cosines"]) for row in rows]
    return {
        "completed_seeds": len(rows),
        "seed_mean_cosines": means,
        "mean_weight_delta_cosine": statistics.mean(means),
        "worst_seed_mean_cosine": min(means),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs=4, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--state-output", type=pathlib.Path, required=True)
    parser.add_argument("--generated-output", type=pathlib.Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260824, 20260825, 20260826, 20260827, 20260828])
    args = parser.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.cuda.reset_peak_memory_stats()
    targets, target_ledger = [], []
    for path_text in args.checkpoints:
        path = pathlib.Path(path_text)
        with safe_open(path, framework="pt", device="cpu") as handle:
            if TARGET_KEY_A not in handle.keys() or TARGET_KEY_B not in handle.keys():
                raise ValueError(f"matched LoRA keys absent: {path}")
            a = handle.get_tensor(TARGET_KEY_A).float().cuda()
            b = handle.get_tensor(TARGET_KEY_B).float().cuda()
        targets.append((a, b))
        target_ledger.append({
            "path": str(path), "a_shape": list(a.shape), "b_shape": list(b.shape),
            "a_sha256": tensor_hash(a), "b_sha256": tensor_hash(b),
            "delta_sha256": tensor_hash(b @ a),
        })
    if len({row["delta_sha256"] for row in target_ledger}) != 4:
        raise ValueError("physical target deltas are not distinct")

    z_dim, hidden, projection = 64, 256, 512
    a_count = targets[0][0].numel()
    out_features = a_count + targets[0][1].numel()

    class FactorizedHyperNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(z_dim, hidden), torch.nn.GELU(),
                torch.nn.Linear(hidden, projection), torch.nn.GELU(),
                torch.nn.Linear(projection, args.rank), torch.nn.GELU(),
                torch.nn.Linear(args.rank, out_features),
            )

        def forward(self, code):
            flat = self.net(code).squeeze(0)
            return flat[:a_count].view_as(targets[0][0]), flat[a_count:].view_as(targets[0][1])

    codes = torch.zeros((4, z_dim), device="cuda")
    codes[torch.arange(4), torch.arange(4)] = 1.0
    state_tensors, generated_tensors, seed_rows, latencies = {}, {}, [], []
    expected_parameters = factorized_parameter_count(z_dim, hidden, projection, args.rank, out_features)

    for seed in args.seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        network = FactorizedHyperNetwork().cuda()
        actual_parameters = sum(parameter.numel() for parameter in network.parameters())
        if actual_parameters != expected_parameters:
            raise AssertionError((actual_parameters, expected_parameters))
        optimizer = torch.optim.AdamW(network.parameters(), lr=5e-3)
        trace = []
        for step in range(args.steps):
            index = step % 4
            optimizer.zero_grad(set_to_none=True)
            generated_a, generated_b = network(codes[index:index + 1])
            target_a, target_b = targets[index]
            loss = torch.nn.functional.mse_loss(generated_a, target_a) + torch.nn.functional.mse_loss(generated_b, target_b)
            loss.backward()
            optimizer.step()
            if step == 0 or (step + 1) % 200 == 0:
                trace.append({"step": step + 1, "target": index, "loss": float(loss.item())})

        network.eval()
        similarities = []
        with torch.inference_mode():
            for index, (target_a, target_b) in enumerate(targets):
                generated_a, generated_b = network(codes[index:index + 1])
                similarities.append(float(torch.nn.functional.cosine_similarity(
                    (generated_b @ generated_a).flatten(), (target_b @ target_a).flatten(), dim=0,
                ).item()))
                generated_tensors[f"seed_{seed}.target_{index}.a"] = generated_a.detach().cpu().contiguous()
                generated_tensors[f"seed_{seed}.target_{index}.b"] = generated_b.detach().cpu().contiguous()
            for _ in range(20):
                network(codes[0:1])
            for _ in range(50):
                for index in range(4):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(); network(codes[index:index + 1]); end.record(); end.synchronize()
                    latencies.append(start.elapsed_time(end))
        for key, value in network.state_dict().items():
            state_tensors[f"seed_{seed}.{key}"] = value.detach().cpu().contiguous()
        seed_rows.append({"seed": seed, "target_cosines": similarities, "training_trace": trace})
        del optimizer, network
        torch.cuda.empty_cache()

    args.state_output.parent.mkdir(parents=True, exist_ok=True)
    args.generated_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_tensors, str(args.state_output))
    save_file(generated_tensors, str(args.generated_output))
    summary = summarize_seed_cosines(seed_rows)
    payload = {
        "schema": "hyper01-factorized-worker-r2", "pid": os.getpid(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": torch.cuda.get_device_name(0), "physical_adapter_targets": 4,
        "distinct_target_deltas": len({row["delta_sha256"] for row in target_ledger}),
        "target_ledger": target_ledger, "seeds": seed_rows, "rank": args.rank,
        "steps_per_seed": args.steps, "generator_parameters": expected_parameters,
        "generator_fp32_storage_mb": expected_parameters * 4 / (1024 ** 2),
        "median_synthesis_latency_ms": statistics.median(latencies),
        "latencies_ms": latencies,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024 ** 2),
        "retained_state_tensors": len(state_tensors),
        "retained_generated_tensors": len(generated_tensors),
        **summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "completed_seeds", "generator_fp32_storage_mb", "mean_weight_delta_cosine",
        "worst_seed_mean_cosine", "median_synthesis_latency_ms")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
