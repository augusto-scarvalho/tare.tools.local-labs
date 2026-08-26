#!/usr/bin/env python3
"""Train HYPER-01 on four physical, matched LoRA module targets."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs=4, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    import torch
    from safetensors import safe_open

    torch.manual_seed(20260824)
    torch.cuda.manual_seed_all(20260824)
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

    z_dim, out_features = 64, targets[0][0].numel() + targets[0][1].numel()

    class HyperNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(z_dim, 256), torch.nn.GELU(),
                torch.nn.Linear(256, 512), torch.nn.GELU(),
                torch.nn.Linear(512, out_features),
            )

        def forward(self, code):
            flat = self.net(code).squeeze(0)
            a_count = targets[0][0].numel()
            return flat[:a_count].view_as(targets[0][0]), flat[a_count:].view_as(targets[0][1])

    codes = torch.zeros((4, z_dim), device="cuda")
    codes[torch.arange(4), torch.arange(4)] = 1.0
    network = HyperNetwork().cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=5e-3)
    trace = []
    for step in range(150):
        index = step % 4
        optimizer.zero_grad()
        generated_a, generated_b = network(codes[index:index + 1])
        target_a, target_b = targets[index]
        loss = torch.nn.functional.mse_loss(generated_a, target_a) + torch.nn.functional.mse_loss(generated_b, target_b)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0:
            trace.append({"step": step + 1, "target": index, "loss": float(loss.item())})

    network.eval()
    similarities = []
    with torch.inference_mode():
        for index, (target_a, target_b) in enumerate(targets):
            generated_a, generated_b = network(codes[index:index + 1])
            similarities.append(float(torch.nn.functional.cosine_similarity(
                (generated_b @ generated_a).flatten(), (target_b @ target_a).flatten(), dim=0,
            ).item()))
        for _ in range(20):
            network(codes[0:1])
        latencies = []
        for _ in range(100):
            for index in range(4):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(); network(codes[index:index + 1]); end.record(); end.synchronize()
                latencies.append(start.elapsed_time(end))

    parameters = sum(parameter.numel() for parameter in network.parameters())
    payload = {
        "schema": "hyper01-real-adapter-worker-v1", "pid": os.getpid(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": torch.cuda.get_device_name(0), "physical_adapter_targets": 4,
        "distinct_target_deltas": len({row["delta_sha256"] for row in target_ledger}),
        "target_ledger": target_ledger, "training_trace": trace,
        "task_codes": codes.cpu().tolist(), "generator_parameters": parameters,
        "generator_vram_overhead_mb": parameters * 4 / (1024 ** 2),
        "target_cosines": similarities, "mean_weight_delta_cosine": statistics.mean(similarities),
        "median_synthesis_latency_ms": statistics.median(latencies),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "physical_adapter_targets", "distinct_target_deltas", "generator_vram_overhead_mb",
        "mean_weight_delta_cosine", "median_synthesis_latency_ms")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
