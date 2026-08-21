#!/usr/bin/env python3
"""RNN-09 RetNet parallel/recurrent/chunkwise retention qualification."""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import statistics
import time
from datetime import datetime, timezone

import torch


def parallel_retention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       gamma: float) -> torch.Tensor:
    length = q.shape[1]
    positions = torch.arange(length, device=q.device)
    distance = positions[:, None] - positions[None, :]
    decay = torch.where(distance >= 0,
                        torch.as_tensor(gamma, dtype=q.dtype, device=q.device) ** distance,
                        torch.zeros((), dtype=q.dtype, device=q.device))
    scores = torch.einsum("btd,bsd->bts", q, k) * decay
    return torch.einsum("bts,bsv->btv", scores, v)


def recurrent_retention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        gamma: float, state: torch.Tensor | None = None
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, d_key = q.shape
    d_value = v.shape[-1]
    if state is None:
        state = torch.zeros(batch, d_key, d_value, dtype=q.dtype, device=q.device)
    outputs = []
    for index in range(q.shape[1]):
        state = gamma * state + torch.einsum("bd,bv->bdv", k[:, index], v[:, index])
        outputs.append(torch.einsum("bd,bdv->bv", q[:, index], state))
    return torch.stack(outputs, dim=1), state


def chunkwise_retention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        gamma: float, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    state, outputs = None, []
    for start in range(0, q.shape[1], chunk_size):
        stop = min(q.shape[1], start + chunk_size)
        output, state = recurrent_retention(q[:, start:stop], k[:, start:stop],
                                            v[:, start:stop], gamma, state)
        outputs.append(output)
    return torch.cat(outputs, dim=1), state


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def timed(fn, rounds: int = 5) -> float:
    fn()
    values = []
    for _ in range(rounds):
        started = time.perf_counter()
        fn()
        values.append((time.perf_counter() - started) * 1000)
    return statistics.median(values)


def run(seed: int) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    gamma = 0.97
    parity = []
    for length in (1, 7, 64, 513):
        q = torch.randn(2, length, 16, dtype=torch.float64)
        k = torch.randn(2, length, 16, dtype=torch.float64)
        v = torch.randn(2, length, 16, dtype=torch.float64)
        parallel = parallel_retention(q, k, v, gamma)
        recurrent, _ = recurrent_retention(q, k, v, gamma)
        parity.append({"length": length, "max_abs": max_abs(parallel, recurrent)})

    q = torch.randn(2, 513, 16, dtype=torch.float64)
    k = torch.randn(2, 513, 16, dtype=torch.float64)
    v = torch.randn(2, 513, 16, dtype=torch.float64)
    recurrent, final_state = recurrent_retention(q, k, v, gamma)
    chunk_parity = []
    for chunk_size in (1, 3, 16, 128):
        chunked, chunk_state = chunkwise_retention(q, k, v, gamma, chunk_size)
        chunk_parity.append({"chunk_size": chunk_size,
                             "output_max_abs": max_abs(recurrent, chunked),
                             "state_max_abs": max_abs(final_state, chunk_state)})

    split = 237
    first, state = recurrent_retention(q[:, :split], k[:, :split], v[:, :split], gamma)
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    restored_state = torch.load(buffer, weights_only=True)
    second, restored_final = recurrent_retention(q[:, split:], k[:, split:], v[:, split:],
                                                  gamma, restored_state)
    restored_output = torch.cat([first, second], dim=1)
    save_reload = {"serialized_bytes": len(buffer.getvalue()),
                   "state_bit_exact": torch.equal(state, restored_state),
                   "output_bit_exact": torch.equal(recurrent, restored_output),
                   "final_state_bit_exact": torch.equal(final_state, restored_final)}

    # Batch processing must equal independent processing for each sequence.
    independent = [recurrent_retention(q[index:index + 1], k[index:index + 1],
                                       v[index:index + 1], gamma)[0] for index in range(2)]
    batch_isolation = {"max_abs": max(max_abs(recurrent[index:index + 1], independent[index])
                                      for index in range(2))}

    # State ownership: carrying A into B must be observably different; reset must recover B.
    qa, ka, va = q[:1, :64], k[:1, :64], v[:1, :64]
    qb, kb, vb = q[1:2, :64], k[1:2, :64], v[1:2, :64]
    _, state_a = recurrent_retention(qa, ka, va, gamma)
    standalone_b, _ = recurrent_retention(qb, kb, vb, gamma)
    leaked_b, _ = recurrent_retention(qb, kb, vb, gamma, state_a)
    reset_b, _ = recurrent_retention(qb, kb, vb, gamma, None)
    lifecycle = {"leak_detected_max_abs": max_abs(standalone_b, leaked_b),
                 "reset_recovery_bit_exact": torch.equal(standalone_b, reset_b)}

    # Autograd identity for q/k/v.
    base = [torch.randn(1, 23, 8, dtype=torch.float64) for _ in range(3)]
    pvars = [tensor.clone().requires_grad_() for tensor in base]
    rvars = [tensor.clone().requires_grad_() for tensor in base]
    parallel_retention(*pvars, gamma).square().mean().backward()
    recurrent_retention(*rvars, gamma)[0].square().mean().backward()
    gradient = {name: max_abs(p.grad, r.grad) for name, p, r in zip(
        ("q", "k", "v"), pvars, rvars)}

    long_q = torch.randn(1, 4096, 16, dtype=torch.float32)
    long_k = torch.randn(1, 4096, 16, dtype=torch.float32)
    long_v = torch.randn(1, 4096, 16, dtype=torch.float32)
    long_output, long_state = recurrent_retention(long_q, long_k, long_v, 0.99)
    stability = {"length": 4096, "output_finite": bool(torch.isfinite(long_output).all()),
                 "state_finite": bool(torch.isfinite(long_state).all()),
                 "state_norm": float(long_state.norm())}

    timing = []
    for length in (64, 128, 256, 512, 1024):
        tq = torch.randn(1, length, 16, dtype=torch.float32)
        tk = torch.randn(1, length, 16, dtype=torch.float32)
        tv = torch.randn(1, length, 16, dtype=torch.float32)
        timing.append({"length": length,
                       "parallel_ms_median": timed(lambda: parallel_retention(tq, tk, tv, gamma), 3),
                       "recurrent_python_ms_median": timed(
                           lambda: recurrent_retention(tq, tk, tv, gamma), 3),
                       "parallel_score_elements": length * length,
                       "recurrent_state_elements": 16 * 16})

    gates = {
        "parallel_recurrent": max(row["max_abs"] for row in parity) <= 1e-10,
        "chunkwise": max(max(row["output_max_abs"], row["state_max_abs"])
                         for row in chunk_parity) <= 1e-10,
        "save_reload": all(save_reload[key] for key in (
            "state_bit_exact", "output_bit_exact", "final_state_bit_exact")),
        "batch_isolation": batch_isolation["max_abs"] == 0.0,
        "state_lifecycle": lifecycle["leak_detected_max_abs"] > 0.0
                           and lifecycle["reset_recovery_bit_exact"],
        "gradient_parity": max(gradient.values()) <= 1e-9,
        "fp32_long_finite": stability["output_finite"] and stability["state_finite"],
    }
    return {"campaign": "RNN-09-retnet-retention", "timestamp": datetime.now(timezone.utc).isoformat(),
            "scope": "mechanism microbenchmark; not official checkpoint reproduction",
            "environment": {"torch": torch.__version__, "device": "cpu", "threads": 1,
                            "seed": seed, "gamma": gamma},
            "parity": parity, "chunk_parity": chunk_parity, "save_reload": save_reload,
            "batch_isolation": batch_isolation, "state_lifecycle": lifecycle,
            "gradient_parity": gradient, "stability": stability, "timing_descriptive": timing,
            "gates": gates, "qualified": all(gates.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("runs/rnn/RNN-09-retnet-retention/results.json"))
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
