#!/usr/bin/env python3
"""SLX-05: Megakernel and Launch-Overhead Oracle on RTX 3090.

Quantifies the exact CPU driver launch bottleneck vs GPU kernel compute time
for sub-3B models (Qwen3.5-0.8B) during single-token autoregressive decode,
benchmarking Eager mode vs captured CUDA Graphs.
"""
from __future__ import annotations

import argparse
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


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _snapshot_cache_state(cache) -> list[dict]:
    """Clone every tensor that a one-token hybrid-cache decode can mutate."""
    snapshot = []
    for layer in cache.layers:
        layer_state = {}
        for attribute in ("keys", "values", "cumulative_length"):
            value = getattr(layer, attribute, None)
            if hasattr(value, "detach"):
                layer_state[attribute] = value.detach().clone()
        for attribute in ("conv_states", "recurrent_states"):
            values = getattr(layer, attribute, None)
            if isinstance(values, dict):
                layer_state[attribute] = {
                    key: value.detach().clone()
                    for key, value in values.items()
                    if hasattr(value, "detach")
                }
        snapshot.append(layer_state)
    return snapshot


def _restore_cache_state(cache, snapshot: list[dict]) -> None:
    """Restore a hybrid StaticCache without changing any tensor address."""
    if len(cache.layers) != len(snapshot):
        raise ValueError("cache topology changed after snapshot")
    for layer, layer_state in zip(cache.layers, snapshot, strict=True):
        for attribute in ("keys", "values", "cumulative_length"):
            if attribute in layer_state:
                getattr(layer, attribute).copy_(layer_state[attribute])
        for attribute in ("conv_states", "recurrent_states"):
            for key, value in layer_state.get(attribute, {}).items():
                getattr(layer, attribute)[key].copy_(value)


def profile_cell(model, tokenizer, batch_size: int, seq_len: int, iterations: int, torch, StaticCache) -> dict:
    # Transformers' hybrid StaticCache advances an internal counter and mutates
    # recurrent states. Freeze the post-prefill state and restore it outside
    # every timed region so every observation is the same one-token decode.
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device="cuda", dtype=torch.long)
    past_key_values = StaticCache(config=model.config, max_cache_len=seq_len + 1)
    prefill_position = torch.arange(seq_len, device="cuda", dtype=torch.long)
    with torch.inference_mode():
        _ = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            cache_position=prefill_position,
            use_cache=True,
        )
    prefill_snapshot = _snapshot_cache_state(past_key_values)

    next_token = torch.randint(0, tokenizer.vocab_size, (batch_size, 1), device="cuda", dtype=torch.long)
    decode_position = torch.tensor([seq_len], device="cuda", dtype=torch.long)

    with torch.inference_mode():
        for _ in range(20):
            _restore_cache_state(past_key_values, prefill_snapshot)
            eager_reference = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                cache_position=decode_position,
                use_cache=True,
            )
    torch.cuda.synchronize()
    reference_logits = eager_reference.logits[:, -1, :].detach().clone()

    # Benchmark eager fixed-cache decode. Wall and CUDA-event samples remain
    # paired; only wall time is used for the end-to-end speedup verdict.
    eager_submit_times = []
    eager_gpu_times = []
    eager_total_times = []
    eager_host_gap_times = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        for _ in range(iterations):
            _restore_cache_state(past_key_values, prefill_snapshot)
            t0 = time.perf_counter()
            start_event.record()
            eager_out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                cache_position=decode_position,
                use_cache=True,
            )
            t1 = time.perf_counter()
            end_event.record()
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            submit_ms = (t1 - t0) * 1000.0
            gpu_ms = start_event.elapsed_time(end_event)
            wall_ms = (t2 - t0) * 1000.0
            eager_submit_times.append(submit_ms)
            eager_gpu_times.append(gpu_ms)
            eager_total_times.append(wall_ms)
            eager_host_gap_times.append(max(0.0, wall_ms - gpu_ms))

    eager_submit_med = _percentile(eager_submit_times, 0.5)
    eager_gpu_med = _percentile(eager_gpu_times, 0.5)
    eager_total_med = _percentile(eager_total_times, 0.5)

    graph_supported = False
    graph_gpu_med = None
    graph_total_med = None
    speedup_ratio = None
    semantic_max_abs_diff = None
    graph_error = None

    try:
        static_input = next_token.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(5):
                _restore_cache_state(past_key_values, prefill_snapshot)
                _ = model(
                    input_ids=static_input,
                    past_key_values=past_key_values,
                    cache_position=decode_position,
                    use_cache=True,
                )
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.inference_mode():
            _restore_cache_state(past_key_values, prefill_snapshot)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.inference_mode():
            graph_out = model(
                input_ids=static_input,
                past_key_values=past_key_values,
                cache_position=decode_position,
                use_cache=True,
            )

        graph_gpu_times = []
        graph_total_times = []
        for _ in range(iterations):
            with torch.inference_mode():
                _restore_cache_state(past_key_values, prefill_snapshot)
            t0 = time.perf_counter()
            start_event.record()
            g.replay()
            end_event.record()
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            graph_gpu_times.append(start_event.elapsed_time(end_event))
            graph_total_times.append((t1 - t0) * 1000.0)

        semantic_max_abs_diff = (
            graph_out.logits[:, -1, :].float() - reference_logits.float()
        ).abs().max().item()
        graph_gpu_med = _percentile(graph_gpu_times, 0.5)
        graph_total_med = _percentile(graph_total_times, 0.5)
        graph_supported = True

        speedup_ratio = eager_total_med / graph_total_med if graph_total_med > 0 else 1.0
    except Exception as e:
        graph_error = repr(e)
        print(f"CUDA Graph capture failure for (B={batch_size}, L={seq_len}): {e}", flush=True)

    del input_ids, next_token, past_key_values, prefill_snapshot, eager_out, eager_reference
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "cache_type": "StaticCache",
        "cache_state_policy": "restore_post_prefill_snapshot_before_every_observation",
        "decode_position": seq_len,
        "iterations": iterations,
        "eager_host_submit_p50_ms": eager_submit_med,
        "eager_cuda_event_p50_ms": eager_gpu_med,
        "eager_wall_p50_ms": eager_total_med,
        "eager_wall_p95_ms": _percentile(eager_total_times, 0.95),
        "eager_host_gap_p50_ms": _percentile(eager_host_gap_times, 0.5),
        "eager_tokens_per_sec": batch_size * 1000.0 / eager_total_med if eager_total_med > 0 else 0,
        "cuda_graph_supported": graph_supported,
        "cuda_graph_error": graph_error,
        "cuda_graph_event_p50_ms": graph_gpu_med,
        "cuda_graph_wall_p50_ms": graph_total_med,
        "cuda_graph_wall_p95_ms": _percentile(graph_total_times, 0.95) if graph_supported else None,
        "cuda_graph_tokens_per_sec": (batch_size * 1000.0 / graph_total_med) if (graph_total_med and graph_total_med > 0) else None,
        "wall_speedup_ratio": speedup_ratio,
        "semantic_max_abs_logit_diff": semantic_max_abs_diff,
    }


def main() -> int:
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_monotonic = time.monotonic()
    parser = argparse.ArgumentParser(description="SLX-05 Launch-Overhead Oracle")
    parser.add_argument("--model-path", default="/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe")
    parser.add_argument("--model-revision", default="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", default="runs/research/SLX-05D-CUDA-GRAPH-REPLAY-2026-08-25/raw/receipt.json")
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== SLX-05 Launch-Overhead Oracle ===", flush=True)
    print(f"Loading model from {args.model_path}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map={"": "cuda"},
        attn_implementation="sdpa")
    model.eval()

    layers_count = len(model.model.layers) if hasattr(model, "model") and hasattr(model.model, "layers") else 24
    print(f"Model loaded: {layers_count} layers. Running matrix...", flush=True)

    matrix_results = []
    matrix_cells = [
        (1, 128),
        (1, 512),
        (1, 2048),
        (2, 512),
        (4, 512),
    ]

    for b, l in matrix_cells:
        print(f"\n--- Profiling Cell: Batch={b}, Context={l} ---", flush=True)
        res = profile_cell(model, tokenizer, b, l, args.iterations, torch, StaticCache)
        matrix_results.append(res)
        print(f"  Eager wall p50: {res['eager_wall_p50_ms']:.3f} ms | CUDA event p50: {res['eager_cuda_event_p50_ms']:.3f} ms ({res['eager_tokens_per_sec']:.1f} t/s)")
        if res['cuda_graph_supported']:
            print(f"  CUDA Graph wall p50: {res['cuda_graph_wall_p50_ms']:.3f} ms ({res['cuda_graph_tokens_per_sec']:.1f} t/s) | Wall speedup: {res['wall_speedup_ratio']:.2f}x | Max logit diff: {res['semantic_max_abs_logit_diff']:.6f}")

    b1_speedups = [
        r["wall_speedup_ratio"] for r in matrix_results
        if r["batch_size"] == 1 and r["wall_speedup_ratio"] is not None
    ]
    median_b1_speedup = _percentile(b1_speedups, 0.5) if b1_speedups else 0.0
    gates = {
        "all_cells_graph_supported": all(r["cuda_graph_supported"] for r in matrix_results),
        "all_cells_semantic_parity": all(
            r["semantic_max_abs_logit_diff"] is not None
            and r["semantic_max_abs_logit_diff"] <= 1e-2
            for r in matrix_results
        ),
        "median_batch1_wall_speedup_ge_1_15x": median_b1_speedup >= 1.15,
        "fixed_static_cache_position": all(
            r["cache_type"] == "StaticCache"
            and r["cache_state_policy"] == "restore_post_prefill_snapshot_before_every_observation"
            for r in matrix_results
        ),
    }
    verdict = "QUALIFIED_CUDA_GRAPH_REPLAY" if all(gates.values()) else "REJECTED_OR_UNVERIFIED"

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Codex",
        "gpu": torch.cuda.get_device_name(0),
        "model": args.model_path,
        "model_revision": args.model_revision,
        "layers_count": layers_count,
        "iterations_per_cell": args.iterations,
        "matrix_results": matrix_results,
        "summary": {
            "median_b1_wall_speedup_ratio": median_b1_speedup,
            "gates": gates,
            "verdict": verdict,
            "interpretation": (
                "This receipt measures fixed-cache CUDA Graph replay versus eager wall time. "
                "It does not attribute the difference exclusively to CUDA driver launch overhead "
                "and does not establish a persistent-megakernel ceiling."
            ),
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
            "transformers_version": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "model_revision": args.model_revision,
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
    print(f"  SLX-05D ORACLE VERDICT: {final_payload['summary']['verdict']}", flush=True)
    print(f"  Median Batch=1 Wall Speedup: {median_b1_speedup:.2f}x", flush=True)
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if final_payload["summary"]["verdict"] == "QUALIFIED_CUDA_GRAPH_REPLAY" else 1


if __name__ == "__main__":
    sys.exit(main())
