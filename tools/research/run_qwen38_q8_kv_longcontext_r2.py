#!/usr/bin/env python3
"""Serving-chat correction of the Qwen3.8 F16/Q8 long-context crossover."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (  # noqa: E402
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
)
from tools.research import run_qwen38_q8_kv_longcontext as r1  # noqa: E402

TASK_ID = "BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-02"
HOST_INPUTS = {
    "config/research_backlog_admissions/BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-02.json": "5caab3f3e03cdea9b4f64f680236a10d42663c7240ed8a4062b585bf7fa26223",
    "runs/research/BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-02/PRE_REGISTRATION.md": "b63baf39d7bd194cd1a96b2684a5f47f53c4ffdc06feee5aa3cf8fa624d1add5",
    "runs/research/BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-01/raw/receipt.json": "289b17a34e4bb1c298f9e34f8d914c47ffec7641850cc29a58d1992f5a8f2093",
    "runs/research/BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-01/REVIEW.json": "c88f586556e10d9120cf6790f6514b5406d9b4a99ffff564899fd16ee34ef393",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-04/raw/receipt.json": "f94153b21ab3196000b321d06fb79b0b59c3862146de519f05d69cb47d2fa9fe",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-04/REVIEW.json": "d3d230ca8fe27b198ce2170d54f1c95feae6cb53b22d15e14ccce98e727d3e54",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    "tools/research/run_qwen38_q8_kv_longcontext.py": "4197afacf8c2bd9c4f3a96802612afcd41a4ca47ed4f1190284d170cf13419c8",
}


def chat_completion(prompt: str, n_predict: int = 32) -> dict[str, Any]:
    started = time.perf_counter()
    status, body = r1.infra.http_json(
        f"{r1.infra.BASE_URL}/v1/chat/completions",
        {
            "model": "qwen38-q8-longcontext",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n_predict,
            "temperature": 0.0,
            "top_k": 1,
            "seed": r1.SEED,
            "cache_prompt": False,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=1800.0,
    )
    try:
        choice = body["choices"][0]
        content = str(choice["message"].get("content") or "")
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError):
        content = ""
        finish_reason = None
    timings = body.get("timings") or {}
    usage = body.get("usage") or {}
    predicted_n = int(timings.get("predicted_n") or usage.get("completion_tokens") or 0)
    predicted_ms = float(timings.get("predicted_ms") or 0.0)
    throughput = predicted_n * 1000.0 / predicted_ms if predicted_ms > 0 else None
    return {
        "http_status": status,
        "error": body.get("_error"),
        "wall_ms": round((time.perf_counter() - started) * 1000, 3),
        "content": content,
        "finish_reason": finish_reason,
        "timings": timings,
        "usage": usage,
        "prompt_n": int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0),
        "predicted_n": predicted_n,
        "predicted_ms": predicted_ms,
        "throughput_tps": throughput,
        "response": body,
    }


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    monotonic = time.monotonic()
    r1.TASK_ID = TASK_ID
    r1.HOST_INPUTS = HOST_INPUTS
    r1.completion = chat_completion
    r1.execute(outdir)

    raw = outdir / "raw"
    samples = r1.read_jsonl(raw / "samples.jsonl")
    metrics_path = raw / "actual_scores.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["nonempty_content_rate"] = sum(bool(r1.normalize(row["content"])) for row in samples) / len(samples)
    metrics["nondegenerate_timing_rate"] = sum(
        int(row.get("predicted_n") or 0) >= 2
        and float(row.get("predicted_ms") or 0.0) >= 1.0
        and 1.0 <= float(row.get("throughput_tps") or 0.0) <= 1000.0
        for row in samples
    ) / len(samples)
    r1.write_json(metrics_path, metrics)
    controls_path = raw / "treatment_controls.json"
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    controls["request_contract"] = {
        "endpoint": "/v1/chat/completions",
        "message_role": "user",
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
        "content_path": "choices[0].message.content",
        "prompt_token_path": "usage.prompt_tokens",
    }
    r1.write_json(controls_path, controls)
    r1.write_json(raw / "independent_evaluation.json", {"executor_metrics": metrics, "independent_review_pending": True, "claim_boundary": "Qwen3.8 single-slot chat-serving associative retrieval at 8k and 16k only"})

    definitions = {
        "binary_model_identity": ("binary_and_model_identity_verified", "eq", True),
        "cache_treatment_identity": ("explicit_cache_controls_verified", "eq", True),
        "balanced_crossover": ("valid_crossover_blocks", "eq", 4),
        "sample_size": ("recorded_requests", "eq", 48),
        "physical_context": ("requests_within_target_token_bands", "eq", 48),
        "request_integrity": ("successful_response_rate", "eq", 1.0),
        "chat_content": ("nonempty_content_rate", "eq", 1.0),
        "timing_integrity": ("nondegenerate_timing_rate", "eq", 1.0),
        "f16_retrieval": ("f16_exact_recall", "ge", 0.95),
        "q8_retrieval": ("q8_exact_recall", "ge", 0.95),
        "paired_noninferiority": ("paired_bootstrap_ci95_low_q8_minus_f16", "ge", -0.05),
        "throughput_nonregression": ("q8_vs_f16_median_tps_ratio", "ge", 0.9),
        "memory_saving": ("median_vram_saving_mib", "ge", 500.0),
        "service_recovery": ("service_gateway_embedding_restored", "eq", True),
    }
    gates = {name: {"metric": metric, "operator": operator, "threshold": threshold, "actual": metrics[metric], "pass": r1.gate_pass(operator, metrics[metric], threshold)} for name, (metric, operator, threshold) in definitions.items()}
    _, frozen_paths = r1.verify_inputs()
    evidence_files = sorted(path for path in raw.rglob("*") if path.is_file() and path.name != "receipt.json")
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(),
        started_at_utc=started,
        started_monotonic=monotonic,
        input_paths=[*frozen_paths, *evidence_files],
        packages=[],
        runtime={"execution_mode": "qwen38_q8_longcontext_chat_crossover", "blocks": 4, "requests": len(samples), "targets": list(r1.TARGETS), "model": r1.infra.MODEL, "thinking": False},
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(errors)
    evidence = {
        "acceptance_gates": "raw/receipt.json", "binary_identity": "raw/binary_identity.json", "dependency_hashes": "raw/dependency_hashes.json",
        "effective_route": "raw/effective_route.json", "end_to_end_artifact": "raw/end_to_end_artifact.json", "environment": "raw/environment.json",
        "hardware_metrics": "raw/hardware_metrics.json", "independent_evaluation": "raw/independent_evaluation.json", "paired_baseline": "raw/paired_baseline.json",
        "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl", "receipt_fingerprint": "raw/receipt.json", "recovery_state": "raw/recovery_state.json",
        "service_identity": "raw/service_identity.json", "service_maintenance": "raw/service_maintenance.json", "treatment_controls": "raw/treatment_controls.json",
    }
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    r1.write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    comparison = metrics["paired_q8_minus_f16"]
    claim = "QWEN38_Q8_KV_LONGCONTEXT_NONINFERIOR_R2" if not failed else "QWEN38_Q8_KV_LONGCONTEXT_NOT_NONINFERIOR_R2"
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"F16/Q8 recall `{metrics['f16_exact_recall']:.4f}`/`{metrics['q8_exact_recall']:.4f}`; paired Q8-minus-F16 `{comparison['point']:.4f}` with 95% CI `[{comparison['lower_95']:.4f}, {comparison['upper_95']:.4f}]`; "
        f"nonempty/timing-integrity `{metrics['nonempty_content_rate']:.4f}`/`{metrics['nondegenerate_timing_rate']:.4f}`; Q8 throughput ratio `{metrics['q8_vs_f16_median_tps_ratio']:.4f}`; median VRAM saving `{metrics['median_vram_saving_mib']:.1f}` MiB; service restored `{metrics['service_gateway_embedding_restored']}`. Failed gates: `{', '.join(failed) if failed else 'none'}`.\n",
        encoding="utf-8", newline="\n",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    execute(args.outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
