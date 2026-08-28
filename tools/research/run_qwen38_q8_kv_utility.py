#!/usr/bin/env python3
"""Broad paired correctness test of Qwen3.8 Q8_0 versus F16 KV cache."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import statistics
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.a2_stats import gsm8k_extract, numeric_equal
from tools.analysis.experiment_provenance import (
    build_provenance, canonical_json_sha256, provenance_complete, sha256_file,
)
from tools.research import run_qwen38_kv_precision as base

TASK_ID = "BACKLOG-QWEN38-Q8-KV-UTILITY-01"
PANEL_HASH = "78338489c487181cc63b42f0f26c90e3068c1bb6cef789257ab522258249a786"
BOOTSTRAP_SEED = 2026082712
BOOTSTRAP_REPLICATES = 20_000
BLOCKS = (
    {"id": "utility_f16", "arm": "f16", "cache": "f16"},
    {"id": "utility_q8", "arm": "q8", "cache": "q8_0"},
)
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-QWEN38-Q8-KV-UTILITY-01.json"
PREREGISTRATION = ROOT / "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-01/PRE_REGISTRATION.md"
SOURCE = ROOT / "runs/research/BACKLOG-QWEN38-KV-PRECISION-02"
EXPECTED_HASHES = {
    ADMISSION: "897c4bfc3820b6376e9124681d1f843b2eee4c13a807f331e65111500da3f139",
    PREREGISTRATION: "8b8d76d73b43fad7f39860452d8701d4ff9e9c91150c816e905efac7b6b4b60f",
    SOURCE / "raw/receipt.json": "aba1cc2685f5b74ab01a16937b13d7c063898e9cf2df6ac97bb30564a4074bd7",
    SOURCE / "raw/samples.jsonl": "24151e0077e34172f25eafdba5ea24377cb4293ae225bca1b90b85edd10be10a",
    SOURCE / "raw/actual_scores.json": "17369a2faaa67899d249772dc96c03f6ef2fb94b0c37b9eea990bcbc29b50b37",
    ROOT / "tools/research/run_qwen38_kv_precision.py": "84da3a32cccf309a6cc7106e25a7afb282f9c57acc4dbd3ab2c6c7694a22baf9",
    ROOT / "tools/research/run_qwen38_kv_precision_r2.py": "fdf2d469a652381821af23d7d4612898d9ec65d1f4fae5323ee9e50efa7d8158",
    ROOT / "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    ROOT / "workloads/gsm8k.jsonl": "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
}


def verify_sources() -> tuple[dict[str, Any], list[pathlib.Path]]:
    own: dict[str, Any] = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"source mismatch: {path}: {actual} != {expected}")
        own[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": actual}
    physical, base_paths = base.verify_inputs()
    return {"utility_sources": own, "physical_runtime": physical}, [*EXPECTED_HASHES, *base_paths]


def panel() -> list[dict[str, Any]]:
    rows = base.read_jsonl(ROOT / "workloads/gsm8k.jsonl")[32:160]
    ids = [row["task_id"] for row in rows]
    source_ids = {row["task_id"] for row in base.read_jsonl(SOURCE / "raw/samples.jsonl")}
    if len(rows) != 128 or canonical_json_sha256(ids) != PANEL_HASH or set(ids) & source_ids:
        raise ValueError("utility panel is not frozen and disjoint from Q8 R2")
    return rows


def paired_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    f16 = {row["task_id"]: row for row in rows if row["arm"] == "f16"}
    q8 = {row["task_id"]: row for row in rows if row["arm"] == "q8"}
    if len(f16) != 128 or set(f16) != set(q8):
        raise ValueError("paired utility coverage is incomplete")
    ids = sorted(f16)
    differences = [int(q8[key]["correct"]) - int(f16[key]["correct"]) for key in ids]
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = [sum(differences[rng.randrange(128)] for _ in range(128)) / 128
                 for _ in range(BOOTSTRAP_REPLICATES)]
    estimates.sort()
    return {"point": statistics.mean(differences), "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "lower_95": estimates[int(0.025 * BOOTSTRAP_REPLICATES)],
            "upper_95": estimates[int(0.975 * BOOTSTRAP_REPLICATES)],
            "q8_only_correct": sum(value == 1 for value in differences),
            "f16_only_correct": sum(value == -1 for value in differences)}


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw, logs, finalized = outdir / "raw", outdir / "raw/logs", outdir / "raw/finalized"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    logs.mkdir(parents=True)
    finalized.mkdir(parents=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()
    frozen, frozen_paths = verify_sources()
    tasks = panel()
    base.write_json(raw / "binary_identity.json", frozen)
    base.write_json(raw / "dataset_hashes.json", {"task_ids": [row["task_id"] for row in tasks],
                    "task_ids_sha256": canonical_json_sha256([row["task_id"] for row in tasks]),
                    "disjoint_from_q8_r2": True})
    initial_service = base.infra.unit_state(base.infra.PERSISTENT_UNIT)
    initial_gateway = base.infra.gateway_status()
    initial_model = str(initial_gateway.get("current_model"))
    embed_status, embed_body = base.infra.health(base.infra.EMBED_URL)
    if initial_service["active_state"] != "active" or embed_status != 200 or embed_body.get("status") != "ok":
        raise RuntimeError("persistent service baseline unhealthy")
    samples: list[dict[str, Any]] = []
    block_records: list[dict[str, Any]] = []
    restoration: dict[str, Any] = {}
    error: Exception | None = None
    try:
        base.infra.systemctl("stop", base.infra.PERSISTENT_UNIT)
        base.infra.wait_unit(base.infra.PERSISTENT_UNIT, active=False)
        occupied, body = base.infra.health(base.infra.BASE_URL)
        if occupied is not None:
            raise RuntimeError(f"temporary endpoint occupied: {occupied} {body}")
        for block in BLOCKS:
            unit = ""
            try:
                unit, launch = base.start_block(block)
                warmups = [base.completion(f"Discarded utility warmup {index}: compute {19 + index} plus {37 + index}.", 64)
                           for index in range(4)]
                if any(row["http_status"] != 200 for row in warmups):
                    raise RuntimeError(f"warmup failed in {block['id']}")
                gpu = base.infra.gpu_state()
                consecutive = 0
                for index, task in enumerate(tasks):
                    prompt = "Solve carefully and end with #### followed by the numeric answer.\n" + str(task["prompt"])
                    response = base.completion(prompt)
                    extracted = gsm8k_extract(response["content"])
                    correct = numeric_equal(extracted, str(task["answer"]))
                    row = {"block_id": block["id"], "arm": block["arm"], "cache": block["cache"],
                           "index": index, "task_id": task["task_id"], "gold": str(task["answer"]),
                           "extracted": extracted, "correct": bool(correct), **response}
                    base.append_jsonl(raw / "samples.jsonl", row)
                    samples.append(row)
                    consecutive = consecutive + 1 if response["http_status"] != 200 or response["error"] else 0
                    if consecutive >= 3:
                        raise RuntimeError(f"three consecutive failures in {block['id']}")
                record = {**block, "block_id": block["id"], "complete": True, "launch": launch,
                          "warmups": warmups, "gpu": gpu, "recorded": len(tasks)}
                block_records.append(record)
                base.append_jsonl(raw / "blocks.jsonl", record)
                base.write_json(finalized / f"{block['arm']}.json", {"arm": block["arm"],
                                "recorded": len(tasks), "correct": sum(row["correct"] for row in samples if row["arm"] == block["arm"])})
            finally:
                if unit:
                    journal = base.infra.wsl("journalctl", "-u", unit, "--no-pager", "-o", "short-iso", "-n", "5000", root=True)["stdout"]
                    (logs / f"{unit}.log").write_text(journal, encoding="utf-8")
                    base.infra.wsl("systemctl", "stop", unit, root=True, timeout=180.0)
                    try:
                        base.infra.wait_unit(unit, active=False, timeout_seconds=180.0)
                    except RuntimeError:
                        pass
            boundary, _ = base.infra.health(base.infra.EMBED_URL)
            if boundary != 200:
                raise RuntimeError(f"embedding unhealthy after {block['id']}")
    except Exception as caught:
        error = caught
    finally:
        try:
            if base.infra.unit_state(base.infra.PERSISTENT_UNIT)["active_state"] != "active":
                base.infra.systemctl("start", base.infra.PERSISTENT_UNIT)
            base.infra.wait_health(base.infra.GATEWAY_URL)
            restored = base.infra.restore_model(initial_model)
            final_service = base.infra.unit_state(base.infra.PERSISTENT_UNIT)
            final_embed, final_embed_body = base.infra.health(base.infra.EMBED_URL)
            restoration = {"gateway": restored, "service": final_service,
                           "embedding": {"http_status": final_embed, "body": final_embed_body},
                           "initial_model_restored": restored.get("current_model") == initial_model}
        except Exception as restore_error:
            restoration = {"error": f"{type(restore_error).__name__}: {restore_error}", "initial_model_restored": False}
            if error is None:
                error = restore_error
        base.write_json(raw / "recovery_state.json", restoration)
    if error:
        raise error

    comparison = paired_bootstrap(samples)
    f16 = [row for row in samples if row["arm"] == "f16"]
    q8 = [row for row in samples if row["arm"] == "q8"]
    f16_accuracy = sum(row["correct"] for row in f16) / 128
    q8_accuracy = sum(row["correct"] for row in q8) / 128
    f16_tps = statistics.median(row["throughput_tps"] for row in f16 if row["throughput_tps"] is not None)
    q8_tps = statistics.median(row["throughput_tps"] for row in q8 if row["throughput_tps"] is not None)
    f16_vram = float(next(row["gpu"]["memory.used"] for row in block_records if row["arm"] == "f16"))
    q8_vram = float(next(row["gpu"]["memory.used"] for row in block_records if row["arm"] == "q8"))
    metrics = {"q8_r2_sources_and_artifacts_verified": True,
               "fresh_panel_disjoint_from_q8_r2": True,
               "explicit_cache_controls_verified": all(row["launch"]["cache_tokens"] == [row["cache"], row["cache"]] for row in block_records),
               "recorded_requests": len(samples),
               "successful_response_rate": sum(row["http_status"] == 200 and not row["error"] for row in samples) / len(samples),
               "f16_accuracy": f16_accuracy, "q8_accuracy": q8_accuracy,
               "f16_minus_q8_accuracy": f16_accuracy - q8_accuracy,
               "paired_q8_minus_f16_accuracy": comparison,
               "paired_bootstrap_95ci_lower_q8_minus_f16_accuracy": comparison["lower_95"],
               "extracted_answer_parity_rate": sum(left["extracted"] == right["extracted"] for left, right in zip(f16, q8)) / 128,
               "f16_median_tps": f16_tps, "q8_median_tps": q8_tps,
               "q8_vs_f16_throughput_ratio": q8_tps / f16_tps,
               "f16_vram_mib": f16_vram, "q8_vram_mib": q8_vram,
               "vram_saving_mib": f16_vram - q8_vram,
               "service_and_embedding_restored": restoration.get("initial_model_restored") is True and restoration.get("embedding", {}).get("http_status") == 200}
    base.write_json(raw / "actual_scores.json", metrics)
    base.write_json(raw / "effective_route.json", {"blocks": block_records})
    base.write_json(raw / "environment.json", {"gpu": base.infra.gpu_state(), "wsl_distro": base.infra.WSL_DISTRO})
    base.write_json(raw / "hardware_metrics.json", {key: metrics[key] for key in (
        "f16_median_tps", "q8_median_tps", "q8_vs_f16_throughput_ratio",
        "f16_vram_mib", "q8_vram_mib", "vram_saving_mib")})
    base.write_json(raw / "independent_evaluation.json", {"paired_correctness": comparison,
                    "f16_accuracy": f16_accuracy, "q8_accuracy": q8_accuracy,
                    "output_parity_descriptive_only": metrics["extracted_answer_parity_rate"]})
    base.write_json(raw / "paired_baseline.json", {"baseline": "f16", "treatment": "q8_0",
                    "paired_tasks": 128, "comparison": comparison})
    base.write_json(raw / "service_identity.json", {"initial_service": initial_service,
                    "initial_gateway": initial_gateway, "restoration": restoration})
    base.write_json(raw / "service_maintenance.json", {"persistent_service_stopped_via_systemd": True,
                    "embedding_service_stopped": False, "restoration": restoration})
    source_receipt = json.loads((SOURCE / "raw/receipt.json").read_text(encoding="utf-8"))
    base.write_json(raw / "source_execution_receipt.json", {"source_task_id": "BACKLOG-QWEN38-KV-PRECISION-02",
                    "receipt_sha256": sha256_file(SOURCE / "raw/receipt.json"),
                    "receipt_fingerprint": source_receipt["receipt_fingerprint"]})
    base.write_json(raw / "treatment_controls.json", {"order": ["f16", "q8_0"],
                    "server_common": base.SERVER_COMMON, "warmups_per_block": 4,
                    "tasks_per_block": 128, "output_parity_is_gate": False})

    definitions = {
        "source_integrity": ("q8_r2_sources_and_artifacts_verified", "eq", True),
        "panel_isolation": ("fresh_panel_disjoint_from_q8_r2", "eq", True),
        "treatment_identity": ("explicit_cache_controls_verified", "eq", True),
        "request_coverage": ("recorded_requests", "eq", 256),
        "request_integrity": ("successful_response_rate", "eq", 1.0),
        "utility_noninferiority": ("paired_bootstrap_95ci_lower_q8_minus_f16_accuracy", "gt", -0.05),
        "quality_regression": ("f16_minus_q8_accuracy", "le", 0.03),
        "physical_memory_saving": ("vram_saving_mib", "ge", 500),
        "throughput_non_regression": ("q8_vs_f16_throughput_ratio", "ge", 0.95),
        "service_recovery": ("service_and_embedding_restored", "eq", True),
    }
    gates: dict[str, Any] = {}
    for gate_id, (metric, operator, threshold) in definitions.items():
        actual = metrics[metric]
        passed = actual == threshold if operator == "eq" else actual > threshold if operator == "gt" else actual >= threshold if operator == "ge" else actual <= threshold
        gates[gate_id] = {"metric": metric, "operator": operator, "threshold": threshold, "actual": actual, "pass": passed}
    evidence = {"acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
                "binary_identity": "raw/binary_identity.json", "block_logs": "raw/logs",
                "dataset_hashes": "raw/dataset_hashes.json", "effective_route": "raw/effective_route.json",
                "environment": "raw/environment.json", "hardware_metrics": "raw/hardware_metrics.json",
                "independent_evaluation": "raw/independent_evaluation.json", "paired_baseline": "raw/paired_baseline.json",
                "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl",
                "receipt_fingerprint": "raw/receipt.json", "recovery_state": "raw/recovery_state.json",
                "service_identity": "raw/service_identity.json", "service_maintenance": "raw/service_maintenance.json",
                "source_execution_receipt": "raw/source_execution_receipt.json", "treatment_controls": "raw/treatment_controls.json"}
    evidence_files = sorted(path for path in raw.rglob("*") if path.is_file())
    provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(), started_at_utc=started,
        started_monotonic=mono, input_paths=[*frozen_paths, *evidence_files], packages=[],
        runtime={"execution_mode": "physical_q8_utility_noninferiority", "host_pid": os.getpid(),
                 "blocks": 2, "requests": len(samples), "timing_is_evidence": False})
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(f"incomplete provenance: {errors}")
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID,
               "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    base.write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = "QWEN38_Q8_KV_UTILITY_NONINFERIOR_R1" if not failed else "QWEN38_Q8_KV_UTILITY_NOT_NONINFERIOR_R1"
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"F16/Q8 accuracy `{f16_accuracy:.4f}`/`{q8_accuracy:.4f}`; Q8-minus-F16 "
        f"paired-bootstrap 95% interval `[{comparison['lower_95']:.4f}, {comparison['upper_95']:.4f}]`. "
        f"Q8 saved `{metrics['vram_saving_mib']:.1f}` MiB at `{metrics['q8_vs_f16_throughput_ratio']:.4f}x` "
        f"throughput; literal answer parity `{metrics['extracted_answer_parity_rate']:.4f}` is descriptive only. "
        f"Failed gates: `{', '.join(failed) if failed else 'none'}`.\n", encoding="utf-8", newline="\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        assert len(panel()) == 128
        assert [row["cache"] for row in BLOCKS] == ["f16", "q8_0"]
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
