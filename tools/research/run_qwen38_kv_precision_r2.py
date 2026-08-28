#!/usr/bin/env python3
"""Run the final physical Qwen3.8 F16/Q8_0 KV-cache crossover."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (
    build_provenance, canonical_json_sha256, provenance_complete, sha256_file,
)
from tools.research import run_qwen38_kv_precision as r1

TASK_ID = "BACKLOG-QWEN38-KV-PRECISION-02"
PREREG_SHA256 = "d19efbe25c860b17dfec323e858a80e47eb4989cee640e4a5e84af61d643c29f"
SOURCE_HASHES = {
    "config/research_backlog_admissions/BACKLOG-QWEN38-KV-PRECISION-02.json": "d93073fa3d6cdebacb02c651bd4c73a4447bdfdcde2c413b7dafa3aa5f754e90",
    "runs/research/BACKLOG-QWEN38-KV-PRECISION-01/raw/receipt.json": "9e068c702f431a8fc6cc34a393f0daee6ac5a02d3329d10233747f451f7a2652",
    "runs/research/BACKLOG-QWEN38-KV-PRECISION-01/raw/samples.jsonl": "bec190f0b731b5931cd14bfcaf4808828ad50449fc3f209827e769b0102c4a5d",
    "tools/research/run_qwen38_kv_precision.py": "84da3a32cccf309a6cc7106e25a7afb282f9c57acc4dbd3ab2c6c7694a22baf9",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    "workloads/gsm8k.jsonl": "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
}
BLOCKS = (
    {"id": "b1_f16", "arm": "f16", "cache": "f16", "pair": 0},
    {"id": "b2_q8", "arm": "q4", "cache": "q8_0", "pair": 0},
    {"id": "b3_q8", "arm": "q4", "cache": "q8_0", "pair": 1},
    {"id": "b4_f16", "arm": "f16", "cache": "f16", "pair": 1},
)


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def repeat_parity(rows: list[dict[str, Any]], arm: str) -> float:
    first = {row["task_id"]: row["extracted"] for row in rows
             if row["arm"] == arm and row["pair"] == 0}
    second = {row["task_id"]: row["extracted"] for row in rows
              if row["arm"] == arm and row["pair"] == 1}
    if len(first) != 32 or set(first) != set(second):
        raise ValueError(f"incomplete repeated arm {arm}")
    return sum(first[key] == second[key] for key in first) / len(first)


def configure_base_runner() -> None:
    r1.TASK_ID = TASK_ID
    r1.PRE_REG_SHA256 = PREREG_SHA256
    r1.SOURCE_HASHES = SOURCE_HASHES
    r1.BLOCKS = BLOCKS
    r1.__file__ = __file__


def finalize_q8(outdir: pathlib.Path, started_utc: str, started_mono: float) -> dict[str, Any]:
    raw = outdir / "raw"
    old = read_json(raw / "actual_scores.json")
    rows = r1.read_jsonl(raw / "samples.jsonl")
    metrics = {
        "explicit_cache_controls_verified": old["explicit_cache_controls_verified"],
        "valid_abba_blocks": old["valid_abba_blocks"],
        "recorded_requests": old["recorded_requests"],
        "successful_response_rate": old["successful_response_rate"],
        "f16_repeat_parity_rate": repeat_parity(rows, "f16"),
        "q8_repeat_parity_rate": repeat_parity(rows, "q4"),
        "q8_f16_extracted_answer_parity_rate": old["extracted_answer_parity_rate"],
        "f16_accuracy": old["f16_accuracy"],
        "q8_accuracy": old["q4_accuracy"],
        "q8_accuracy_regression": old["q4_accuracy_regression"],
        "f16_median_tps": old["f16_median_tps"],
        "q8_median_tps": old["q4_median_tps"],
        "q8_vs_f16_throughput_ratio": old["q4_vs_f16_throughput_ratio"],
        "f16_vram_mib": old["f16_vram_mib"],
        "q8_vram_mib": old["q4_vram_mib"],
        "vram_saving_mib": old["vram_saving_mib"],
        "service_and_embedding_restored": old["service_and_embedding_restored"],
    }
    r1.write_json(raw / "actual_scores.json", metrics)
    definitions = {
        "treatment_identity": ("explicit_cache_controls_verified", "eq", True),
        "balanced_crossover": ("valid_abba_blocks", "eq", 4),
        "request_integrity": ("successful_response_rate", "eq", 1.0),
        "f16_repeatability": ("f16_repeat_parity_rate", "eq", 1.0),
        "q8_repeatability": ("q8_repeat_parity_rate", "eq", 1.0),
        "semantic_parity": ("q8_f16_extracted_answer_parity_rate", "ge", 0.95),
        "quality_non_regression": ("q8_accuracy_regression", "le", 0.03125),
        "physical_memory_saving": ("vram_saving_mib", "ge", 500),
        "throughput_non_regression": ("q8_vs_f16_throughput_ratio", "ge", 0.95),
        "service_recovery": ("service_and_embedding_restored", "eq", True),
    }
    gates: dict[str, Any] = {}
    for gate_id, (metric, operator, threshold) in definitions.items():
        actual = metrics[metric]
        passed = actual == threshold if operator == "eq" else actual >= threshold if operator == "ge" else actual <= threshold
        gates[gate_id] = {"metric": metric, "operator": operator,
                          "threshold": threshold, "actual": actual, "pass": passed}

    source_receipt_path = ROOT / "runs/research/BACKLOG-QWEN38-KV-PRECISION-01/raw/receipt.json"
    source_receipt = read_json(source_receipt_path)
    r1.write_json(raw / "source_execution_receipt.json", {
        "source_task_id": "BACKLOG-QWEN38-KV-PRECISION-01",
        "receipt_sha256": sha256_file(source_receipt_path),
        "receipt_fingerprint": source_receipt["receipt_fingerprint"],
        "source_q4_parity": 0.75,
    })
    r1.write_json(raw / "hardware_metrics.json", {key: metrics[key] for key in (
        "f16_median_tps", "q8_median_tps", "q8_vs_f16_throughput_ratio",
        "f16_vram_mib", "q8_vram_mib", "vram_saving_mib")})
    r1.write_json(raw / "paired_baseline.json", {
        "baseline": "f16", "treatment": "q8_0", "cross_precision_pairs": 64,
        "f16_repeat_pairs": 32, "q8_repeat_pairs": 32,
    })
    r1.write_json(raw / "treatment_controls.json", {
        "order": ["f16", "q8_0", "q8_0", "f16"],
        "server_common": r1.SERVER_COMMON, "cache_types": ["f16", "q8_0"],
        "warmups_per_block": 4,
    })
    r1.write_json(raw / "independent_evaluation.json", {
        "executor_rescore": metrics, "source_q4_parity": 0.75,
        "independent_review_pending": True,
    })
    evidence = {
        "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
        "binary_identity": "raw/binary_identity.json", "block_logs": "raw/logs",
        "dataset_hashes": "raw/dataset_hashes.json", "effective_route": "raw/effective_route.json",
        "environment": "raw/environment.json", "hardware_metrics": "raw/hardware_metrics.json",
        "independent_evaluation": "raw/independent_evaluation.json", "paired_baseline": "raw/paired_baseline.json",
        "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl",
        "receipt_fingerprint": "raw/receipt.json", "recovery_state": "raw/recovery_state.json",
        "service_identity": "raw/service_identity.json", "service_maintenance": "raw/service_maintenance.json",
        "source_execution_receipt": "raw/source_execution_receipt.json",
        "treatment_controls": "raw/treatment_controls.json",
    }
    frozen_paths = [ROOT / relative for relative in SOURCE_HASHES]
    evidence_files = sorted(path for path in raw.rglob("*")
                            if path.is_file() and path.name != "receipt.json")
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(), started_at_utc=started_utc,
        started_monotonic=started_mono, input_paths=[*frozen_paths, *evidence_files],
        packages=[], runtime={"execution_mode": "physical_q8_kv_cache_abba",
        "host_pid": os.getpid(), "blocks": 4, "requests": len(rows),
        "source_q4_parity": 0.75},
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(f"incomplete provenance: {errors}")
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID,
               "provenance": provenance, "provenance_complete": True,
               "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    r1.write_json(raw / "receipt.json", receipt)
    failed = [gate_id for gate_id, gate in gates.items() if not gate["pass"]]
    claim = ("QWEN38_Q8_KV_PHYSICALLY_QUALIFIED_R2" if not failed
             else "QWEN38_Q8_KV_PHYSICALLY_REJECTED_R2")
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Q8_0 saved `{metrics['vram_saving_mib']:.1f}` MiB, reached "
        f"`{metrics['q8_vs_f16_throughput_ratio']:.4f}x` F16 throughput, and had "
        f"cross-precision answer parity `{metrics['q8_f16_extracted_answer_parity_rate']:.4f}`. "
        f"F16/Q8 repeat parity was `{metrics['f16_repeat_parity_rate']:.4f}`/"
        f"`{metrics['q8_repeat_parity_rate']:.4f}`. Failed gates: "
        f"`{', '.join(failed) if failed else 'none'}`. The KV precision family stops here.\n",
        encoding="utf-8", newline="\n")
    state = read_json(raw / "runner_state.json")
    state.update({"status": "completed", "claim": claim, "failed_gates": failed})
    r1.write_json(raw / "runner_state.json", state)
    return receipt


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    configure_base_runner()
    r1.execute(outdir)
    return finalize_q8(outdir, started_utc, started_mono)


def selfcheck() -> None:
    assert [row["cache"] for row in BLOCKS] == ["f16", "q8_0", "q8_0", "f16"]
    assert [row["pair"] for row in BLOCKS] == [0, 0, 1, 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
