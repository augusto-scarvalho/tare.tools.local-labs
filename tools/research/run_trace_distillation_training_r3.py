#!/usr/bin/env python3
"""Continuation of the frozen trace-distillation R2 design after host decoder abort."""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file
from tools.research import run_trace_distillation_training_r2 as r2

TASK_ID = "BACKLOG-ADAPT-TRACE-DISTILL-03"
SOURCE = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-02"
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-ADAPT-TRACE-DISTILL-03.json"
EXPECTED_SOURCE_HASHES = {
    SOURCE / "ABORTED.md": "648970adb100d68f8a63523b5ef5f7a0eb9bf30228e3161f7bb0b7e4aeb770a9",
    SOURCE / "raw/training_pairs.json": "e6dd3bb9d86b0c8d34f89f68d07768e6d0f53451295e70a02fafc9d6a0748966",
    SOURCE / "raw/service_maintenance.json": "029892b166778978911c22f6847ade1f155019f632b11b3d5b0ae9a3608bbbbc",
    SOURCE / "raw/workers/seed_20260824_answer_only.json": "be30915b4b5c8b402a98953ecd0aba829afbe9fa58be5116616546614ccde79c",
    SOURCE / "raw/workers/seed_20260824_full_trace.json": "c389a8290effa80b45685516375499a0f0c41a79348b3490e489608d27eaa7db",
    SOURCE / "raw/workers/seed_20260825_answer_only.json": "a8fcd0a8bb782176840882513e60d9670550e13c1493f994b09f409c55a2ef36",
    SOURCE / "raw/workers/seed_20260825_full_trace.json": "0220a38e5ae1c74695ac91b25807c85a907fa8d0c949306c59e6be6e35f872f0",
    SOURCE / "raw/checkpoints/seed_20260824_answer_only/adapter_config.json": "5acceba987552a5aa7f128d3840b9c465225345f60eb585ab0e9d7b7742e5e14",
    SOURCE / "raw/checkpoints/seed_20260824_answer_only/adapter_model.safetensors": "ef5bec8822e856883eaec930d2b851892bb6b681bde1fda5f76005667adbf1a2",
    SOURCE / "raw/checkpoints/seed_20260824_full_trace/adapter_config.json": "4872214e344ce266b0b10a1f70db33775224441ff1d78dbe0c7fb4dd70aeac84",
    SOURCE / "raw/checkpoints/seed_20260824_full_trace/adapter_model.safetensors": "174832aa1bd25cbc5ed7f0ff717ad253ec94e2c23edc82e6f828ceadeed566b7",
    SOURCE / "raw/checkpoints/seed_20260825_answer_only/adapter_config.json": "091ddc225ff85380e6815963fad17391f4d8e89fb38523ddc2b798086c82ecb1",
    SOURCE / "raw/checkpoints/seed_20260825_answer_only/adapter_model.safetensors": "56ff9be8c5ac0876389cf12fe23a2ac301eac7c99cef977fa455b76f5817a2e6",
    SOURCE / "raw/checkpoints/seed_20260825_full_trace/adapter_config.json": "091ddc225ff85380e6815963fad17391f4d8e89fb38523ddc2b798086c82ecb1",
    SOURCE / "raw/checkpoints/seed_20260825_full_trace/adapter_model.safetensors": "dc696b7553cf8e4d920f8554ec4e3dee484a04da374ef0d54bcb48160044050a",
    ADMISSION: "17915c826fe75ba2ede1a1be151824f9ae8c276f6e6047a0d9f3713cdf94dbed",
}
IMPORTED_LABELS = [
    "seed_20260824_answer_only", "seed_20260824_full_trace",
    "seed_20260825_answer_only", "seed_20260825_full_trace",
]


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_sources() -> dict:
    ledger = {}
    for path, expected in EXPECTED_SOURCE_HASHES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"continuation source mismatch: {path}: {actual} != {expected}")
        ledger[str(path.relative_to(ROOT).as_posix())] = {"bytes": path.stat().st_size, "sha256": actual}
    return ledger


def materiality(payloads: list[dict], manifest: dict) -> tuple[bool, list[dict]]:
    rows = []
    for seed in r2.SEEDS:
        arms = {payload["arm"]: payload for payload in payloads if payload["seed"] == seed}
        selected = manifest["seeds"][str(seed)]
        row = {
            "seed": seed,
            "same_task_order": arms["answer_only"]["training_task_ids"] == arms["full_trace"]["training_task_ids"],
            "target_texts_distinct": sum(item["answer_only"] != item["full_trace"] for item in selected),
            "answer_target_sha256": arms["answer_only"]["training_target_sha256"],
            "trace_target_sha256": arms["full_trace"]["training_target_sha256"],
        }
        rows.append(row)
    verified = all(
        row["same_task_order"] and row["target_texts_distinct"] == 128
        and row["answer_target_sha256"] != row["trace_target_sha256"]
        for row in rows
    )
    return verified, rows


def run_experiment(outdir: pathlib.Path) -> dict:
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError(f"raw output directory is not empty: {raw}")
    workers = raw / "workers"
    checkpoints = raw / "checkpoints"
    workers.mkdir(parents=True)
    checkpoints.mkdir(parents=True)

    continuation = verify_sources()
    original_inputs = r2.verify_host_inputs()
    base_ledger = r2.verify_base_model()
    manifest_source = SOURCE / "raw/training_pairs.json"
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    rebuilt = r2.build_training_manifest()
    if canonical_json_sha256(manifest) != canonical_json_sha256(rebuilt):
        raise ValueError("frozen training manifest differs from deterministic rebuild")
    manifest_path = raw / "training_pairs.json"
    shutil.copy2(manifest_source, manifest_path)
    write_json(raw / "continuation_ledger.json", continuation)
    write_json(raw / "dataset_hashes.json", {"original_inputs": original_inputs, "continuation": continuation})
    write_json(raw / "model_hash.json", base_ledger)

    payloads = []
    for label in IMPORTED_LABELS:
        source_json = SOURCE / "raw/workers" / f"{label}.json"
        target_json = workers / f"{label}.json"
        shutil.copy2(source_json, target_json)
        shutil.copytree(SOURCE / "raw/checkpoints" / label, checkpoints / label)
        payload = json.loads(target_json.read_text(encoding="utf-8"))
        payload["continuation_source"] = str(source_json.relative_to(ROOT).as_posix())
        payloads.append(payload)

    initial_service = r2.query_service()
    initial_gpu = r2.query_gpu()
    initial_embedding = r2.http_get_json("http://127.0.0.1:8081/health")
    maintenance = {
        "initial_service": initial_service, "initial_gpu": initial_gpu,
        "initial_embedding": initial_embedding, "service_stopped_for_vram": False,
    }
    service_stopped = False
    script_r2 = ROOT / "tools/research/run_trace_distillation_training_r2.py"
    try:
        if initial_gpu["memory_free_mib"] < 6000 and initial_service["active_state"] == "active":
            r2.systemctl("stop")
            service_stopped = True
            maintenance["service_stopped_for_vram"] = True
            maintenance["service_after_stop"] = r2.query_service()
            maintenance["embedding_after_stop"] = r2.http_get_json("http://127.0.0.1:8081/health")
            if maintenance["embedding_after_stop"].get("status") != "ok":
                raise RuntimeError("embedding service became unhealthy")
        for arm in r2.ARM_ORDERS[20260826]:
            label = f"seed_20260826_{arm}"
            output = workers / f"{label}.json"
            checkpoint = checkpoints / label
            command = r2.worker_command(script_r2, manifest_path, output, checkpoint, arm, 20260826)
            completed = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=1800, check=False,
            )
            (workers / f"{label}.stdout.log").write_text(completed.stdout, encoding="utf-8")
            (workers / f"{label}.stderr.log").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"worker {label} failed: {completed.stderr[-3000:]}")
            payload = json.loads(output.read_text(encoding="utf-8"))
            payload["host_command"] = command
            payloads.append(payload)
            print(f"[HOST] {label} complete: math={payload['math_correct']}/32 qa={payload['qa_correct']}/16", flush=True)
    finally:
        if service_stopped:
            r2.systemctl("start")
            maintenance["inference_health_final"] = r2.wait_for_health("http://127.0.0.1:8080/health", timeout_seconds=120)
        maintenance["final_service"] = r2.query_service()
        maintenance["final_embedding"] = r2.wait_for_health("http://127.0.0.1:8081/health", timeout_seconds=30)
        maintenance["final_gpu"] = r2.query_gpu()
        maintenance["service_and_embedding_restored"] = (
            maintenance["final_service"]["active_state"] == initial_service["active_state"]
            and r2.normalize_exec_start(maintenance["final_service"]["exec_start"]) == r2.normalize_exec_start(initial_service["exec_start"])
            and maintenance["final_service"]["n_restarts"] == initial_service["n_restarts"]
            and maintenance["final_embedding"].get("status") == "ok"
            and (not service_stopped or maintenance.get("inference_health_final", {}).get("status") == "ok")
        )
        write_json(raw / "service_maintenance.json", maintenance)

    if len(payloads) != 6:
        raise ValueError(f"expected six combined workers, got {len(payloads)}")
    clean = [{
        "arm": payload["arm"], "seed": payload["seed"], "pid": payload["pid"],
        "base_preexisting_peft_module_count": payload["base_preexisting_peft_module_count"],
        "post_injection_peft_module_count": payload["post_injection_peft_module_count"],
        "training_pair_count": payload["training_pair_count"],
        "training_target_sha256": payload["training_target_sha256"],
        "imported": payload["seed"] in {20260824, 20260825},
    } for payload in payloads]
    write_json(raw / "clean_base_receipts.json", clean)
    treatment_verified, treatment_rows = materiality(payloads, manifest)
    write_json(raw / "treatment_materiality.json", {"verified": treatment_verified, "seeds": treatment_rows})

    checkpoint_ledger = {}
    for payload in payloads:
        label = f"seed_{payload['seed']}_{payload['arm']}"
        checkpoint = checkpoints / label
        checkpoint_ledger[label] = {
            "config_sha256": sha256_file(checkpoint / "adapter_config.json"),
            "weights_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
            "imported": payload["seed"] in {20260824, 20260825},
        }
    write_json(raw / "checkpoint_hashes.json", checkpoint_ledger)
    write_json(raw / "training_trace.json", [{
        "arm": payload["arm"], "seed": payload["seed"], "trace": payload["training_trace"]
    } for payload in payloads])
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for payload in payloads:
            for sample in [*payload["math_samples"], *payload["qa_samples"]]:
                stream.write(json.dumps({"arm": payload["arm"], "seed": payload["seed"], **sample}, ensure_ascii=False) + "\n")
    write_json(raw / "student_samples.json", [{
        "arm": payload["arm"], "seed": payload["seed"],
        "math_samples": payload["math_samples"], "qa_samples": payload["qa_samples"],
    } for payload in payloads])
    write_json(raw / "teacher_samples.json", {
        "source": str(r2.TEACHER_PATH.relative_to(ROOT).as_posix()),
        "selected": {seed: [{"task_id": row["task_id"], "full_trace": row["full_trace"]} for row in rows]
                     for seed, rows in manifest["seeds"].items()},
    })
    scores, independent = r2.score_outputs(payloads)
    write_json(raw / "actual_scores.json", scores)
    write_json(raw / "independent_evaluation.json", {"independent_scorer_match": independent, "scores": scores})

    receipt_inputs = [
        raw / "actual_scores.json", raw / "checkpoint_hashes.json", raw / "clean_base_receipts.json",
        raw / "continuation_ledger.json", raw / "dataset_hashes.json", raw / "independent_evaluation.json",
        raw / "model_hash.json", raw / "samples.jsonl", raw / "service_maintenance.json",
        raw / "student_samples.json", raw / "teacher_samples.json", raw / "training_pairs.json",
        raw / "training_trace.json", raw / "treatment_materiality.json", *EXPECTED_SOURCE_HASHES.keys(),
        *r2.EXPECTED_HOST_HASHES.keys(),
    ]
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(), started_at_utc=started_utc,
        started_monotonic=started_mono, input_paths=receipt_inputs, packages=["pytest"],
        runtime={"execution_mode": "frozen_trace_distillation_continuation", "imported_workers": 4, "new_workers": 2},
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise ValueError(f"incomplete provenance: {errors}")
    clean_count = sum(row["base_preexisting_peft_module_count"] == 0 for row in clean)
    gates = {
        "continuation_integrity": {"metric": "frozen_partial_workers_verified", "operator": "eq", "threshold": 4, "actual": len(IMPORTED_LABELS), "pass": len(IMPORTED_LABELS) == 4},
        "treatment_materiality": {"metric": "matched_distinct_training_targets_verified", "operator": "eq", "threshold": True, "actual": treatment_verified, "pass": treatment_verified is True},
        "clean_base": {"metric": "fresh_base_workers", "operator": "eq", "threshold": 6, "actual": clean_count, "pass": clean_count == 6},
        "paired_training": {"metric": "matched_pairs_per_arm_per_seed", "operator": "eq", "threshold": 128, "actual": min(row["training_pair_count"] for row in clean), "pass": all(row["training_pair_count"] == 128 for row in clean)},
        "seed_coverage": {"metric": "completed_paired_seeds", "operator": "eq", "threshold": 3, "actual": len(scores["seeds"]), "pass": len(scores["seeds"]) == 3},
        "heldout_gain": {"metric": "mean_trace_math_gain_over_answer_only", "operator": "gt", "threshold": 0.0, "actual": scores["mean_trace_math_gain_over_answer_only"], "pass": scores["mean_trace_math_gain_over_answer_only"] > 0},
        "directional_repeatability": {"metric": "seeds_with_nonnegative_trace_math_gain", "operator": "ge", "threshold": 2, "actual": scores["seeds_with_nonnegative_trace_math_gain"], "pass": scores["seeds_with_nonnegative_trace_math_gain"] >= 2},
        "protected_regression": {"metric": "mean_protected_qa_regression_vs_answer_only", "operator": "le", "threshold": 0.05, "actual": scores["mean_protected_qa_regression_vs_answer_only"], "pass": scores["mean_protected_qa_regression_vs_answer_only"] <= 0.05},
        "service_recovery": {"metric": "service_and_embedding_restored", "operator": "eq", "threshold": True, "actual": maintenance["service_and_embedding_restored"], "pass": maintenance["service_and_embedding_restored"] is True},
    }
    receipt = {
        "schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID,
        "provenance": provenance, "provenance_complete": complete, "gates": gates,
        "evidence": {
            "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
            "checkpoint_hashes": "raw/checkpoint_hashes.json", "clean_base_receipts": "raw/clean_base_receipts.json",
            "continuation_ledger": "raw/continuation_ledger.json", "dataset_hashes": "raw/dataset_hashes.json",
            "independent_evaluation": "raw/independent_evaluation.json", "model_hash": "raw/model_hash.json",
            "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl",
            "receipt_fingerprint": "raw/receipt.json", "service_maintenance": "raw/service_maintenance.json",
            "student_samples": "raw/student_samples.json", "teacher_samples": "raw/teacher_samples.json",
            "training_pairs": "raw/training_pairs.json", "training_trace": "raw/training_trace.json",
            "treatment_materiality": "raw/treatment_materiality.json",
        },
    }
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt = run_experiment(args.outdir.resolve())
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
