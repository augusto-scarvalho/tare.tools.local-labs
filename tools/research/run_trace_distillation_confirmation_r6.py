#!/usr/bin/env python3
"""Continue R5 with nine frozen workers and the actual 48-task QA panel."""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file
from tools.research import run_trace_distillation_confirmation_r5 as r5
from tools.research import run_trace_distillation_training_r2 as r2

TASK_ID = "BACKLOG-ADAPT-TRACE-DISTILL-06"
SOURCE = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-05"
LEDGER_PATH = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-06/CONTINUATION_SOURCES.json"
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-ADAPT-TRACE-DISTILL-06.json"
EXPECTED_STATIC = {
    ADMISSION: "31d8cb59f17bc01a1eafcd7550372cc19f3c7faa47ca3b3f106149d69c950fa3",
    LEDGER_PATH: "817d595739eff09e3b7d2a78f82b331f8a411d0874b54abe8d43055a2d3066fc",
    SOURCE / "PRE_REGISTRATION.md": "30b87154ef906703bc04f35eea67ba110b9968ba6e2c0b2bb2264526bfdd86e1",
    SOURCE / "raw/training_pairs.json": "5c3f0d5fd80d97351839bca1e38685e5e21b3357dfa56077f44f02b857bfe4cc",
    SOURCE / "raw/service_maintenance.json": "0831e29cf2e138eb90ed663c07ab1252e0a82ae4a024c5db6df4649f0df49825",
    r2.DEFAULT_MATH_PATH: "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    r2.TEACHER_PATH: "dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e",
    r2.DEFAULT_QA_PATH: "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
}
QA_ID_HASH = "5377ee57e27a3480fdad26c05cc7cc13b7e177c69abdda77795f898d43df45f3"
IMPORTED_LABELS = [
    "seed_20260830_answer_only", "seed_20260830_full_trace",
    "seed_20260831_answer_only", "seed_20260831_full_trace",
    "seed_20260832_answer_only", "seed_20260832_full_trace",
    "seed_20260833_answer_only", "seed_20260833_full_trace",
    "seed_20260834_answer_only",
]
FRESH_LABELS = [
    "seed_20260834_full_trace",
    "seed_20260835_answer_only", "seed_20260835_full_trace",
    "seed_20260836_answer_only", "seed_20260836_full_trace",
]


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def actual_qa_ids() -> list[str]:
    ids = [
        json.loads(line)["id"]
        for line in r2.DEFAULT_QA_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(ids) != 48 or len(set(ids)) != 48 or canonical_json_sha256(ids) != QA_ID_HASH:
        raise ValueError("actual QA panel differs from preregistration")
    return ids


def verify_sources() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    static: dict[str, Any] = {}
    for path, expected in EXPECTED_STATIC.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"static continuation source mismatch: {path}: {actual} != {expected}")
        static[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": actual}
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    worker_ledger: dict[str, dict[str, Any]] = {}
    expected_existing_qa = set(actual_qa_ids()[:10])
    for label, expected in ledger["completed_workers"].items():
        worker_path = SOURCE / "raw/workers" / f"{label}.json"
        checkpoint = SOURCE / "raw/checkpoints" / label
        observed = {
            "worker_sha256": sha256_file(worker_path),
            "config_sha256": sha256_file(checkpoint / "adapter_config.json"),
            "weights_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
        }
        if observed != expected:
            raise ValueError(f"partial source mismatch for {label}: {observed} != {expected}")
        payload = json.loads(worker_path.read_text(encoding="utf-8"))
        qa_ids = {sample["task_id"] for sample in payload["qa_samples"]}
        if payload["training_step_count"] != 504 or payload["math_total"] != 256 or payload["qa_total"] != 10:
            raise ValueError(f"invalid imported dimensions for {label}")
        if qa_ids != expected_existing_qa:
            raise ValueError(f"unexpected imported QA IDs for {label}: {sorted(qa_ids)}")
        worker_ledger[label] = {**observed, "seed": payload["seed"], "arm": payload["arm"]}
    if sorted(worker_ledger) != sorted(IMPORTED_LABELS):
        raise ValueError("continuation ledger does not bind exactly the nine imported workers")
    return static, worker_ledger


def augmentation_worker(output: str, checkpoint: str, qa_ids_json: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    qa_ids = json.loads(pathlib.Path(qa_ids_json).read_text(encoding="utf-8"))["qa_ids"]
    tokenizer = AutoTokenizer.from_pretrained(r2.BASE_MODEL_WSL, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        r2.BASE_MODEL_WSL, local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, checkpoint).eval()
    tasks = r2.load_qa_panel(r2.DEFAULT_QA_PATH, qa_ids)
    samples: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, 1):
        tokens = tokenizer(task["prompt"], return_tensors="pt").to("cuda")
        prompt_n = tokens["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **tokens, max_new_tokens=128, do_sample=False, temperature=None, top_p=None,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )
        output_ids = generated[0][prompt_n:]
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        correct, detail = r2.grade_qa(task, text)
        samples.append({
            "panel": "qa", "task_id": task["id"], "prompt": task["prompt"],
            "output_text": text, "correct": correct, "grade_detail": detail,
            "new_tokens": len(output_ids),
            "natural_eos": bool(tokenizer.eos_token_id is not None and generated[0][-1].item() == tokenizer.eos_token_id),
        })
        if index % 10 == 0:
            print(f"[AUGMENT] QA {index}/{len(tasks)}", flush=True)
    if len(samples) != len(qa_ids):
        raise ValueError(f"augmentation produced {len(samples)} of {len(qa_ids)} QA samples")
    write_json(pathlib.Path(output), {"qa_ids": qa_ids, "qa_samples": samples})


def fresh_worker(output: str, checkpoint: str, manifest: str, arm: str, seed: int) -> None:
    r5.QA_IDS = actual_qa_ids()
    r5.worker(output, checkpoint, manifest, arm, seed)


def wsl_command(*arguments: str) -> list[str]:
    return ["wsl", "-d", "Ubuntu-24.04", "--", r2.WSL_PYTHON, r2.windows_path_to_wsl(pathlib.Path(__file__).resolve()), *arguments]


def run_checked(command: list[str], stdout_path: pathlib.Path, stderr_path: pathlib.Path, timeout: int) -> None:
    completed = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"worker failed ({completed.returncode}): {completed.stderr[-4000:]}")


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    workers = raw / "workers"
    checkpoints = raw / "checkpoints"
    augmentations = raw / "augmentations"
    finalized = raw / "finalized"
    for directory in (workers, checkpoints, augmentations, finalized):
        directory.mkdir(parents=True)

    static_ledger, imported_ledger = verify_sources()
    qa_ids = actual_qa_ids()
    missing_qa_ids = qa_ids[10:]
    qa_panel = raw / "missing_qa_panel.json"
    write_json(qa_panel, {"qa_ids": missing_qa_ids, "full_qa_ids": qa_ids, "full_sha256": canonical_json_sha256(qa_ids)})
    manifest = raw / "training_pairs.json"
    shutil.copy2(SOURCE / "raw/training_pairs.json", manifest)
    write_json(raw / "dataset_hashes.json", {"static": static_ledger, "qa_ids": qa_ids, "qa_id_sha256": canonical_json_sha256(qa_ids)})
    write_json(raw / "model_hash.json", r2.verify_base_model())

    initial_service = r2.query_service()
    initial_gpu = r2.query_gpu()
    initial_embedding = r2.http_get_json("http://127.0.0.1:8081/health")
    maintenance: dict[str, Any] = {
        "initial_service": initial_service, "initial_gpu": initial_gpu,
        "initial_embedding": initial_embedding, "service_stopped_for_vram": False,
    }
    service_stopped = False
    payloads: list[dict[str, Any]] = []
    continuation_rows: list[dict[str, Any]] = []
    try:
        if initial_gpu["memory_free_mib"] < 6000 and initial_service["active_state"] == "active":
            r2.systemctl("stop")
            service_stopped = True
            maintenance["service_stopped_for_vram"] = True
            maintenance["service_after_stop"] = r2.query_service()
            maintenance["embedding_after_stop"] = r2.http_get_json("http://127.0.0.1:8081/health")
            if maintenance["embedding_after_stop"].get("status") != "ok":
                raise RuntimeError("embedding service became unhealthy")

        for label in IMPORTED_LABELS:
            source_worker = SOURCE / "raw/workers" / f"{label}.json"
            target_worker = workers / f"{label}.json"
            source_checkpoint = SOURCE / "raw/checkpoints" / label
            target_checkpoint = checkpoints / label
            shutil.copy2(source_worker, target_worker)
            shutil.copytree(source_checkpoint, target_checkpoint)
            augmentation = augmentations / f"{label}.json"
            command = wsl_command(
                "--augment-mode", "--worker-out", r2.windows_path_to_wsl(augmentation),
                "--checkpoint-out", r2.windows_path_to_wsl(target_checkpoint),
                "--qa-panel", r2.windows_path_to_wsl(qa_panel),
            )
            run_checked(command, augmentations / f"{label}.stdout.log", augmentations / f"{label}.stderr.log", 3600)
            payload = json.loads(target_worker.read_text(encoding="utf-8"))
            added = json.loads(augmentation.read_text(encoding="utf-8"))["qa_samples"]
            existing_ids = {sample["task_id"] for sample in payload["qa_samples"]}
            if existing_ids.intersection(sample["task_id"] for sample in added):
                raise ValueError(f"augmentation overlaps existing QA for {label}")
            payload["qa_samples"].extend(added)
            payload["qa_total"] = len(payload["qa_samples"])
            payload["qa_correct"] = sum(sample["correct"] for sample in payload["qa_samples"])
            payload["continuation_source_sha256"] = imported_ledger[label]["worker_sha256"]
            write_json(target_worker, payload)
            if payload["math_total"] != 256 or payload["qa_total"] != 48:
                raise ValueError(f"augmented dimensions invalid for {label}")
            payloads.append(payload)
            marker = {"label": label, "mode": "imported_plus_38_qa", "worker_sha256": sha256_file(target_worker)}
            write_json(finalized / f"{label}.json", marker)
            continuation_rows.append(marker)
            print(f"[HOST] finalized imported {label}: math={payload['math_correct']}/256 qa={payload['qa_correct']}/48", flush=True)

        for label in FRESH_LABELS:
            _, seed_text, arm_suffix = label.split("_", 2)
            seed = int(seed_text)
            arm = arm_suffix
            output = workers / f"{label}.json"
            checkpoint = checkpoints / label
            command = wsl_command(
                "--fresh-worker-mode", "--worker-out", r2.windows_path_to_wsl(output),
                "--checkpoint-out", r2.windows_path_to_wsl(checkpoint),
                "--manifest", r2.windows_path_to_wsl(manifest), "--arm", arm, "--seed", str(seed),
            )
            run_checked(command, workers / f"{label}.stdout.log", workers / f"{label}.stderr.log", 14400)
            payload = json.loads(output.read_text(encoding="utf-8"))
            if payload["training_step_count"] != 504 or payload["math_total"] != 256 or payload["qa_total"] != 48:
                raise ValueError(f"fresh worker dimensions invalid for {label}")
            payloads.append(payload)
            marker = {"label": label, "mode": "fresh", "worker_sha256": sha256_file(output)}
            write_json(finalized / f"{label}.json", marker)
            print(f"[HOST] finalized fresh {label}: math={payload['math_correct']}/256 qa={payload['qa_correct']}/48", flush=True)
    finally:
        if service_stopped:
            r2.systemctl("start")
            maintenance["inference_health_final"] = r2.wait_for_health("http://127.0.0.1:8080/health", timeout_seconds=180)
        maintenance["final_service"] = r2.query_service()
        maintenance["final_embedding"] = r2.wait_for_health("http://127.0.0.1:8081/health", timeout_seconds=30)
        maintenance["final_gpu"] = r2.query_gpu()
        maintenance["service_and_embedding_restored"] = (
            maintenance["final_service"]["active_state"] == initial_service["active_state"]
            and r2.normalize_exec_start(maintenance["final_service"]["exec_start"])
            == r2.normalize_exec_start(initial_service["exec_start"])
            and maintenance["final_service"]["n_restarts"] == initial_service["n_restarts"]
            and maintenance["final_embedding"].get("status") == "ok"
            and (not service_stopped or maintenance.get("inference_health_final", {}).get("status") == "ok")
        )
        write_json(raw / "service_maintenance.json", maintenance)

    if len(payloads) != 14:
        raise ValueError(f"expected 14 final workers, got {len(payloads)}")
    payloads.sort(key=lambda item: (item["seed"], item["arm"]))
    write_json(raw / "continuation_ledger.json", {
        "source_ledger_sha256": sha256_file(LEDGER_PATH),
        "hash_verified_imported_workers": len(imported_ledger),
        "sources": imported_ledger,
        "finalized": continuation_rows,
    })

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    treatment_rows = []
    for seed in r5.SEEDS:
        arms = {payload["arm"]: payload for payload in payloads if payload["seed"] == seed}
        treatment_rows.append({
            "seed": seed,
            "same_task_order": arms["answer_only"]["training_task_ids"] == arms["full_trace"]["training_task_ids"],
            "target_texts_distinct": sum(row["answer_only"] != row["full_trace"] for row in manifest_payload["pool"]),
            "answer_target_sha256": arms["answer_only"]["training_target_sha256"],
            "trace_target_sha256": arms["full_trace"]["training_target_sha256"],
        })
    treatment_verified = all(
        row["same_task_order"] and row["target_texts_distinct"] == 168
        and row["answer_target_sha256"] != row["trace_target_sha256"]
        for row in treatment_rows
    )
    write_json(raw / "treatment_materiality.json", {"verified": treatment_verified, "seeds": treatment_rows})

    checkpoint_ledger = {}
    for payload in payloads:
        label = f"seed_{payload['seed']}_{payload['arm']}"
        checkpoint = checkpoints / label
        checkpoint_ledger[label] = {
            "config_sha256": sha256_file(checkpoint / "adapter_config.json"),
            "weights_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
            "imported": label in IMPORTED_LABELS,
        }
    write_json(raw / "checkpoint_hashes.json", checkpoint_ledger)
    write_json(raw / "training_trace.json", [{"arm": p["arm"], "seed": p["seed"], "trace": p["training_trace"]} for p in payloads])
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for payload in payloads:
            for sample in [*payload["math_samples"], *payload["qa_samples"]]:
                stream.write(json.dumps({"arm": payload["arm"], "seed": payload["seed"], **sample}, ensure_ascii=False) + "\n")
    write_json(raw / "student_samples.json", [{
        "arm": p["arm"], "seed": p["seed"], "math_samples": p["math_samples"], "qa_samples": p["qa_samples"]
    } for p in payloads])
    write_json(raw / "teacher_samples.json", {
        "source": r2.TEACHER_PATH.relative_to(ROOT).as_posix(),
        "training_pool": [{"task_id": row["task_id"], "full_trace": row["full_trace"]} for row in manifest_payload["pool"]],
    })

    r5.QA_IDS = qa_ids
    scores, independent_match = r5.score_outputs(payloads)
    write_json(raw / "actual_scores.json", scores)
    write_json(raw / "independent_evaluation.json", {"independent_scorer_match": independent_match, "scores": scores})
    bootstrap_lower = scores["hierarchical_bootstrap"]["lower_95"]
    gates = {
        "continuation_integrity": {"metric": "hash_verified_imported_workers", "operator": "eq", "threshold": 9, "actual": len(imported_ledger), "pass": len(imported_ledger) == 9},
        "treatment_materiality": {"metric": "matched_distinct_training_targets_verified", "operator": "eq", "threshold": True, "actual": treatment_verified, "pass": treatment_verified is True},
        "training_budget": {"metric": "training_steps_per_arm_per_seed", "operator": "eq", "threshold": 504, "actual": min(p["training_step_count"] for p in payloads), "pass": all(p["training_step_count"] == 504 for p in payloads)},
        "seed_coverage": {"metric": "completed_paired_seeds", "operator": "eq", "threshold": 7, "actual": len(scores["seeds"]), "pass": len(scores["seeds"]) == 7},
        "math_coverage": {"metric": "heldout_math_samples_per_arm_per_seed", "operator": "eq", "threshold": 256, "actual": min(p["math_total"] for p in payloads), "pass": all(p["math_total"] == 256 for p in payloads)},
        "qa_coverage": {"metric": "protected_qa_samples_per_arm_per_seed", "operator": "eq", "threshold": 48, "actual": min(p["qa_total"] for p in payloads), "pass": all(p["qa_total"] == 48 for p in payloads)},
        "confirmed_gain": {"metric": "hierarchical_bootstrap_95ci_lower_trace_math_gain", "operator": "gt", "threshold": 0.0, "actual": bootstrap_lower, "pass": bootstrap_lower > 0.0},
        "directional_repeatability": {"metric": "seeds_with_positive_trace_math_gain", "operator": "ge", "threshold": 5, "actual": scores["seeds_with_positive_trace_math_gain"], "pass": scores["seeds_with_positive_trace_math_gain"] >= 5},
        "protected_retention": {"metric": "mean_protected_qa_regression_vs_answer_only", "operator": "le", "threshold": 0.05, "actual": scores["mean_protected_qa_regression_vs_answer_only"], "pass": scores["mean_protected_qa_regression_vs_answer_only"] <= 0.05},
        "service_recovery": {"metric": "service_and_embedding_restored", "operator": "eq", "threshold": True, "actual": maintenance["service_and_embedding_restored"], "pass": maintenance["service_and_embedding_restored"] is True},
    }
    evidence = {
        "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
        "checkpoint_hashes": "raw/checkpoint_hashes.json", "continuation_ledger": "raw/continuation_ledger.json",
        "dataset_hashes": "raw/dataset_hashes.json", "independent_evaluation": "raw/independent_evaluation.json",
        "model_hash": "raw/model_hash.json", "provenance": "raw/receipt.json",
        "raw_samples": "raw/samples.jsonl", "receipt_fingerprint": "raw/receipt.json",
        "service_maintenance": "raw/service_maintenance.json", "student_samples": "raw/student_samples.json",
        "teacher_samples": "raw/teacher_samples.json", "training_pairs": "raw/training_pairs.json",
        "training_trace": "raw/training_trace.json", "treatment_materiality": "raw/treatment_materiality.json",
    }
    evidence_files = sorted({raw / value.removeprefix("raw/") for value in evidence.values() if value != "raw/receipt.json"})
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(), started_at_utc=started, started_monotonic=mono,
        input_paths=[*EXPECTED_STATIC.keys(), *evidence_files], packages=["pytest"],
        runtime={"execution_mode": "r5_hash_bound_continuation_with_qa_augmentation", "imported_workers": 9, "fresh_workers": 5, "qa_tasks": 48},
    )
    complete, errors = provenance_complete(provenance)
    if not complete or not independent_match:
        raise ValueError(f"evidence validation failed: provenance={errors}, scorer_match={independent_match}")
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = "TRACE_DISTILLATION_CONFIRMED_R6" if not failed else "TRACE_DISTILLATION_NOT_CONFIRMED_R6"
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Seven paired seeds completed with 504 steps, 256 math and 48 protected-QA tasks per arm. "
        f"Mean trace math gain `{scores['mean_trace_math_gain_over_answer_only']:.6f}`; bootstrap 95% interval "
        f"`[{scores['hierarchical_bootstrap']['lower_95']:.6f}, {scores['hierarchical_bootstrap']['upper_95']:.6f}]`; "
        f"positive seeds `{scores['seeds_with_positive_trace_math_gain']}/7`; "
        f"mean QA regression `{scores['mean_protected_qa_regression_vs_answer_only']:.6f}`. "
        f"Failed gates: `{', '.join(failed) if failed else 'none'}`.\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--augment-mode", action="store_true")
    parser.add_argument("--fresh-worker-mode", action="store_true")
    parser.add_argument("--worker-out")
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--qa-panel")
    parser.add_argument("--manifest")
    parser.add_argument("--arm", choices=["answer_only", "full_trace"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.augment_mode:
        if not all((args.worker_out, args.checkpoint_out, args.qa_panel)):
            parser.error("augment mode requires worker-out, checkpoint-out and qa-panel")
        augmentation_worker(args.worker_out, args.checkpoint_out, args.qa_panel)
        return 0
    if args.fresh_worker_mode:
        if not all((args.worker_out, args.checkpoint_out, args.manifest, args.arm, args.seed)):
            parser.error("fresh worker mode requires output, checkpoint, manifest, arm and seed")
        fresh_worker(args.worker_out, args.checkpoint_out, args.manifest, args.arm, args.seed)
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
