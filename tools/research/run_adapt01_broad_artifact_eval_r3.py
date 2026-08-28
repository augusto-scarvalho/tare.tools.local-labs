#!/usr/bin/env python3
"""Complete R2 with the 38 protected-QA tasks omitted by synthetic IDs."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)
from tools.research import run_adapt01_broad_artifact_eval_r2 as r2
from tools.research import run_trace_distillation_training_r2 as training

TASK_ID = "BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-03"
SOURCE = ROOT / "runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-02"
SOURCE_WORKER = SOURCE / "raw/worker.json"
SOURCE_RECEIPT = SOURCE / "raw/receipt.json"
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-03.json"
PREREGISTRATION = ROOT / "runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-03/PRE_REGISTRATION.md"
QA_ID_HASH = "5377ee57e27a3480fdad26c05cc7cc13b7e177c69abdda77795f898d43df45f3"
EXPECTED_HASHES = {
    ADMISSION: "d1a3ac2ae7ea792845de41b1d8c40e3227fa074b250db6a7f27b0dde894afee8",
    SOURCE_WORKER: "930172bc74c565ab93e9afe36a54c23fb7e84e7dc0b2b66a15b1dba6b1e5a38b",
    SOURCE_RECEIPT: "c64faac1231dcbba3b0c7e2ae9071ce6e503ebf62766e6c53b9b95b416ca4780",
    SOURCE / "PRE_REGISTRATION.md": "477e3ece2ff5c70250c45a9c91160f563c36fa10d5344059268c84b99939b196",
    ROOT / "tools/research/run_adapt01_broad_artifact_eval_r2.py": "7978351859fc4c03a534088339b9e58c91074fdc43ee6bb1c5cef99a87aee022",
    r2.ADAPTER / "adapter_model.safetensors": "7f6d082243f6b406259791dc15a65e4b092b48597fad9b68018d507872ad8fa7",
    r2.ADAPTER / "adapter_config.json": "08cf4d254e2a6c9aba9d34ba6a0c76926b478d7cd0ad771062acefb71a31d934",
    training.DEFAULT_MATH_PATH: "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    training.TEACHER_PATH: "dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e",
    training.DEFAULT_QA_PATH: "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
}


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def actual_qa_ids() -> list[str]:
    ids = [
        json.loads(line)["id"]
        for line in training.DEFAULT_QA_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(ids) != 48 or len(set(ids)) != 48 or canonical_json_sha256(ids) != QA_ID_HASH:
        raise ValueError("actual QA panel differs from preregistration")
    return ids


def verify_sources() -> dict[str, dict[str, Any]]:
    ledger = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"source mismatch: {path}: {actual} != {expected}")
        ledger[path.relative_to(ROOT).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    return ledger


def long_systemctl(action: str) -> None:
    completed = subprocess.run(
        [
            "wsl", "-d", "Ubuntu-24.04", "-u", "root", "--",
            "systemctl", action, "llm-inference.service",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"systemctl {action} failed ({completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )


def wsl_command(*arguments: str) -> list[str]:
    return [
        "wsl", "-d", "Ubuntu-24.04", "--", training.WSL_PYTHON,
        training.windows_path_to_wsl(pathlib.Path(__file__).resolve()),
        *arguments,
    ]


def qa_worker(output: str, panel: str, adapter: str) -> None:
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    qa_ids = json.loads(pathlib.Path(panel).read_text(encoding="utf-8"))["qa_ids"]
    tasks = training.load_qa_panel(training.DEFAULT_QA_PATH, qa_ids)
    if len(tasks) != len(qa_ids):
        raise ValueError("worker QA panel is incomplete")
    torch.manual_seed(20260827)
    tokenizer = AutoTokenizer.from_pretrained(
        training.BASE_MODEL_WSL, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        training.BASE_MODEL_WSL,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
    )
    preexisting = training.count_peft_modules(model)
    if preexisting:
        raise RuntimeError(f"clean base has {preexisting} PEFT modules")

    def evaluate(active_model: Any, arm: str) -> dict[str, Any]:
        samples = []
        for index, task in enumerate(tasks, 1):
            tokens = tokenizer(task["prompt"], return_tensors="pt").to("cuda")
            prompt_n = tokens["input_ids"].shape[1]
            with torch.inference_mode():
                generated = active_model.generate(
                    **tokens,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            output_ids = generated[0][prompt_n:]
            text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            correct, detail = training.grade_qa(task, text)
            samples.append({
                "panel": "qa",
                "task_id": task["id"],
                "prompt": task["prompt"],
                "output_text": text,
                "correct": correct,
                "grade_detail": detail,
                "new_tokens": len(output_ids),
                "natural_eos": bool(
                    tokenizer.eos_token_id is not None
                    and generated[0][-1].item() == tokenizer.eos_token_id
                ),
            })
            if index % 10 == 0:
                print(f"[QA] {arm} {index}/{len(tasks)}", flush=True)
        return {"arm": arm, "qa_samples": samples}

    base = evaluate(model.eval(), "base")
    adapted = PeftModel.from_pretrained(model, adapter).eval()
    treatment = evaluate(adapted, "lokr_3ep_lr1e4")
    write_json(pathlib.Path(output), {
        "qa_ids": qa_ids,
        "base_preexisting_peft_module_count": preexisting,
        "post_load_peft_module_count": training.count_peft_modules(adapted),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_vram_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "arms": [base, treatment],
    })


def merge_payload(source: dict[str, Any], fresh: dict[str, Any], qa_ids: list[str]) -> dict[str, Any]:
    merged = copy.deepcopy(source)
    source_arms = {arm["arm"]: arm for arm in merged["arms"]}
    fresh_arms = {arm["arm"]: arm for arm in fresh["arms"]}
    for arm_name in ("base", "lokr_3ep_lr1e4"):
        arm = source_arms[arm_name]
        imported_ids = [sample["task_id"] for sample in arm["qa_samples"]]
        added = fresh_arms[arm_name]["qa_samples"]
        added_ids = [sample["task_id"] for sample in added]
        if len(arm["math_samples"]) != 256 or imported_ids != qa_ids[:10]:
            raise ValueError(f"source dimensions or QA prefix invalid for {arm_name}")
        if len(added_ids) != 38 or set(imported_ids).intersection(added_ids):
            raise ValueError(f"fresh QA continuation invalid for {arm_name}")
        if imported_ids + added_ids != qa_ids:
            raise ValueError(f"merged QA ordering invalid for {arm_name}")
        arm["qa_samples"].extend(added)
        arm["qa_correct"] = sum(sample["correct"] for sample in arm["qa_samples"])
    return merged


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    finalized = raw / "finalized"
    finalized.mkdir(parents=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()
    sources = verify_sources()
    qa_ids = actual_qa_ids()
    source = json.loads(SOURCE_WORKER.read_text(encoding="utf-8"))
    source_arms = {arm["arm"]: arm for arm in source["arms"]}
    if set(source_arms) != {"base", "lokr_3ep_lr1e4"}:
        raise ValueError("source worker arms differ from preregistration")
    missing_ids = qa_ids[10:]
    panel = raw / "missing_qa_panel.json"
    write_json(panel, {
        "qa_ids": missing_ids,
        "full_qa_ids": qa_ids,
        "full_sha256": canonical_json_sha256(qa_ids),
    })
    write_json(raw / "artifact_hashes.json", sources)
    write_json(raw / "dataset_hashes.json", {
        "math": sources[training.DEFAULT_MATH_PATH.relative_to(ROOT).as_posix()],
        "qa": sources[training.DEFAULT_QA_PATH.relative_to(ROOT).as_posix()],
        "qa_ids": qa_ids,
        "qa_id_sha256": canonical_json_sha256(qa_ids),
    })
    write_json(raw / "model_hash.json", training.verify_base_model())

    initial_service = training.query_service()
    initial_gpu = training.query_gpu()
    initial_embedding = training.http_get_json("http://127.0.0.1:8081/health")
    maintenance: dict[str, Any] = {
        "initial_service": initial_service,
        "initial_gpu": initial_gpu,
        "initial_embedding": initial_embedding,
        "service_stopped_for_vram": False,
    }
    service_stopped = False
    fresh_path = raw / "fresh_qa.json"
    worker_stdout = raw / "worker.stdout.log"
    worker_stderr = raw / "worker.stderr.log"
    worker_error: Exception | None = None
    try:
        if initial_gpu["memory_free_mib"] < 12_000 and initial_service["active_state"] == "active":
            long_systemctl("stop")
            service_stopped = True
            maintenance["service_stopped_for_vram"] = True
        maintenance["service_after_stop"] = training.query_service()
        maintenance["embedding_after_stop"] = training.http_get_json("http://127.0.0.1:8081/health")
        maintenance["gpu_after_stop"] = training.query_gpu()
        if maintenance["embedding_after_stop"].get("status") != "ok":
            raise RuntimeError("embedding service became unhealthy")
        if maintenance["gpu_after_stop"]["memory_free_mib"] < 12_000:
            raise RuntimeError("insufficient free VRAM after bounded maintenance")
        command = wsl_command(
            "--worker-mode",
            "--worker-out", training.windows_path_to_wsl(fresh_path),
            "--panel", training.windows_path_to_wsl(panel),
            "--adapter", training.windows_path_to_wsl(r2.ADAPTER),
        )
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
            check=False,
        )
        worker_stdout.write_text(completed.stdout, encoding="utf-8")
        worker_stderr.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"QA worker failed ({completed.returncode}): {completed.stderr[-4000:]}")
    except Exception as error:
        worker_error = error
    finally:
        if service_stopped:
            long_systemctl("start")
            maintenance["inference_health_final"] = training.wait_for_health(
                "http://127.0.0.1:8080/health", timeout_seconds=180
            )
        maintenance["final_service"] = training.query_service()
        maintenance["final_embedding"] = training.wait_for_health(
            "http://127.0.0.1:8081/health", timeout_seconds=30
        )
        maintenance["final_gpu"] = training.query_gpu()
        maintenance["service_and_embedding_restored"] = (
            maintenance["final_service"]["active_state"] == initial_service["active_state"]
            and training.normalize_exec_start(maintenance["final_service"]["exec_start"])
            == training.normalize_exec_start(initial_service["exec_start"])
            and maintenance["final_service"]["n_restarts"] == initial_service["n_restarts"]
            and maintenance["final_embedding"].get("status") == "ok"
            and (
                not service_stopped
                or maintenance.get("inference_health_final", {}).get("status") == "ok"
            )
        )
        write_json(raw / "service_maintenance.json", maintenance)
    if worker_error is not None:
        raise worker_error
    if not fresh_path.is_file():
        raise RuntimeError("QA worker returned without fresh_qa.json")

    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    merged = merge_payload(source, fresh, qa_ids)
    write_json(raw / "merged_worker.json", merged)
    arms = {arm["arm"]: arm for arm in merged["arms"]}
    math_ids = r2.heldout_ids()
    math_map = {
        task["task_id"]: task
        for task in training.load_math_panel(training.DEFAULT_MATH_PATH, math_ids)
    }
    qa_map = {
        task["id"]: task
        for task in training.load_qa_panel(training.DEFAULT_QA_PATH, qa_ids)
    }
    independent_match = True
    differences = []
    for task_id in math_ids:
        values = {}
        for arm_name in ("base", "lokr_3ep_lr1e4"):
            sample = next(row for row in arms[arm_name]["math_samples"] if row["task_id"] == task_id)
            extracted = training.extract_gsm8k_pred(sample["output_text"])
            correct = int(training.is_gsm8k_correct(extracted, math_map[task_id]["gold"]))
            independent_match &= bool(correct) == bool(sample["correct"])
            values[arm_name] = correct
        differences.append(values["lokr_3ep_lr1e4"] - values["base"])
    for arm_name in ("base", "lokr_3ep_lr1e4"):
        for sample in arms[arm_name]["qa_samples"]:
            correct, _ = training.grade_qa(qa_map[sample["task_id"]], sample["output_text"])
            independent_match &= bool(correct) == bool(sample["correct"])
    bootstrap = r2.paired_bootstrap(differences)
    base_math = arms["base"]["math_correct"] / 256
    adapter_math = arms["lokr_3ep_lr1e4"]["math_correct"] / 256
    base_qa = arms["base"]["qa_correct"] / 48
    adapter_qa = arms["lokr_3ep_lr1e4"]["qa_correct"] / 48
    metrics = {
        "source_worker_hash_verified": sha256_file(SOURCE_WORKER) == EXPECTED_HASHES[SOURCE_WORKER],
        "artifact_hashes_verified": True,
        "teacher_disjoint_math_tasks": len(math_ids),
        "paired_math_and_qa_generations": sum(
            len(arm["math_samples"]) + len(arm["qa_samples"])
            for arm in arms.values()
        ),
        "base_math_correct": arms["base"]["math_correct"],
        "adapter_math_correct": arms["lokr_3ep_lr1e4"]["math_correct"],
        "math_gain": round(adapter_math - base_math, 8),
        "paired_bootstrap": bootstrap,
        "base_qa_correct": arms["base"]["qa_correct"],
        "adapter_qa_correct": arms["lokr_3ep_lr1e4"]["qa_correct"],
        "protected_qa_regression": round(max(0.0, base_qa - adapter_qa), 8),
        "independent_rescore_match": independent_match,
        "service_and_embedding_restored": maintenance["service_and_embedding_restored"],
    }
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "independent_evaluation.json", {
        "match": independent_match,
        "metrics": metrics,
    })
    write_json(raw / "clean_base_receipts.json", {
        "source_base_preexisting_peft_module_count": source["base_preexisting_peft_module_count"],
        "fresh_base_preexisting_peft_module_count": fresh["base_preexisting_peft_module_count"],
    })
    write_json(raw / "continuation_ledger.json", {
        "source_worker_sha256": sha256_file(SOURCE_WORKER),
        "source_receipt_sha256": sha256_file(SOURCE_RECEIPT),
        "imported_math_per_arm": 256,
        "imported_qa_per_arm": 10,
        "fresh_qa_per_arm": 38,
        "source_worker_hash_verified": metrics["source_worker_hash_verified"],
    })
    write_json(raw / "isolation_smoke.json", {
        "teacher_ids_disjoint": True,
        "math_id_sha256": canonical_json_sha256(math_ids),
        "qa_id_sha256": canonical_json_sha256(qa_ids),
        "fresh_and_imported_qa_disjoint": True,
    })
    write_json(raw / "paired_baseline.json", {
        "base": {"math_correct": metrics["base_math_correct"], "qa_correct": metrics["base_qa_correct"]},
        "adapter": {"math_correct": metrics["adapter_math_correct"], "qa_correct": metrics["adapter_qa_correct"]},
    })
    write_json(raw / "scorer_hashes.json", {
        "runner": sha256_file(pathlib.Path(__file__).resolve()),
        "r2_runner": sha256_file(ROOT / "tools/research/run_adapt01_broad_artifact_eval_r2.py"),
        "math": sha256_file(ROOT / "tools/analysis/a2_stats.py"),
        "qa": sha256_file(ROOT / "tools/benchmarks/normal_qa_ab.py"),
    })
    write_json(raw / "source_execution_receipt.json", {
        "source_task_id": "BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-02",
        "source_worker_sha256": sha256_file(SOURCE_WORKER),
        "source_receipt_sha256": sha256_file(SOURCE_RECEIPT),
    })
    write_json(raw / "wsl_environment.json", fresh["versions"] | {
        "gpu": fresh["gpu"],
        "peak_allocated_vram_gib": fresh["peak_allocated_vram_gib"],
    })
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for arm in merged["arms"]:
            for sample in [*arm["math_samples"], *arm["qa_samples"]]:
                stream.write(json.dumps({"arm": arm["arm"], **sample}, ensure_ascii=False) + "\n")

    gates = {
        "continuation_integrity": {"metric": "source_worker_hash_verified", "operator": "eq", "threshold": True, "actual": metrics["source_worker_hash_verified"], "pass": metrics["source_worker_hash_verified"] is True},
        "artifact_identity": {"metric": "artifact_hashes_verified", "operator": "eq", "threshold": True, "actual": True, "pass": True},
        "panel_isolation": {"metric": "teacher_disjoint_math_tasks", "operator": "eq", "threshold": 256, "actual": metrics["teacher_disjoint_math_tasks"], "pass": metrics["teacher_disjoint_math_tasks"] == 256},
        "evaluation_coverage": {"metric": "paired_math_and_qa_generations", "operator": "eq", "threshold": 608, "actual": metrics["paired_math_and_qa_generations"], "pass": metrics["paired_math_and_qa_generations"] == 608},
        "broad_gain": {"metric": "paired_bootstrap_95ci_lower_math_gain", "operator": "gt", "threshold": 0.0, "actual": bootstrap["lower_95"], "pass": bootstrap["lower_95"] > 0.0},
        "protected_retention": {"metric": "protected_qa_regression", "operator": "le", "threshold": 0.05, "actual": metrics["protected_qa_regression"], "pass": metrics["protected_qa_regression"] <= 0.05},
        "independent_score": {"metric": "independent_rescore_match", "operator": "eq", "threshold": True, "actual": independent_match, "pass": independent_match is True},
        "service_recovery": {"metric": "service_and_embedding_restored", "operator": "eq", "threshold": True, "actual": maintenance["service_and_embedding_restored"], "pass": maintenance["service_and_embedding_restored"] is True},
    }
    evidence = {
        "acceptance_gates": "raw/receipt.json",
        "actual_scores": "raw/actual_scores.json",
        "artifact_hashes": "raw/artifact_hashes.json",
        "clean_base_receipts": "raw/clean_base_receipts.json",
        "continuation_ledger": "raw/continuation_ledger.json",
        "dataset_hashes": "raw/dataset_hashes.json",
        "independent_evaluation": "raw/independent_evaluation.json",
        "isolation_smoke": "raw/isolation_smoke.json",
        "model_hash": "raw/model_hash.json",
        "paired_baseline": "raw/paired_baseline.json",
        "provenance": "raw/receipt.json",
        "raw_samples": "raw/samples.jsonl",
        "receipt_fingerprint": "raw/receipt.json",
        "scorer_hashes": "raw/scorer_hashes.json",
        "service_maintenance": "raw/service_maintenance.json",
        "source_execution_receipt": "raw/source_execution_receipt.json",
        "wsl_environment": "raw/wsl_environment.json",
    }
    evidence_files = sorted({
        raw / value.removeprefix("raw/")
        for value in evidence.values()
        if value != "raw/receipt.json"
    })
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(),
        started_at_utc=started,
        started_monotonic=mono,
        input_paths=[*EXPECTED_HASHES.keys(), PREREGISTRATION, fresh_path, panel, *evidence_files],
        packages=["pytest"],
        runtime={
            "execution_mode": "hash_bound_partial_qa_continuation",
            "host_pid": os.getpid(),
            "timing_is_evidence": False,
        },
    )
    complete, errors = provenance_complete(provenance)
    if not complete or not independent_match:
        raise ValueError(f"evidence validation failed: provenance={errors}, scorer_match={independent_match}")
    receipt = {
        "schema": "local-labs-backlog-receipt-v1",
        "task_id": TASK_ID,
        "provenance": provenance,
        "provenance_complete": True,
        "gates": gates,
        "evidence": evidence,
    }
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = (
        "ADAPT01_384_ARTIFACT_BROAD_GAIN_R3"
        if not failed
        else "ADAPT01_384_ARTIFACT_BROAD_GAIN_NOT_CONFIRMED_R3"
    )
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Imported 256 math and 10 QA samples per arm; generated the missing 38 QA "
        f"samples per arm. Base math `{metrics['base_math_correct']}/256`; adapter math "
        f"`{metrics['adapter_math_correct']}/256`; gain `{metrics['math_gain']:.6f}` with "
        f"paired bootstrap 95% interval `[{bootstrap['lower_95']:.6f}, "
        f"{bootstrap['upper_95']:.6f}]`. Base QA `{metrics['base_qa_correct']}/48`; "
        f"adapter QA `{metrics['adapter_qa_correct']}/48`. Failed gates: "
        f"`{', '.join(failed) if failed else 'none'}`.\n",
        encoding="utf-8",
    )
    write_json(finalized / "complete.json", {
        "task_id": TASK_ID,
        "receipt_fingerprint": receipt["receipt_fingerprint"],
        "failed_gates": failed,
    })
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-out")
    parser.add_argument("--panel")
    parser.add_argument("--adapter")
    args = parser.parse_args()
    if args.worker_mode:
        if not all((args.worker_out, args.panel, args.adapter)):
            parser.error("worker mode requires output, panel and adapter")
        qa_worker(args.worker_out, args.panel, args.adapter)
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
