#!/usr/bin/env python3
"""Broad paired base-versus-LoKr artifact evaluation under controlled GPU co-tenancy."""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file
from tools.research import run_trace_distillation_training_r2 as r2

TASK_ID = "BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-02"
MODEL = r2.BASE_MODEL_WSL
PYTHON = r2.WSL_PYTHON
PARENT = ROOT / "runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01"
ADAPTER = PARENT / "raw/mechanisms/adapt01/lokr_3ep_lr1e4/adapter"
METRICS = PARENT / "raw/mechanisms/adapt01/lokr_3ep_lr1e4/metrics.json"
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-02.json"
EXPECTED_HASHES = {
    ADMISSION: "1691233003e4ab97b9986c035e78581e4ef04b565f766dd3e19a1449e4d81b8c",
    PARENT / "raw/receipt.json": "dd975197993bab7943ea2407a664f20fda927bb6fc581714eba093f7e93be0c6",
    ADAPTER / "adapter_model.safetensors": "7f6d082243f6b406259791dc15a65e4b092b48597fad9b68018d507872ad8fa7",
    ADAPTER / "adapter_config.json": "08cf4d254e2a6c9aba9d34ba6a0c76926b478d7cd0ad771062acefb71a31d934",
    METRICS: "09d6295f934843fa85cd2a4757a1b045695e1b83c8a56c19c73c3d8bbecc0a9c",
    r2.DEFAULT_MATH_PATH: "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    r2.TEACHER_PATH: "dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e",
    r2.DEFAULT_QA_PATH: "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
}
HELDOUT_HASH = "78a3b7ef26cdf932b79eb6f64dfb576d66770d30a5ce0fd3251536cb6e76901f"
QA_IDS = [f"f{index:02d}" for index in range(1, 49)]


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def heldout_ids() -> list[str]:
    teacher = json.loads(r2.TEACHER_PATH.read_text(encoding="utf-8"))
    teacher_ids = {row["task_id"] for row in teacher}
    ids = [f"gsm8k/{index}" for index in range(1319) if f"gsm8k/{index}" not in teacher_ids][:256]
    if len(ids) != 256 or canonical_json_sha256(ids) != HELDOUT_HASH:
        raise ValueError("held-out panel differs from preregistration")
    return ids


def verify_sources() -> dict[str, dict[str, Any]]:
    ledger: dict[str, dict[str, Any]] = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"source mismatch: {path}: {actual} != {expected}")
        ledger[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": actual}
    return ledger


def process_alive(pid: int) -> bool:
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"if(Get-Process -Id {pid} -ErrorAction SilentlyContinue){{exit 0}}else{{exit 1}}"],
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def http_status(port: int) -> int | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            return response.status
    except Exception:
        return None


def gpu_free_mib() -> int:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(completed.stdout.strip().splitlines()[0])


def paired_bootstrap(differences: list[int], replicates: int = 20_000) -> dict[str, Any]:
    if len(differences) != 256:
        raise ValueError("paired bootstrap requires 256 prompt differences")
    rng = random.Random(2026082602)
    estimates = [sum(differences[rng.randrange(256)] for _ in range(256)) / 256 for _ in range(replicates)]
    estimates.sort()
    return {
        "replicates": replicates,
        "seed": 2026082602,
        "lower_95": round(estimates[int(0.025 * replicates)], 8),
        "upper_95": round(estimates[min(replicates - 1, int(0.975 * replicates))], 8),
    }


def worker(output: str, panel: str, adapter: str) -> None:
    import torch
    import peft
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    frozen = json.loads(pathlib.Path(panel).read_text(encoding="utf-8"))
    math_ids = frozen["math_ids"]
    qa_ids = frozen["qa_ids"]
    torch.manual_seed(20260827)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa",
    )
    preexisting = r2.count_peft_modules(model)
    if preexisting:
        raise RuntimeError(f"clean base has {preexisting} PEFT modules")

    def generate(active_model: Any, prompt: str, maximum: int) -> tuple[str, int, bool]:
        tokens = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_n = tokens["input_ids"].shape[1]
        with torch.inference_mode():
            generated = active_model.generate(
                **tokens, max_new_tokens=maximum, do_sample=False,
                temperature=None, top_p=None,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )
        output_ids = generated[0][prompt_n:]
        return (
            tokenizer.decode(output_ids, skip_special_tokens=True).strip(),
            len(output_ids),
            bool(tokenizer.eos_token_id is not None and generated[0][-1].item() == tokenizer.eos_token_id),
        )

    math_tasks = r2.load_math_panel(r2.DEFAULT_MATH_PATH, math_ids)
    qa_tasks = r2.load_qa_panel(r2.DEFAULT_QA_PATH, qa_ids)

    def evaluate(active_model: Any, arm: str) -> dict[str, Any]:
        math_samples: list[dict[str, Any]] = []
        for index, task in enumerate(math_tasks, 1):
            text, new_tokens, eos = generate(active_model, task["prompt"], 192)
            extracted = r2.extract_gsm8k_pred(text)
            math_samples.append({
                "panel": "math", "task_id": task["task_id"], "prompt": task["prompt"],
                "gold": task["gold"], "output_text": text, "extracted": extracted,
                "correct": r2.is_gsm8k_correct(extracted, task["gold"]),
                "new_tokens": new_tokens, "natural_eos": eos,
            })
            if index % 32 == 0:
                print(f"[WORKER] {arm} math {index}/256", flush=True)
        qa_samples: list[dict[str, Any]] = []
        for task in qa_tasks:
            text, new_tokens, eos = generate(active_model, task["prompt"], 128)
            correct, detail = r2.grade_qa(task, text)
            qa_samples.append({
                "panel": "qa", "task_id": task["id"], "prompt": task["prompt"],
                "output_text": text, "correct": correct, "grade_detail": detail,
                "new_tokens": new_tokens, "natural_eos": eos,
            })
        return {
            "arm": arm,
            "math_samples": math_samples,
            "qa_samples": qa_samples,
            "math_correct": sum(sample["correct"] for sample in math_samples),
            "qa_correct": sum(sample["correct"] for sample in qa_samples),
        }

    base = evaluate(model.eval(), "base")
    adapted = PeftModel.from_pretrained(model, adapter).eval()
    treatment = evaluate(adapted, "lokr_3ep_lr1e4")
    payload = {
        "base_preexisting_peft_module_count": preexisting,
        "post_load_peft_module_count": r2.count_peft_modules(adapted),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_vram_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "arms": [base, treatment],
    }
    pathlib.Path(output).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def execute(outdir: pathlib.Path, outer_runner_pid: int) -> dict[str, Any]:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()
    sources = verify_sources()
    model_hash = r2.verify_base_model()
    math_ids = heldout_ids()
    panel = raw / "panel.json"
    write_json(panel, {"math_ids": math_ids, "qa_ids": QA_IDS, "math_id_sha256": canonical_json_sha256(math_ids)})
    write_json(raw / "artifact_hashes.json", sources)
    write_json(raw / "dataset_hashes.json", {key: value for key, value in sources.items() if key.endswith(("gsm8k.jsonl", "gsm8k.json", "tasks.jsonl"))})
    write_json(raw / "model_hash.json", model_hash)

    before = {
        "outer_runner_alive": process_alive(outer_runner_pid),
        "http_8080": http_status(8080),
        "http_8081": http_status(8081),
        "gpu_free_mib": gpu_free_mib(),
    }
    if not before["outer_runner_alive"] or before["http_8080"] is not None or before["http_8081"] != 200:
        raise RuntimeError(f"invalid outer maintenance boundary: {before}")
    if before["gpu_free_mib"] < 12_000:
        raise RuntimeError(f"insufficient free VRAM for co-tenant: {before['gpu_free_mib']} MiB")

    worker_script = pathlib.Path(__file__).resolve()
    worker_json = raw / "worker.json"
    command = [
        "wsl", "-d", "Ubuntu-24.04", "--", PYTHON, r2.windows_path_to_wsl(worker_script),
        "--worker-mode", "--worker-out", r2.windows_path_to_wsl(worker_json),
        "--panel", r2.windows_path_to_wsl(panel), "--adapter", r2.windows_path_to_wsl(ADAPTER),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200, check=False)
    (raw / "worker.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (raw / "worker.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"worker failed: {completed.stderr[-4000:]}")
    payload = json.loads(worker_json.read_text(encoding="utf-8"))
    after = {
        "outer_runner_alive": process_alive(outer_runner_pid),
        "http_8080": http_status(8080),
        "http_8081": http_status(8081),
        "gpu_free_mib": gpu_free_mib(),
    }

    arms = {arm["arm"]: arm for arm in payload["arms"]}
    math_map = {task["task_id"]: task for task in r2.load_math_panel(r2.DEFAULT_MATH_PATH, math_ids)}
    qa_map = {task["id"]: task for task in r2.load_qa_panel(r2.DEFAULT_QA_PATH, QA_IDS)}
    independent_match = True
    differences: list[int] = []
    for task_id in math_ids:
        values: dict[str, int] = {}
        for arm_name in ("base", "lokr_3ep_lr1e4"):
            sample = next(row for row in arms[arm_name]["math_samples"] if row["task_id"] == task_id)
            extracted = r2.extract_gsm8k_pred(sample["output_text"])
            correct = int(r2.is_gsm8k_correct(extracted, math_map[task_id]["gold"]))
            independent_match &= bool(correct) == bool(sample["correct"])
            values[arm_name] = correct
        differences.append(values["lokr_3ep_lr1e4"] - values["base"])
    for arm_name in ("base", "lokr_3ep_lr1e4"):
        for sample in arms[arm_name]["qa_samples"]:
            correct, _ = r2.grade_qa(qa_map[sample["task_id"]], sample["output_text"])
            independent_match &= bool(correct) == bool(sample["correct"])
    base_math = arms["base"]["math_correct"] / 256
    adapter_math = arms["lokr_3ep_lr1e4"]["math_correct"] / 256
    base_qa = arms["base"]["qa_correct"] / 48
    adapter_qa = arms["lokr_3ep_lr1e4"]["qa_correct"] / 48
    bootstrap = paired_bootstrap(differences)
    metrics = {
        "artifact_hashes_verified": True,
        "teacher_disjoint_math_tasks": len(math_ids),
        "paired_math_and_qa_generations": sum(len(arm["math_samples"]) + len(arm["qa_samples"]) for arm in arms.values()),
        "base_math_correct": arms["base"]["math_correct"],
        "adapter_math_correct": arms["lokr_3ep_lr1e4"]["math_correct"],
        "math_gain": round(adapter_math - base_math, 8),
        "paired_bootstrap": bootstrap,
        "base_qa_correct": arms["base"]["qa_correct"],
        "adapter_qa_correct": arms["lokr_3ep_lr1e4"]["qa_correct"],
        "protected_qa_regression": round(max(0.0, base_qa - adapter_qa), 8),
        "outer_runner_alive_and_embedding_healthy": bool(after["outer_runner_alive"] and after["http_8080"] is None and after["http_8081"] == 200),
    }
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "independent_evaluation.json", {"match": independent_match, "metrics": metrics})
    write_json(raw / "clean_base_receipts.json", {"base_preexisting_peft_module_count": payload["base_preexisting_peft_module_count"]})
    write_json(raw / "isolation_smoke.json", {"teacher_ids_disjoint": True, "math_id_sha256": canonical_json_sha256(math_ids), "prompt_pairing": True})
    write_json(raw / "paired_baseline.json", {"base": {"math_correct": metrics["base_math_correct"], "qa_correct": metrics["base_qa_correct"]}, "adapter": {"math_correct": metrics["adapter_math_correct"], "qa_correct": metrics["adapter_qa_correct"]}})
    write_json(raw / "scorer_hashes.json", {"runner": sha256_file(worker_script), "math": sha256_file(ROOT / "tools/analysis/a2_stats.py"), "qa": sha256_file(ROOT / "tools/benchmarks/normal_qa_ab.py")})
    write_json(raw / "service_maintenance.json", {"ownership": "outer BACKLOG-ADAPT-TRACE-DISTILL-05 runner", "service_mutations": 0, "before": before, "after": after})
    write_json(raw / "source_execution_receipt.json", {"parent_receipt_sha256": sources["runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/receipt.json"]["sha256"]})
    write_json(raw / "wsl_environment.json", payload["versions"] | {"gpu": payload["gpu"], "peak_allocated_vram_gib": payload["peak_allocated_vram_gib"]})
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for arm in payload["arms"]:
            for sample in [*arm["math_samples"], *arm["qa_samples"]]:
                stream.write(json.dumps({"arm": arm["arm"], **sample}, ensure_ascii=False) + "\n")

    gates = {
        "artifact_identity": {"metric": "artifact_hashes_verified", "operator": "eq", "threshold": True, "actual": True, "pass": True},
        "panel_isolation": {"metric": "teacher_disjoint_math_tasks", "operator": "eq", "threshold": 256, "actual": metrics["teacher_disjoint_math_tasks"], "pass": metrics["teacher_disjoint_math_tasks"] == 256},
        "evaluation_coverage": {"metric": "paired_math_and_qa_generations", "operator": "eq", "threshold": 608, "actual": metrics["paired_math_and_qa_generations"], "pass": metrics["paired_math_and_qa_generations"] == 608},
        "broad_gain": {"metric": "paired_bootstrap_95ci_lower_math_gain", "operator": "gt", "threshold": 0.0, "actual": bootstrap["lower_95"], "pass": bootstrap["lower_95"] > 0.0},
        "protected_retention": {"metric": "protected_qa_regression", "operator": "le", "threshold": 0.05, "actual": metrics["protected_qa_regression"], "pass": metrics["protected_qa_regression"] <= 0.05},
        "co_tenancy_boundary": {"metric": "outer_runner_alive_and_embedding_healthy", "operator": "eq", "threshold": True, "actual": metrics["outer_runner_alive_and_embedding_healthy"], "pass": metrics["outer_runner_alive_and_embedding_healthy"] is True},
    }
    evidence = {
        "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
        "artifact_hashes": "raw/artifact_hashes.json", "clean_base_receipts": "raw/clean_base_receipts.json",
        "dataset_hashes": "raw/dataset_hashes.json", "independent_evaluation": "raw/independent_evaluation.json",
        "isolation_smoke": "raw/isolation_smoke.json", "model_hash": "raw/model_hash.json",
        "paired_baseline": "raw/paired_baseline.json", "provenance": "raw/receipt.json",
        "raw_samples": "raw/samples.jsonl", "receipt_fingerprint": "raw/receipt.json",
        "scorer_hashes": "raw/scorer_hashes.json", "service_maintenance": "raw/service_maintenance.json",
        "source_execution_receipt": "raw/source_execution_receipt.json", "wsl_environment": "raw/wsl_environment.json",
    }
    evidence_files = sorted({raw / value.removeprefix("raw/") for value in evidence.values() if value != "raw/receipt.json"})
    provenance = build_provenance(
        script_path=worker_script, started_at_utc=started, started_monotonic=mono,
        input_paths=[*EXPECTED_HASHES.keys(), worker_json, panel, *evidence_files], packages=["pytest"],
        runtime={"execution_mode": "gpu_cotenant_artifact_evaluation", "outer_runner_pid": outer_runner_pid, "timing_is_evidence": False},
    )
    complete, errors = provenance_complete(provenance)
    if not complete or not independent_match:
        raise ValueError(f"evidence validation failed: provenance={errors}, scorer_match={independent_match}")
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = "ADAPT01_384_ARTIFACT_BROAD_GAIN_R2" if not failed else "ADAPT01_384_ARTIFACT_BROAD_GAIN_NOT_CONFIRMED_R2"
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Base math `{metrics['base_math_correct']}/256`; adapter math `{metrics['adapter_math_correct']}/256`; "
        f"gain `{metrics['math_gain']:.6f}` with paired bootstrap 95% interval "
        f"`[{bootstrap['lower_95']:.6f}, {bootstrap['upper_95']:.6f}]`. "
        f"Base QA `{metrics['base_qa_correct']}/48`; adapter QA `{metrics['adapter_qa_correct']}/48`. "
        f"Failed gates: `{', '.join(failed) if failed else 'none'}`. Timing is not evidence.\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--outer-runner-pid", type=int)
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-out")
    parser.add_argument("--panel")
    parser.add_argument("--adapter")
    args = parser.parse_args()
    if args.worker_mode:
        if not all((args.worker_out, args.panel, args.adapter)):
            parser.error("worker mode requires output, panel and adapter")
        worker(args.worker_out, args.panel, args.adapter)
        return 0
    if args.outer_runner_pid is None:
        parser.error("host mode requires --outer-runner-pid")
    receipt = execute(args.outdir.resolve(), args.outer_runner_pid)
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
