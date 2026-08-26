#!/usr/bin/env python3
"""Process-isolated adapter requalification for BACKLOG-ADAPT-REQUAL-02."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
import urllib.request
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)
from tools.research.run_adapter_requalification import (
    ADAPTER_SPECS,
    BASE_MODEL_WSL,
    DEFAULT_MATH_PATH,
    DEFAULT_QA_PATH,
    FROZEN_GSM8K_IDS,
    FROZEN_QA_IDS,
    extract_gsm8k_pred,
    grade_qa,
    is_gsm8k_correct,
    load_math_panel,
    load_qa_panel,
)

TASK_ID = "BACKLOG-ADAPT-REQUAL-02"
WSL_DISTRO = "Ubuntu-24.04"
WSL_PYTHON = "/home/augus/.venvs/adapt00-20260824/bin/python"
SMOKE_ORDER_A = ["base", "lokr_1ep", "target_mlp_only"]
SMOKE_ORDER_B = list(reversed(SMOKE_ORDER_A))
EXPECTED_WORKERS = 17
EXPECTED_BASE_HASHES = {
    "model.safetensors-00001-of-00001.safetensors": "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c",
    "config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
}
FROZEN_REPO_INPUTS = {
    "config/research_backlog_admissions/BACKLOG-ADAPT-REQUAL-02.json": "789eaff0490d7d8f939d3af2445551400b22ae2432ea2a67de07abc616a64f8a",
    "docs/AUDIT_2026-08-25_CODEX_INDEPENDENT_AGY_EXECUTION.md": "e4364456156a3c2f015306d986192792fb1aa9ae9333b63a2237ec46e3ffc11f",
    "runs/research/BACKLOG-ADAPT-REQUAL-01/raw/artifact_hashes.json": "b19fa60e5d122219934a1563cdf231dac0a847393327d35b214763711582c5fc",
    "runs/research/BACKLOG-ADAPT-REQUAL-01/raw/dataset_hashes.json": "ec389b0a3eb63460edd92eef26ca3361966f0bd6b869cd1b47078b008ee2d652",
    "workloads/gsm8k.jsonl": "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    "runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl": "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
}
ARTIFACT_LEDGER_PATH = ROOT / "runs" / "research" / "BACKLOG-ADAPT-REQUAL-01" / "raw" / "artifact_hashes.json"


def windows_path_to_wsl(path: pathlib.Path) -> str:
    resolved = str(path.resolve())
    if len(resolved) < 3 or resolved[1:3] != ":\\":
        raise ValueError(f"expected a Windows drive path, got {resolved}")
    return f"/mnt/{resolved[0].lower()}/{resolved[3:].replace(chr(92), '/')}"


def http_get_json(url: str, timeout: float = 5.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "AdapterRequalR2/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object from {url}")
    return value


def run_command(command: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
    )


def query_service() -> dict:
    completed = run_command(
        [
            "wsl", "-d", WSL_DISTRO, "--", "systemctl", "show", "llm-inference.service",
            "--property=MainPID,NRestarts,ActiveState,SubState,ExecStart",
        ]
    )
    props: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            props[key] = value
    return {
        "returncode": completed.returncode,
        "main_pid": int(props.get("MainPID", "0") or 0),
        "n_restarts": int(props.get("NRestarts", "0") or 0),
        "active_state": props.get("ActiveState", ""),
        "sub_state": props.get("SubState", ""),
        "exec_start": props.get("ExecStart", ""),
        "stderr": completed.stderr.strip(),
    }


def query_gpu() -> dict:
    completed = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr}")
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if len(fields) != 5:
        raise ValueError(f"unexpected nvidia-smi output: {completed.stdout!r}")
    return {
        "name": fields[0],
        "memory_total_mib": int(fields[1]),
        "memory_used_mib": int(fields[2]),
        "memory_free_mib": int(fields[3]),
        "utilization_percent": int(fields[4]),
    }


def systemctl(action: str) -> None:
    completed = run_command(
        ["wsl", "-d", WSL_DISTRO, "-u", "root", "--", "systemctl", action, "llm-inference.service"],
        timeout=60.0,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"systemctl {action} failed: {completed.stderr or completed.stdout}")


def wait_for_health(url: str, *, timeout_seconds: float) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            value = http_get_json(url, timeout=3.0)
            if value.get("status") == "ok":
                return value
            last_error = f"unexpected payload: {value!r}"
        except Exception as exc:  # bounded recovery polling
            last_error = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"health did not recover at {url}: {last_error}")


def verify_frozen_inputs() -> list[pathlib.Path]:
    verified: list[pathlib.Path] = []
    for relative, expected in FROZEN_REPO_INPUTS.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen input mismatch: {relative}: {actual} != {expected}")
        verified.append(path)
    return verified


def verify_artifact_ledger() -> tuple[dict, list[pathlib.Path]]:
    ledger = json.loads(ARTIFACT_LEDGER_PATH.read_text(encoding="utf-8"))
    if set(ledger) != {spec["id"] for spec in ADAPTER_SPECS}:
        raise ValueError("artifact ledger arms do not match frozen adapter specs")
    verified_paths: list[pathlib.Path] = []
    for spec in ADAPTER_SPECS:
        arm = spec["id"]
        entry = ledger[arm]
        expected_relative = spec["rel_path"]
        if entry.get("relative_path") != expected_relative:
            raise ValueError(f"artifact path mismatch for {arm}")
        directory = ROOT / expected_relative
        checks = [
            (directory / "adapter_config.json", entry["config"]),
            (directory / "adapter_model.safetensors", entry["safetensors"]),
        ]
        for path, frozen in checks:
            if path.stat().st_size != frozen["bytes"]:
                raise ValueError(f"artifact byte-size mismatch: {path}")
            if sha256_file(path) != frozen["sha256"]:
                raise ValueError(f"artifact SHA-256 mismatch: {path}")
            verified_paths.append(path)
    return ledger, verified_paths


def verify_base_model() -> dict:
    paths = [f"{BASE_MODEL_WSL}/{name}" for name in EXPECTED_BASE_HASHES]
    completed = run_command(["wsl", "-d", WSL_DISTRO, "--", "sha256sum", *paths], timeout=120.0)
    if completed.returncode != 0:
        raise RuntimeError(f"base model hash failed: {completed.stderr}")
    observed: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        digest, raw_path = line.split(maxsplit=1)
        observed[pathlib.PurePosixPath(raw_path).name] = digest
    if observed != EXPECTED_BASE_HASHES:
        raise ValueError(f"base model identity mismatch: {observed!r}")
    return {"model_path": BASE_MODEL_WSL, "files": observed}


def capture_wsl_environment() -> dict:
    probe = (
        "import json,platform,torch,transformers,peft;"
        "print(json.dumps({'python':platform.python_version(),'platform':platform.platform(),"
        "'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,"
        "'cuda_available':torch.cuda.is_available(),'cuda':torch.version.cuda,"
        "'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    )
    completed = run_command(["wsl", "-d", WSL_DISTRO, "--", WSL_PYTHON, "-c", probe], timeout=30.0)
    if completed.returncode != 0:
        raise RuntimeError(f"WSL environment probe failed: {completed.stderr}")
    return json.loads(completed.stdout)


def _peft_module_count(model: Any) -> int:
    count = 0
    for module in model.modules():
        module_name = module.__class__.__module__
        class_name = module.__class__.__name__.casefold()
        if module_name.startswith("peft.") or any(token in class_name for token in ("lora", "lokr", "promptembedding")):
            count += 1
    return count


def execute_worker(arm: str, output_path: str, adapter_path: str | None) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if arm != "base" and not adapter_path:
        raise ValueError("adapter_path is required for an adapter worker")
    torch.manual_seed(20260824)
    torch.cuda.manual_seed_all(20260824)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_WSL, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_WSL,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    base_model.eval()
    preexisting = _peft_module_count(base_model)
    if preexisting != 0:
        raise RuntimeError(f"fresh base contains {preexisting} PEFT/tuner modules")

    model = base_model
    peft_type = None
    if arm != "base":
        config_path = pathlib.Path(adapter_path) / "adapter_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        peft_type = config.get("peft_type")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

    math_tasks = load_math_panel(DEFAULT_MATH_PATH, FROZEN_GSM8K_IDS)
    qa_tasks = load_qa_panel(DEFAULT_QA_PATH, FROZEN_QA_IDS)

    def generate(prompt: str, max_new_tokens: int) -> tuple[str, int, bool, float]:
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = encoded["input_ids"].shape[1]
        started = time.monotonic()
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.monotonic() - started
        output_ids = generated[0][input_length:]
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        natural_eos = bool(generated[0][-1].item() == tokenizer.eos_token_id)
        return text, len(output_ids), natural_eos, round(elapsed, 4)

    samples: list[dict] = []
    for task in math_tasks:
        text, tokens, natural_eos, elapsed = generate(task["prompt"], 192)
        extracted = extract_gsm8k_pred(text)
        samples.append(
            {
                "panel": "math",
                "task_id": task["task_id"],
                "prompt": task["prompt"],
                "gold": task["gold"],
                "extracted": extracted,
                "correct": is_gsm8k_correct(extracted, task["gold"]),
                "output_text": text,
                "new_tokens": tokens,
                "natural_eos": natural_eos,
                "elapsed_s": elapsed,
            }
        )
    for task in qa_tasks:
        text, tokens, natural_eos, elapsed = generate(task["prompt"], 128)
        correct, detail = grade_qa(task, text)
        samples.append(
            {
                "panel": "qa",
                "task_id": task["id"],
                "category": task["category"],
                "prompt": task["prompt"],
                "correct": correct,
                "grade_detail": detail,
                "output_text": text,
                "new_tokens": tokens,
                "natural_eos": natural_eos,
                "elapsed_s": elapsed,
            }
        )

    payload = {
        "arm": arm,
        "pid": os.getpid(),
        "base_preexisting_peft_module_count": preexisting,
        "loaded_adapter_peft_type": peft_type,
        "post_load_peft_module_count": _peft_module_count(model),
        "sample_count": len(samples),
        "samples": samples,
    }
    pathlib.Path(output_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def semantic_samples(payload: dict) -> list[dict]:
    fields = (
        "panel", "task_id", "prompt", "gold", "category", "extracted", "correct",
        "grade_detail", "output_text", "new_tokens", "natural_eos",
    )
    return [{key: sample[key] for key in fields if key in sample} for sample in payload["samples"]]


def compare_smoke(order_a: dict[str, dict], order_b: dict[str, dict]) -> dict:
    arms: dict[str, dict] = {}
    all_match = True
    for arm in SMOKE_ORDER_A:
        projection_a = semantic_samples(order_a[arm])
        projection_b = semantic_samples(order_b[arm])
        digest_a = canonical_json_sha256(projection_a)
        digest_b = canonical_json_sha256(projection_b)
        match = projection_a == projection_b
        all_match = all_match and match
        arms[arm] = {"order_a_sha256": digest_a, "order_b_sha256": digest_b, "semantic_match": match}
    return {"order_a": SMOKE_ORDER_A, "order_b": SMOKE_ORDER_B, "arms": arms, "order_invariant": all_match}


def score_samples(samples: list[dict]) -> tuple[dict, bool]:
    qa_by_id = {task["id"]: task for task in load_qa_panel(DEFAULT_QA_PATH, FROZEN_QA_IDS)}
    scores: dict[str, dict[str, int]] = {}
    match = True
    for sample in samples:
        arm = sample["arm"]
        scores.setdefault(arm, {"math_correct": 0, "math_total": 0, "qa_correct": 0, "qa_total": 0})
        if sample["panel"] == "math":
            scores[arm]["math_total"] += 1
            extracted = extract_gsm8k_pred(sample["output_text"])
            correct = is_gsm8k_correct(extracted, sample["gold"])
            scores[arm]["math_correct"] += int(correct)
        else:
            scores[arm]["qa_total"] += 1
            correct, _ = grade_qa(qa_by_id[sample["task_id"]], sample["output_text"])
            scores[arm]["qa_correct"] += int(correct)
        if correct != sample["correct"]:
            match = False
    return scores, match


def worker_command(arm: str, worker_out: pathlib.Path) -> list[str]:
    command = [
        "wsl", "-d", WSL_DISTRO, "--", WSL_PYTHON, windows_path_to_wsl(pathlib.Path(__file__)),
        "--worker-mode", "--worker-arm", arm, "--worker-out", windows_path_to_wsl(worker_out),
    ]
    if arm != "base":
        spec = next(item for item in ADAPTER_SPECS if item["id"] == arm)
        command.extend(["--adapter-path", windows_path_to_wsl(ROOT / spec["rel_path"])])
    return command


def run_worker(arm: str, label: str, workers_dir: pathlib.Path) -> dict:
    output_path = workers_dir / f"{label}.json"
    command = worker_command(arm, output_path)
    started = time.monotonic()
    completed = run_command(command, timeout=900.0)
    elapsed = time.monotonic() - started
    (workers_dir / f"{label}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (workers_dir / f"{label}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"worker {label} failed ({completed.returncode}): {completed.stderr[-2000:]}")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if payload.get("arm") != arm or payload.get("sample_count") != 48:
        raise ValueError(f"worker {label} returned an invalid payload")
    payload["host_worker_label"] = label
    payload["host_elapsed_seconds"] = round(elapsed, 3)
    payload["command"] = command
    return payload


def run_experiment(outdir: pathlib.Path) -> dict:
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    raw_dir = outdir / "raw"
    if any(raw_dir.iterdir()):
        raise RuntimeError(f"raw output directory is not empty: {raw_dir}")
    workers_dir = raw_dir / "workers"
    workers_dir.mkdir(parents=True)

    frozen_paths = verify_frozen_inputs()
    artifact_ledger, artifact_paths = verify_artifact_ledger()
    base_ledger = verify_base_model()
    wsl_environment = capture_wsl_environment()
    (raw_dir / "artifact_hashes.json").write_text(
        json.dumps(artifact_ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    dataset_ledger = {
        "math_panel": {"path": str(DEFAULT_MATH_PATH.relative_to(ROOT).as_posix()), "sha256": sha256_file(DEFAULT_MATH_PATH), "sample_count": 32},
        "qa_panel": {"path": str(DEFAULT_QA_PATH.relative_to(ROOT).as_posix()), "sha256": sha256_file(DEFAULT_QA_PATH), "sample_count": 16},
        "base_model": base_ledger,
    }
    (raw_dir / "dataset_hashes.json").write_text(
        json.dumps(dataset_ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (raw_dir / "wsl_environment.json").write_text(
        json.dumps(wsl_environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    scorer_ledger = {
        "runner": {"path": "tools/research/run_adapter_requalification_r2.py", "sha256": sha256_file(pathlib.Path(__file__))},
        "shared_scorer": {"path": "tools/research/run_adapter_requalification.py", "sha256": sha256_file(ROOT / "tools/research/run_adapter_requalification.py")},
        "scorers": ["extract_gsm8k_pred", "is_gsm8k_correct", "grade_qa"],
    }
    (raw_dir / "scorer_hashes.json").write_text(
        json.dumps(scorer_ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    initial_service = query_service()
    initial_gpu = query_gpu()
    initial_embedding = http_get_json("http://127.0.0.1:8081/health")
    service_stopped = False
    maintenance: dict[str, Any] = {
        "initial_service": initial_service,
        "initial_gpu": initial_gpu,
        "initial_embedding": initial_embedding,
        "service_stopped_for_vram": False,
    }

    worker_payloads: list[dict] = []
    order_a: dict[str, dict] = {}
    order_b: dict[str, dict] = {}
    full_payloads: dict[str, dict] = {}
    try:
        if initial_gpu["memory_free_mib"] < 6000 and initial_service["active_state"] == "active":
            systemctl("stop")
            service_stopped = True
            maintenance["service_stopped_for_vram"] = True
            maintenance["service_after_stop"] = query_service()
            maintenance["embedding_after_stop"] = http_get_json("http://127.0.0.1:8081/health")
            maintenance["gpu_after_stop"] = query_gpu()
            if maintenance["embedding_after_stop"].get("status") != "ok":
                raise RuntimeError("embedding service became unhealthy after inference stop")
            if maintenance["gpu_after_stop"]["memory_free_mib"] < 6000:
                raise RuntimeError("insufficient GPU memory after stopping inference service")

        for index, arm in enumerate(SMOKE_ORDER_A, 1):
            payload = run_worker(arm, f"smoke_a_{index:02d}_{arm}", workers_dir)
            order_a[arm] = payload
            worker_payloads.append(payload)
        for index, arm in enumerate(SMOKE_ORDER_B, 1):
            payload = run_worker(arm, f"smoke_b_{index:02d}_{arm}", workers_dir)
            order_b[arm] = payload
            worker_payloads.append(payload)

        smoke = compare_smoke(order_a, order_b)
        (raw_dir / "isolation_smoke.json").write_text(
            json.dumps(smoke, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if not smoke["order_invariant"]:
            raise RuntimeError("process-isolation smoke was not order invariant")

        full_payloads.update(order_a)
        for index, spec in enumerate(ADAPTER_SPECS, 1):
            arm = spec["id"]
            if arm in full_payloads:
                continue
            payload = run_worker(arm, f"full_{index:02d}_{arm}", workers_dir)
            full_payloads[arm] = payload
            worker_payloads.append(payload)
    finally:
        if service_stopped:
            systemctl("start")
            maintenance["inference_health_final"] = wait_for_health("http://127.0.0.1:8080/health", timeout_seconds=120.0)
        maintenance["final_service"] = query_service()
        maintenance["final_embedding"] = wait_for_health("http://127.0.0.1:8081/health", timeout_seconds=30.0)
        maintenance["final_gpu"] = query_gpu()
        maintenance["service_restored"] = maintenance["final_service"]["active_state"] == initial_service["active_state"]
        maintenance["exec_start_restored"] = maintenance["final_service"]["exec_start"] == initial_service["exec_start"]
        (raw_dir / "service_maintenance.json").write_text(
            json.dumps(maintenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    if set(full_payloads) != {"base", *[spec["id"] for spec in ADAPTER_SPECS]}:
        raise RuntimeError("full arm set is incomplete")
    if len(worker_payloads) != EXPECTED_WORKERS:
        raise RuntimeError(f"expected {EXPECTED_WORKERS} workers, got {len(worker_payloads)}")

    clean_receipts = [
        {
            "label": payload["host_worker_label"],
            "arm": payload["arm"],
            "pid": payload["pid"],
            "base_preexisting_peft_module_count": payload["base_preexisting_peft_module_count"],
            "loaded_adapter_peft_type": payload["loaded_adapter_peft_type"],
            "post_load_peft_module_count": payload["post_load_peft_module_count"],
            "sample_count": payload["sample_count"],
            "semantic_sha256": canonical_json_sha256(semantic_samples(payload)),
            "command": payload["command"],
        }
        for payload in worker_payloads
    ]
    (raw_dir / "clean_base_receipts.json").write_text(
        json.dumps(clean_receipts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    all_samples: list[dict] = []
    samples_path = raw_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8", newline="\n") as handle:
        for arm in ["base", *[spec["id"] for spec in ADAPTER_SPECS]]:
            for source in full_payloads[arm]["samples"]:
                sample = copy.deepcopy(source)
                sample["arm"] = arm
                all_samples.append(sample)
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    scores, independent_match = score_samples(all_samples)
    independent = {
        "independent_scorer_match": independent_match,
        "arm_scores": scores,
        "total_samples_evaluated": len(all_samples),
        "selection_rule_applied": False,
    }
    (raw_dir / "independent_evaluation.json").write_text(
        json.dumps(independent, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    minimum_math = min(value["math_total"] for value in scores.values())
    minimum_qa = min(value["qa_total"] for value in scores.values())
    clean_worker_count = sum(receipt["base_preexisting_peft_module_count"] == 0 for receipt in clean_receipts)
    smoke = json.loads((raw_dir / "isolation_smoke.json").read_text(encoding="utf-8"))

    receipt_inputs = [
        raw_dir / "artifact_hashes.json",
        raw_dir / "clean_base_receipts.json",
        raw_dir / "dataset_hashes.json",
        raw_dir / "independent_evaluation.json",
        raw_dir / "isolation_smoke.json",
        raw_dir / "samples.jsonl",
        raw_dir / "scorer_hashes.json",
        raw_dir / "service_maintenance.json",
        raw_dir / "wsl_environment.json",
        *frozen_paths,
        *artifact_paths,
    ]
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(),
        started_at_utc=started_utc,
        started_monotonic=started_mono,
        input_paths=receipt_inputs,
        packages=["pytest"],
        runtime={
            "execution_mode": "process_isolated_wsl_gpu_adapter_requalification",
            "full_arms": len(full_payloads),
            "workers": len(worker_payloads),
            "service_stopped_for_vram": service_stopped,
        },
    )
    provenance_ok, provenance_errors = provenance_complete(provenance)
    if not provenance_ok:
        raise RuntimeError("provenance incomplete: " + ", ".join(provenance_errors))

    gates = {
        "artifact_identity": {"metric": "hashed_artifacts", "operator": "eq", "threshold": 13, "actual": len(artifact_ledger), "pass": len(artifact_ledger) == 13},
        "clean_base_per_arm": {"metric": "workers_with_zero_preexisting_peft_modules", "operator": "eq", "threshold": EXPECTED_WORKERS, "actual": clean_worker_count, "pass": clean_worker_count == EXPECTED_WORKERS},
        "isolation_smoke": {"metric": "smoke_order_invariant", "operator": "eq", "threshold": True, "actual": smoke["order_invariant"], "pass": smoke["order_invariant"] is True},
        "frozen_math_panel": {"metric": "scored_math_samples_per_arm", "operator": "ge", "threshold": 32, "actual": minimum_math, "pass": minimum_math >= 32},
        "frozen_qa_panel": {"metric": "scored_qa_samples_per_arm", "operator": "ge", "threshold": 16, "actual": minimum_qa, "pass": minimum_qa >= 16},
        "base_control": {"metric": "base_control_present", "operator": "eq", "threshold": True, "actual": "base" in scores, "pass": "base" in scores},
        "independent_score": {"metric": "independent_scorer_match", "operator": "eq", "threshold": True, "actual": independent_match, "pass": independent_match is True},
    }
    receipt = {
        "schema": "local-labs-backlog-receipt-v1",
        "task_id": TASK_ID,
        "provenance": provenance,
        "provenance_complete": provenance_ok,
        "gates": gates,
        "evidence": {
            "acceptance_gates": "raw/receipt.json",
            "artifact_hashes": "raw/artifact_hashes.json",
            "clean_base_receipts": "raw/clean_base_receipts.json",
            "dataset_hashes": "raw/dataset_hashes.json",
            "independent_evaluation": "raw/independent_evaluation.json",
            "isolation_smoke": "raw/isolation_smoke.json",
            "provenance": "raw/receipt.json",
            "raw_samples": "raw/samples.jsonl",
            "receipt_fingerprint": "raw/receipt.json",
            "scorer_hashes": "raw/scorer_hashes.json",
            "wsl_environment": "raw/wsl_environment.json",
        },
    }
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    (raw_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs" / "research" / TASK_ID)
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-arm")
    parser.add_argument("--worker-out")
    parser.add_argument("--adapter-path")
    args = parser.parse_args()
    if args.worker_mode:
        if not args.worker_arm or not args.worker_out:
            parser.error("--worker-arm and --worker-out are required in worker mode")
        execute_worker(args.worker_arm, args.worker_out, args.adapter_path)
        return 0
    receipt = run_experiment(args.outdir)
    print(json.dumps(receipt["gates"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
