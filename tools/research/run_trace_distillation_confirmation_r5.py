#!/usr/bin/env python3
"""Large paired confirmation of answer-only versus full-trace SFT."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import random
import statistics
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
from tools.research import run_trace_distillation_training_r2 as r2

TASK_ID = "BACKLOG-ADAPT-TRACE-DISTILL-05"
SEEDS = list(range(20260830, 20260837))
EPOCHS = 3
TRAINING_EXAMPLES = 168
TRAINING_STEPS = TRAINING_EXAMPLES * EPOCHS
QA_IDS = [f"f{index:02d}" for index in range(1, 49)]
BOOTSTRAP_REPLICATES = 20_000
ADMISSION = ROOT / "config/research_backlog_admissions/BACKLOG-ADAPT-TRACE-DISTILL-05.json"
R4_PREREGISTRATION = ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-04/PRE_REGISTRATION.md"
TEACHER_TASK_IDS = {
    row["task_id"] for row in json.loads(r2.TEACHER_PATH.read_text(encoding="utf-8"))
}
MATH_IDS = [
    f"gsm8k/{index}" for index in range(1319)
    if f"gsm8k/{index}" not in TEACHER_TASK_IDS
][:256]
EXPECTED_HELDOUT_ID_HASH = "78a3b7ef26cdf932b79eb6f64dfb576d66770d30a5ce0fd3251536cb6e76901f"
EXPECTED_INPUT_HASHES = {
    ADMISSION: "83b748014a470a4e6f88409a38f6e87538ae7e7ee5a40b29e89f94525ead96b7",
    r2.TEACHER_PATH: "dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e",
    r2.DEFAULT_MATH_PATH: "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    r2.DEFAULT_QA_PATH: "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
    R4_PREREGISTRATION: "725e178bf4349298f70f6183971f268dc8f4dae6b8199a96a77e703a535eb9f4",
}


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_inputs() -> dict[str, dict[str, Any]]:
    ledger: dict[str, dict[str, Any]] = {}
    for path, expected in EXPECTED_INPUT_HASHES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen input mismatch: {path}: {actual} != {expected}")
        ledger[str(path.relative_to(ROOT).as_posix())] = {
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    return ledger


def build_training_manifest() -> dict[str, Any]:
    prompts = r2.load_prompt_map()
    teacher = json.loads(r2.TEACHER_PATH.read_text(encoding="utf-8"))
    pool: list[dict[str, str]] = []
    for row in teacher:
        task_id = row.get("task_id")
        completion = (row.get("completion") or "").strip()
        if task_id in r2.FROZEN_GSM8K_IDS:
            continue
        if not row.get("ok") or task_id not in prompts or not completion:
            continue
        prompt = prompts[task_id]
        gold = str(prompt["answer"]).split("####")[-1].strip().replace(",", "")
        pool.append({
            "task_id": task_id,
            "prompt": r2.PROMPT_TEMPLATE.format(prompt=prompt["prompt"]),
            "gold": gold,
            "answer_only": f"#### {gold}",
            "full_trace": completion,
        })
    pool.sort(key=lambda item: int(item["task_id"].split("/")[-1]))
    if len(pool) != TRAINING_EXAMPLES or len({item["task_id"] for item in pool}) != TRAINING_EXAMPLES:
        raise ValueError(f"expected {TRAINING_EXAMPLES} unique training examples, got {len(pool)}")
    training_ids = {item["task_id"] for item in pool}
    if len(MATH_IDS) != 256 or canonical_json_sha256(MATH_IDS) != EXPECTED_HELDOUT_ID_HASH:
        raise ValueError("teacher-disjoint held-out ID panel differs from preregistration")
    if training_ids.intersection(MATH_IDS):
        raise ValueError("training and held-out math panels overlap")

    orders: dict[str, list[list[str]]] = {}
    base_ids = [item["task_id"] for item in pool]
    for seed in SEEDS:
        epochs: list[list[str]] = []
        for epoch in range(EPOCHS):
            order = list(base_ids)
            random.Random(seed * 100 + epoch).shuffle(order)
            epochs.append(order)
        orders[str(seed)] = epochs
    return {
        "schema": "local-labs-trace-training-pairs-v2",
        "pool": pool,
        "orders": orders,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "training_steps_per_arm": TRAINING_STEPS,
        "heldout_math_ids": MATH_IDS,
        "protected_qa_ids": QA_IDS,
    }


def _training_rows(manifest: dict[str, Any], seed: int) -> list[dict[str, str]]:
    by_id = {row["task_id"]: row for row in manifest["pool"]}
    rows = [by_id[task_id] for epoch in manifest["orders"][str(seed)] for task_id in epoch]
    if len(rows) != TRAINING_STEPS:
        raise ValueError(f"expected {TRAINING_STEPS} training rows, got {len(rows)}")
    return rows


def worker(output_path: str, checkpoint_path: str, manifest_path: str, arm: str, seed: int) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(r2.BASE_MODEL_WSL, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        r2.BASE_MODEL_WSL, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    preexisting = r2.count_peft_modules(model)
    if preexisting != 0:
        raise RuntimeError(f"fresh base contained {preexisting} PEFT modules")
    model.config.use_cache = False
    trained = get_peft_model(model, LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    ))
    trained.train()
    optimizer = torch.optim.AdamW(trained.parameters(), lr=1e-4, weight_decay=0.01)
    manifest = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
    rows = _training_rows(manifest, seed)
    target_key = arm
    trace: list[dict[str, Any]] = []
    started = time.monotonic()
    for step, row in enumerate(rows, 1):
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"][-256:]
        completion_ids = tokenizer(row[target_key], add_special_tokens=False)["input_ids"]
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        completion_ids = completion_ids[-max(1, 512 - len(prompt_ids) - len(eos)):]
        input_ids = prompt_ids + completion_ids + eos
        completion_n = len(completion_ids) + len(eos)
        labels = [-100] * (len(input_ids) - completion_n) + input_ids[-completion_n:]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda:0")
        label_tensor = torch.tensor([labels], dtype=torch.long, device="cuda:0")
        optimizer.zero_grad()
        result = trained(
            input_ids=input_tensor,
            attention_mask=torch.ones_like(input_tensor),
            labels=label_tensor,
        )
        loss = result.loss
        value = float(loss.item())
        if torch.isnan(loss) or torch.isinf(loss) or value > 100:
            raise RuntimeError(f"loss diverged at step {step}: {value}")
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(trained.parameters(), 1.0))
        optimizer.step()
        trace.append({
            "step": step,
            "epoch": (step - 1) // TRAINING_EXAMPLES + 1,
            "task_id": row["task_id"],
            "loss": round(value, 6),
            "grad_norm": round(gradient, 6),
            "elapsed_s": round(time.monotonic() - started, 4),
        })
        if step % 42 == 0:
            print(f"[WORKER] {arm} seed={seed} step={step}/{TRAINING_STEPS} loss={value:.4f}", flush=True)

    checkpoint = pathlib.Path(checkpoint_path)
    checkpoint.mkdir(parents=True, exist_ok=True)
    trained.save_pretrained(checkpoint)
    trained.eval()

    def generate(prompt: str, maximum: int) -> tuple[str, int, bool, float]:
        tokens = tokenizer(prompt, return_tensors="pt").to(trained.device)
        prompt_n = tokens["input_ids"].shape[1]
        before = time.monotonic()
        with torch.no_grad():
            generated = trained.generate(
                **tokens,
                max_new_tokens=maximum,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        output_ids = generated[0][prompt_n:]
        return (
            tokenizer.decode(output_ids, skip_special_tokens=True).strip(),
            len(output_ids),
            bool(tokenizer.eos_token_id is not None and generated[0][-1].item() == tokenizer.eos_token_id),
            round(time.monotonic() - before, 4),
        )

    math_samples: list[dict[str, Any]] = []
    for task in r2.load_math_panel(r2.DEFAULT_MATH_PATH, MATH_IDS):
        text, token_n, eos, elapsed = generate(task["prompt"], 192)
        extracted = r2.extract_gsm8k_pred(text)
        math_samples.append({
            "panel": "math",
            "task_id": task["task_id"],
            "prompt": task["prompt"],
            "gold": task["gold"],
            "output_text": text,
            "extracted": extracted,
            "correct": r2.is_gsm8k_correct(extracted, task["gold"]),
            "new_tokens": token_n,
            "natural_eos": eos,
            "elapsed_s": elapsed,
        })
    qa_samples: list[dict[str, Any]] = []
    for task in r2.load_qa_panel(r2.DEFAULT_QA_PATH, QA_IDS):
        text, token_n, eos, elapsed = generate(task["prompt"], 128)
        correct, detail = r2.grade_qa(task, text)
        qa_samples.append({
            "panel": "qa",
            "task_id": task["id"],
            "prompt": task["prompt"],
            "output_text": text,
            "correct": correct,
            "grade_detail": detail,
            "new_tokens": token_n,
            "natural_eos": eos,
            "elapsed_s": elapsed,
        })
    payload = {
        "arm": arm,
        "seed": seed,
        "pid": __import__("os").getpid(),
        "base_preexisting_peft_module_count": preexisting,
        "post_injection_peft_module_count": r2.count_peft_modules(trained),
        "training_pair_count": TRAINING_EXAMPLES,
        "training_step_count": len(rows),
        "training_task_ids": [row["task_id"] for row in rows],
        "training_target_sha256": canonical_json_sha256([row[target_key] for row in rows]),
        "training_trace": trace,
        "checkpoint": str(checkpoint),
        "math_samples": math_samples,
        "qa_samples": qa_samples,
        "math_correct": sum(sample["correct"] for sample in math_samples),
        "math_total": len(math_samples),
        "qa_correct": sum(sample["correct"] for sample in qa_samples),
        "qa_total": len(qa_samples),
    }
    pathlib.Path(output_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def worker_command(
    script: pathlib.Path,
    manifest: pathlib.Path,
    output: pathlib.Path,
    checkpoint: pathlib.Path,
    arm: str,
    seed: int,
) -> list[str]:
    return [
        "wsl", "-d", "Ubuntu-24.04", "--", r2.WSL_PYTHON,
        r2.windows_path_to_wsl(script),
        "--worker-mode",
        "--worker-out", r2.windows_path_to_wsl(output),
        "--checkpoint-out", r2.windows_path_to_wsl(checkpoint),
        "--manifest", r2.windows_path_to_wsl(manifest),
        "--arm", arm,
        "--seed", str(seed),
    ]


def hierarchical_bootstrap(seed_differences: list[list[int]], replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    if len(seed_differences) != len(SEEDS) or any(len(row) != len(MATH_IDS) for row in seed_differences):
        raise ValueError("bootstrap input dimensions do not match the frozen design")
    rng = random.Random(2026082604)
    estimates: list[float] = []
    seed_n = len(seed_differences)
    prompt_n = len(seed_differences[0])
    for _ in range(replicates):
        total = 0
        count = 0
        for _seed_draw in range(seed_n):
            row = seed_differences[rng.randrange(seed_n)]
            for _prompt_draw in range(prompt_n):
                total += row[rng.randrange(prompt_n)]
                count += 1
        estimates.append(total / count)
    estimates.sort()
    lower = estimates[int(0.025 * replicates)]
    upper = estimates[min(replicates - 1, int(0.975 * replicates))]
    return {
        "replicates": replicates,
        "seed": 2026082604,
        "lower_95": round(lower, 8),
        "upper_95": round(upper, 8),
    }


def exact_sign_flip_pvalue(seed_deltas: list[float]) -> float:
    observed = statistics.mean(seed_deltas)
    magnitudes = [abs(value) for value in seed_deltas]
    outcomes = [statistics.mean(sign * magnitude for sign, magnitude in zip(signs, magnitudes))
                for signs in itertools.product((-1, 1), repeat=len(magnitudes))]
    return sum(value >= observed - 1e-15 for value in outcomes) / len(outcomes)


def binomial_upper_tail(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    return sum(math.comb(trials, k) for k in range(successes, trials + 1)) / (2 ** trials)


def score_outputs(payloads: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    math_tasks = {task["task_id"]: task for task in r2.load_math_panel(r2.DEFAULT_MATH_PATH, MATH_IDS)}
    qa_tasks = {task["id"]: task for task in r2.load_qa_panel(r2.DEFAULT_QA_PATH, QA_IDS)}
    seed_rows: list[dict[str, Any]] = []
    seed_differences: list[list[int]] = []
    independent_match = True
    trace_only = 0
    answer_only = 0
    for seed in SEEDS:
        arms = {payload["arm"]: payload for payload in payloads if payload["seed"] == seed}
        recomputed: dict[str, dict[str, Any]] = {}
        math_correct_by_arm: dict[str, dict[str, int]] = {}
        for arm in ("answer_only", "full_trace"):
            payload = arms[arm]
            math_values: dict[str, int] = {}
            for sample in payload["math_samples"]:
                extracted = r2.extract_gsm8k_pred(sample["output_text"])
                correct = int(r2.is_gsm8k_correct(extracted, math_tasks[sample["task_id"]]["gold"]))
                math_values[sample["task_id"]] = correct
                independent_match &= bool(correct) == bool(sample["correct"])
            qa_correct = 0
            for sample in payload["qa_samples"]:
                correct, _ = r2.grade_qa(qa_tasks[sample["task_id"]], sample["output_text"])
                qa_correct += int(correct)
                independent_match &= bool(correct) == bool(sample["correct"])
            math_correct_by_arm[arm] = math_values
            recomputed[arm] = {"math_correct": sum(math_values.values()), "qa_correct": qa_correct}
        differences: list[int] = []
        for task_id in MATH_IDS:
            answer_value = math_correct_by_arm["answer_only"][task_id]
            trace_value = math_correct_by_arm["full_trace"][task_id]
            differences.append(trace_value - answer_value)
            trace_only += int(trace_value == 1 and answer_value == 0)
            answer_only += int(answer_value == 1 and trace_value == 0)
        seed_differences.append(differences)
        answer_math = recomputed["answer_only"]["math_correct"] / len(MATH_IDS)
        trace_math = recomputed["full_trace"]["math_correct"] / len(MATH_IDS)
        answer_qa = recomputed["answer_only"]["qa_correct"] / len(QA_IDS)
        trace_qa = recomputed["full_trace"]["qa_correct"] / len(QA_IDS)
        seed_rows.append({
            "seed": seed,
            "answer_math_correct": recomputed["answer_only"]["math_correct"],
            "trace_math_correct": recomputed["full_trace"]["math_correct"],
            "math_gain": round(trace_math - answer_math, 8),
            "answer_qa_correct": recomputed["answer_only"]["qa_correct"],
            "trace_qa_correct": recomputed["full_trace"]["qa_correct"],
            "qa_regression": round(max(0.0, answer_qa - trace_qa), 8),
        })
    seed_deltas = [row["math_gain"] for row in seed_rows]
    bootstrap = hierarchical_bootstrap(seed_differences)
    discordant = trace_only + answer_only
    scores = {
        "seeds": seed_rows,
        "mean_trace_math_gain_over_answer_only": round(statistics.mean(seed_deltas), 8),
        "hierarchical_bootstrap": bootstrap,
        "seeds_with_positive_trace_math_gain": sum(value > 0 for value in seed_deltas),
        "mean_protected_qa_regression_vs_answer_only": round(
            statistics.mean(row["qa_regression"] for row in seed_rows), 8
        ),
        "exact_one_sided_seed_sign_flip_p": round(exact_sign_flip_pvalue(seed_deltas), 8),
        "pooled_discordant_pairs": {
            "trace_only_correct": trace_only,
            "answer_only_correct": answer_only,
            "one_sided_exact_mcnemar_p": round(binomial_upper_tail(trace_only, discordant), 12),
        },
    }
    return scores, independent_match


def run_experiment(outdir: pathlib.Path) -> dict[str, Any]:
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError(f"raw output directory is not empty: {raw}")
    workers = raw / "workers"
    checkpoints = raw / "checkpoints"
    workers.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    input_ledger = verify_inputs()
    base_ledger = r2.verify_base_model()
    manifest = build_training_manifest()
    manifest_path = raw / "training_pairs.json"
    write_json(manifest_path, manifest)
    write_json(raw / "dataset_hashes.json", {
        "inputs": input_ledger,
        "eligible_training_examples": len(manifest["pool"]),
        "heldout_math_count": len(MATH_IDS),
        "heldout_qa_count": len(QA_IDS),
        "training_heldout_overlap": sorted({row["task_id"] for row in manifest["pool"]}.intersection(MATH_IDS)),
    })
    write_json(raw / "model_hash.json", base_ledger)

    initial_service = r2.query_service()
    initial_gpu = r2.query_gpu()
    initial_embedding = r2.http_get_json("http://127.0.0.1:8081/health")
    maintenance: dict[str, Any] = {
        "initial_service": initial_service,
        "initial_gpu": initial_gpu,
        "initial_embedding": initial_embedding,
        "service_stopped_for_vram": False,
    }
    service_stopped = False
    payloads: list[dict[str, Any]] = []
    script = pathlib.Path(__file__).resolve()
    try:
        if initial_gpu["memory_free_mib"] < 6000 and initial_service["active_state"] == "active":
            r2.systemctl("stop")
            service_stopped = True
            maintenance["service_stopped_for_vram"] = True
            maintenance["service_after_stop"] = r2.query_service()
            maintenance["embedding_after_stop"] = r2.http_get_json("http://127.0.0.1:8081/health")
            if maintenance["embedding_after_stop"].get("status") != "ok":
                raise RuntimeError("embedding service became unhealthy")
        for seed_index, seed in enumerate(SEEDS):
            arm_order = ("answer_only", "full_trace") if seed_index % 2 == 0 else ("full_trace", "answer_only")
            for arm in arm_order:
                label = f"seed_{seed}_{arm}"
                output = workers / f"{label}.json"
                checkpoint = checkpoints / label
                command = worker_command(script, manifest_path, output, checkpoint, arm, seed)
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=14_400,
                    check=False,
                )
                (workers / f"{label}.stdout.log").write_text(completed.stdout, encoding="utf-8")
                (workers / f"{label}.stderr.log").write_text(completed.stderr, encoding="utf-8")
                if completed.returncode != 0:
                    raise RuntimeError(f"worker {label} failed: {completed.stderr[-4000:]}")
                payload = json.loads(output.read_text(encoding="utf-8"))
                payload["host_command"] = command
                payloads.append(payload)
                print(
                    f"[HOST] {label} complete: math={payload['math_correct']}/{len(MATH_IDS)} "
                    f"qa={payload['qa_correct']}/{len(QA_IDS)}",
                    flush=True,
                )
    finally:
        if service_stopped:
            r2.systemctl("start")
            maintenance["inference_health_final"] = r2.wait_for_health(
                "http://127.0.0.1:8080/health", timeout_seconds=180
            )
        maintenance["final_service"] = r2.query_service()
        maintenance["final_embedding"] = r2.wait_for_health(
            "http://127.0.0.1:8081/health", timeout_seconds=30
        )
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

    if len(payloads) != len(SEEDS) * 2:
        raise ValueError(f"expected {len(SEEDS) * 2} workers, got {len(payloads)}")
    treatment_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        arms = {payload["arm"]: payload for payload in payloads if payload["seed"] == seed}
        treatment_rows.append({
            "seed": seed,
            "same_task_order": arms["answer_only"]["training_task_ids"] == arms["full_trace"]["training_task_ids"],
            "target_texts_distinct": sum(row["answer_only"] != row["full_trace"] for row in manifest["pool"]),
            "answer_target_sha256": arms["answer_only"]["training_target_sha256"],
            "trace_target_sha256": arms["full_trace"]["training_target_sha256"],
        })
    treatment_verified = all(
        row["same_task_order"]
        and row["target_texts_distinct"] == TRAINING_EXAMPLES
        and row["answer_target_sha256"] != row["trace_target_sha256"]
        for row in treatment_rows
    )
    write_json(raw / "treatment_materiality.json", {"verified": treatment_verified, "seeds": treatment_rows})

    checkpoint_ledger: dict[str, Any] = {}
    for payload in payloads:
        label = f"seed_{payload['seed']}_{payload['arm']}"
        checkpoint = checkpoints / label
        checkpoint_ledger[label] = {
            "config_sha256": sha256_file(checkpoint / "adapter_config.json"),
            "weights_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
        }
    write_json(raw / "checkpoint_hashes.json", checkpoint_ledger)
    write_json(raw / "training_trace.json", [
        {"arm": payload["arm"], "seed": payload["seed"], "trace": payload["training_trace"]}
        for payload in payloads
    ])
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for payload in payloads:
            for sample in [*payload["math_samples"], *payload["qa_samples"]]:
                stream.write(json.dumps({
                    "arm": payload["arm"], "seed": payload["seed"], **sample
                }, ensure_ascii=False) + "\n")
    write_json(raw / "student_samples.json", [{
        "arm": payload["arm"],
        "seed": payload["seed"],
        "math_samples": payload["math_samples"],
        "qa_samples": payload["qa_samples"],
    } for payload in payloads])
    write_json(raw / "teacher_samples.json", {
        "source": str(r2.TEACHER_PATH.relative_to(ROOT).as_posix()),
        "training_pool": [{"task_id": row["task_id"], "full_trace": row["full_trace"]} for row in manifest["pool"]],
    })
    scores, independent_match = score_outputs(payloads)
    write_json(raw / "actual_scores.json", scores)
    write_json(raw / "independent_evaluation.json", {
        "independent_scorer_match": independent_match,
        "scores": scores,
    })

    provenance_inputs = [
        raw / "actual_scores.json",
        raw / "checkpoint_hashes.json",
        raw / "dataset_hashes.json",
        raw / "independent_evaluation.json",
        raw / "model_hash.json",
        raw / "samples.jsonl",
        raw / "service_maintenance.json",
        raw / "student_samples.json",
        raw / "teacher_samples.json",
        raw / "training_pairs.json",
        raw / "training_trace.json",
        raw / "treatment_materiality.json",
        *EXPECTED_INPUT_HASHES.keys(),
    ]
    provenance = build_provenance(
        script_path=script,
        started_at_utc=started_utc,
        started_monotonic=started_mono,
        input_paths=provenance_inputs,
        packages=["pytest"],
        runtime={
            "execution_mode": "large_paired_trace_distillation_confirmation",
            "workers": len(payloads),
            "seeds": SEEDS,
            "training_steps_per_worker": TRAINING_STEPS,
            "heldout_math_per_worker": len(MATH_IDS),
            "heldout_qa_per_worker": len(QA_IDS),
        },
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise ValueError(f"incomplete provenance: {errors}")
    training_step_minimum = min(payload["training_step_count"] for payload in payloads)
    math_sample_minimum = min(payload["math_total"] for payload in payloads)
    bootstrap_lower = scores["hierarchical_bootstrap"]["lower_95"]
    gates = {
        "treatment_materiality": {"metric": "matched_distinct_training_targets_verified", "operator": "eq", "threshold": True, "actual": treatment_verified, "pass": treatment_verified is True},
        "training_budget": {"metric": "training_steps_per_arm_per_seed", "operator": "eq", "threshold": TRAINING_STEPS, "actual": training_step_minimum, "pass": all(payload["training_step_count"] == TRAINING_STEPS for payload in payloads)},
        "seed_coverage": {"metric": "completed_paired_seeds", "operator": "eq", "threshold": len(SEEDS), "actual": len(scores["seeds"]), "pass": len(scores["seeds"]) == len(SEEDS)},
        "heldout_coverage": {"metric": "heldout_math_samples_per_arm_per_seed", "operator": "eq", "threshold": len(MATH_IDS), "actual": math_sample_minimum, "pass": all(payload["math_total"] == len(MATH_IDS) for payload in payloads)},
        "confirmed_gain": {"metric": "hierarchical_bootstrap_95ci_lower_trace_math_gain", "operator": "gt", "threshold": 0.0, "actual": bootstrap_lower, "pass": bootstrap_lower > 0.0},
        "directional_repeatability": {"metric": "seeds_with_positive_trace_math_gain", "operator": "ge", "threshold": 5, "actual": scores["seeds_with_positive_trace_math_gain"], "pass": scores["seeds_with_positive_trace_math_gain"] >= 5},
        "protected_retention": {"metric": "mean_protected_qa_regression_vs_answer_only", "operator": "le", "threshold": 0.05, "actual": scores["mean_protected_qa_regression_vs_answer_only"], "pass": scores["mean_protected_qa_regression_vs_answer_only"] <= 0.05},
        "service_recovery": {"metric": "service_and_embedding_restored", "operator": "eq", "threshold": True, "actual": maintenance["service_and_embedding_restored"], "pass": maintenance["service_and_embedding_restored"] is True},
    }
    receipt = {
        "schema": "local-labs-backlog-receipt-v1",
        "task_id": TASK_ID,
        "provenance": provenance,
        "provenance_complete": complete,
        "gates": gates,
        "evidence": {
            "acceptance_gates": "raw/receipt.json",
            "actual_scores": "raw/actual_scores.json",
            "checkpoint_hashes": "raw/checkpoint_hashes.json",
            "dataset_hashes": "raw/dataset_hashes.json",
            "independent_evaluation": "raw/independent_evaluation.json",
            "model_hash": "raw/model_hash.json",
            "provenance": "raw/receipt.json",
            "raw_samples": "raw/samples.jsonl",
            "receipt_fingerprint": "raw/receipt.json",
            "service_maintenance": "raw/service_maintenance.json",
            "student_samples": "raw/student_samples.json",
            "teacher_samples": "raw/teacher_samples.json",
            "training_pairs": "raw/training_pairs.json",
            "training_trace": "raw/training_trace.json",
            "treatment_materiality": "raw/treatment_materiality.json",
        },
    }
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    failed = [gate_id for gate_id, gate in gates.items() if not gate["pass"]]
    claim = "TRACE_DISTILLATION_CONFIRMED_R5" if not failed else "TRACE_DISTILLATION_NOT_CONFIRMED_R5"
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n"
        f"`{claim}` pending independent review.\n\n"
        f"Seven paired seeds completed with {TRAINING_STEPS} training steps per arm and "
        f"{len(MATH_IDS)} held-out math plus {len(QA_IDS)} protected-QA samples per worker. "
        f"Mean trace math gain: `{scores['mean_trace_math_gain_over_answer_only']:.6f}`; "
        f"hierarchical bootstrap 95% interval: "
        f"`[{scores['hierarchical_bootstrap']['lower_95']:.6f}, {scores['hierarchical_bootstrap']['upper_95']:.6f}]`; "
        f"positive seeds: `{scores['seeds_with_positive_trace_math_gain']}/7`; "
        f"failed gates: `{', '.join(failed) if failed else 'none'}`.\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-out")
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--manifest")
    parser.add_argument("--arm", choices=["answer_only", "full_trace"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.worker_mode:
        required = [args.worker_out, args.checkpoint_out, args.manifest, args.arm, args.seed]
        if any(value is None for value in required):
            parser.error("worker mode requires output, checkpoint, manifest, arm and seed")
        worker(args.worker_out, args.checkpoint_out, args.manifest, args.arm, args.seed)
        return 0
    receipt = run_experiment(args.outdir.resolve())
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
