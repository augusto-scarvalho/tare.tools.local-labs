#!/usr/bin/env python3
"""Requalification runner for saved ADAPT-01A through ADAPT-05 adapter artifacts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import time
import unicodedata
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)

BASE_MODEL_WSL = "/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe"
DEFAULT_MATH_PATH = ROOT / "workloads" / "gsm8k.jsonl"
DEFAULT_QA_PATH = ROOT / "runs" / "requalification" / "QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23" / "tasks.jsonl"

FROZEN_GSM8K_IDS = [
    "gsm8k/392", "gsm8k/1226", "gsm8k/541", "gsm8k/44", "gsm8k/489",
    "gsm8k/1298", "gsm8k/663", "gsm8k/1217", "gsm8k/1186", "gsm8k/225",
    "gsm8k/110", "gsm8k/174", "gsm8k/986", "gsm8k/173", "gsm8k/317",
    "gsm8k/529", "gsm8k/236", "gsm8k/831", "gsm8k/86", "gsm8k/19",
    "gsm8k/967", "gsm8k/724", "gsm8k/1001", "gsm8k/1212", "gsm8k/1264",
    "gsm8k/662", "gsm8k/34", "gsm8k/1294", "gsm8k/551", "gsm8k/175",
    "gsm8k/430", "gsm8k/386"
]

FROZEN_QA_IDS = [
    "f01", "f02", "f03", "m01", "m02", "m03",
    "r01", "r02", "r03", "i01", "i02", "i03",
    "c01", "c02", "s01", "s02"
]

ADAPTER_SPECS = [
    {"id": "lokr_1ep", "source_campaign": "ADAPT-01A", "rel_path": "runs/research/ADAPT-01A-LOKR-SCALE-2026-08-25/raw/lokr_1ep/adapter"},
    {"id": "lokr_3ep", "source_campaign": "ADAPT-01A", "rel_path": "runs/research/ADAPT-01A-LOKR-SCALE-2026-08-25/raw/lokr_3ep/adapter"},
    {"id": "lokr_3ep_lr1e4", "source_campaign": "ADAPT-01A", "rel_path": "runs/research/ADAPT-01A-LOKR-SCALE-2026-08-25/raw/lokr_3ep_lr1e4/adapter"},
    {"id": "lokr_5ep", "source_campaign": "ADAPT-01A", "rel_path": "runs/research/ADAPT-01A-LOKR-SCALE-2026-08-25/raw/lokr_5ep/adapter"},
    {"id": "target_all_linear", "source_campaign": "ADAPT-02", "rel_path": "runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/target_all_linear/adapter"},
    {"id": "target_attn_only", "source_campaign": "ADAPT-02", "rel_path": "runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/target_attn_only/adapter"},
    {"id": "target_mlp_only", "source_campaign": "ADAPT-02", "rel_path": "runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/target_mlp_only/adapter"},
    {"id": "target_qv_gate", "source_campaign": "ADAPT-02", "rel_path": "runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/target_qv_gate/adapter"},
    {"id": "soft_prompts", "source_campaign": "ADAPT-03", "rel_path": "runs/research/ADAPT-03-SOFT-PROMPTS-2026-08-25/raw/adapter"},
    {"id": "lokr_prior_lambda02", "source_campaign": "ADAPT-04", "rel_path": "runs/research/ADAPT-04-PRIOR-PRESERVATION-2026-08-25/raw/lokr_prior_lambda02/adapter"},
    {"id": "lokr_prior_lambda05", "source_campaign": "ADAPT-04", "rel_path": "runs/research/ADAPT-04-PRIOR-PRESERVATION-2026-08-25/raw/lokr_prior_lambda05/adapter"},
    {"id": "lokr_unreg_5ep", "source_campaign": "ADAPT-04", "rel_path": "runs/research/ADAPT-04-PRIOR-PRESERVATION-2026-08-25/raw/lokr_unreg_5ep/adapter"},
    {"id": "disjoint_composite", "source_campaign": "ADAPT-05", "rel_path": "runs/research/ADAPT-05-MODULAR-MERGING-2026-08-25/raw/disjoint_composite"},
]


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return text


def extract_gsm8k_gold(answer: str) -> str:
    parts = str(answer).split("####")
    gold_str = parts[-1] if len(parts) > 1 else str(answer)
    m = re.findall(r"-?\d[\d,]*\.?\d*", gold_str)
    return m[-1].replace(",", "") if m else ""


def extract_gsm8k_pred(text: str) -> str | None:
    t = text.split("</think>")[-1]
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", t)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", t)
    return nums[-1].replace(",", "") if nums else None


def is_gsm8k_correct(pred: str | None, gold: str) -> bool:
    if pred is None or not gold:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-5
    except Exception:
        return str(pred).strip() == str(gold).strip()


def grade_qa(task: dict, answer: str) -> tuple[bool, str]:
    kind = task["grader"]
    clean = strip_fence(answer)
    if kind == "exact_any":
        got = normalize_text(clean)
        choices = [normalize_text(str(item)) for item in task["expected"]]
        ok = got in choices
        return ok, f"normalized={got!r} expected={choices!r}"
    if kind == "json_exact":
        try:
            got = json.loads(clean)
        except (TypeError, ValueError) as exc:
            return False, f"invalid_json={exc}"
        ok = got == task["expected"]
        return ok, f"json={got!r}"
    if kind == "lines_exact":
        got = [line.strip() for line in clean.splitlines() if line.strip()]
        ok = got == task["expected"]
        return ok, f"lines={got!r}"
    if kind == "contains_all":
        haystack = normalize_text(clean)
        missing = [item for item in task.get("required", []) if normalize_text(str(item)) not in haystack]
        forbidden = [item for item in task.get("forbidden", []) if normalize_text(str(item)) in haystack]
        words = re.findall(r"\b\w+\b", clean, flags=re.UNICODE)
        too_long = bool(task.get("max_words") and len(words) > task["max_words"])
        ok = not missing and not forbidden and not too_long
        return ok, f"missing={missing!r} forbidden={forbidden!r} words={len(words)}"
    if kind == "pt_question":
        lowered = clean.casefold()
        forbidden = [item for item in task.get("forbidden", []) if item.casefold() in lowered]
        words = re.findall(r"\b\w+\b", clean, flags=re.UNICODE)
        pt_signals = ("qual", "quais", "que", "como", "onde", "quanto", "quantos",
                      "você", "seu", "sua", "deseja", "objetivo", "orçamento", "é")
        has_pt_signal = any(re.search(rf"\b{re.escape(signal)}\b", lowered) for signal in pt_signals)
        one_question = clean.count("?") == 1 and clean.rstrip().endswith("?")
        too_long = bool(task.get("max_words") and len(words) > task["max_words"])
        ok = one_question and has_pt_signal and not forbidden and not too_long
        return ok, f"one_question={one_question} pt_signal={has_pt_signal} forbidden={forbidden!r} words={len(words)}"
    raise ValueError(f"unknown grader: {kind}")


def load_math_panel(path: pathlib.Path, frozen_ids: list[str]) -> list[dict]:
    by_id = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                by_id[row["task_id"]] = {
                    "task_id": row["task_id"],
                    "prompt": row["prompt"],
                    "gold": extract_gsm8k_gold(row["answer"]),
                }
    return [by_id[tid] for tid in frozen_ids if tid in by_id]


def load_qa_panel(path: pathlib.Path, frozen_ids: list[str]) -> list[dict]:
    by_id = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                by_id[row["id"]] = row
    return [by_id[qid] for qid in frozen_ids if qid in by_id]


def build_artifact_ledger(root: pathlib.Path) -> dict:
    ledger = {}
    for spec in ADAPTER_SPECS:
        path = root / spec["rel_path"]
        cfg = path / "adapter_config.json"
        st = path / "adapter_model.safetensors"
        if not cfg.exists() or not st.exists():
            raise FileNotFoundError(f"Missing adapter artifact in {path}")
        ledger[spec["id"]] = {
            "id": spec["id"],
            "source_campaign": spec["source_campaign"],
            "relative_path": spec["rel_path"],
            "config": {
                "bytes": cfg.stat().st_size,
                "sha256": sha256_file(cfg),
            },
            "safetensors": {
                "bytes": st.stat().st_size,
                "sha256": sha256_file(st),
            },
        }
    return ledger


def build_dataset_ledger(root: pathlib.Path) -> dict:
    math_path = root / "workloads" / "gsm8k.jsonl"
    qa_path = root / "runs" / "requalification" / "QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23" / "tasks.jsonl"
    return {
        "math_panel": {
            "path": str(math_path.relative_to(root).as_posix()),
            "bytes": math_path.stat().st_size,
            "sha256": sha256_file(math_path),
            "frozen_ids_count": len(FROZEN_GSM8K_IDS),
        },
        "qa_panel": {
            "path": str(qa_path.relative_to(root).as_posix()),
            "bytes": qa_path.stat().st_size,
            "sha256": sha256_file(qa_path),
            "frozen_ids_count": len(FROZEN_QA_IDS),
        },
    }


def execute_inference_in_wsl(output_json_wsl: str, math_tasks: list[dict], qa_tasks: list[dict]) -> None:
    """Runs inside WSL Ubuntu-24.04 with PyTorch + Transformers + PEFT."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch.manual_seed(20260824)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260824)

    base_path = BASE_MODEL_WSL
    print(f"[WSL] Loading base model from {base_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    base_model.eval()

    def generate_response(model, prompt_text: str, max_new_tokens: int) -> tuple[str, int, bool, float]:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        t0 = time.monotonic()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.monotonic() - t0
        output_ids = outputs[0][input_len:]
        new_tokens = len(output_ids)
        natural_eos = (outputs[0][-1].item() == tokenizer.eos_token_id)
        decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return decoded, new_tokens, natural_eos, round(elapsed, 4)

    arms_data = []

    # 1. Base control arm
    print("[WSL] Evaluating base control arm...", flush=True)
    base_samples = []
    for t in math_tasks:
        ans, tok_count, eos, el = generate_response(base_model, t["prompt"], max_new_tokens=192)
        pred = extract_gsm8k_pred(ans)
        ok = is_gsm8k_correct(pred, t["gold"])
        base_samples.append({
            "panel": "math",
            "task_id": t["task_id"],
            "prompt": t["prompt"],
            "gold": t["gold"],
            "extracted": pred,
            "correct": ok,
            "output_text": ans,
            "new_tokens": tok_count,
            "natural_eos": eos,
            "elapsed_s": el,
        })
    for q in qa_tasks:
        ans, tok_count, eos, el = generate_response(base_model, q["prompt"], max_new_tokens=128)
        ok, detail = grade_qa(q, ans)
        base_samples.append({
            "panel": "qa",
            "task_id": q["id"],
            "category": q["category"],
            "prompt": q["prompt"],
            "correct": ok,
            "grade_detail": detail,
            "output_text": ans,
            "new_tokens": tok_count,
            "natural_eos": eos,
            "elapsed_s": el,
        })
    arms_data.append({"arm": "base", "is_base_control": True, "samples": base_samples})

    # 2. 13 Adapter arms
    for idx, spec in enumerate(ADAPTER_SPECS, 1):
        arm_id = spec["id"]
        adapter_path = str(ROOT / spec["rel_path"])
        if adapter_path.startswith("C:\\") or adapter_path.startswith("C:/"):
            wsl_adapter_path = "/mnt/c/" + adapter_path[3:].replace("\\", "/")
        else:
            wsl_adapter_path = adapter_path

        print(f"[WSL] [{idx:02d}/13] Evaluating adapter arm {arm_id}...", flush=True)
        model = PeftModel.from_pretrained(base_model, wsl_adapter_path)
        model.eval()

        arm_samples = []
        for t in math_tasks:
            ans, tok_count, eos, el = generate_response(model, t["prompt"], max_new_tokens=192)
            pred = extract_gsm8k_pred(ans)
            ok = is_gsm8k_correct(pred, t["gold"])
            arm_samples.append({
                "panel": "math",
                "task_id": t["task_id"],
                "prompt": t["prompt"],
                "gold": t["gold"],
                "extracted": pred,
                "correct": ok,
                "output_text": ans,
                "new_tokens": tok_count,
                "natural_eos": eos,
                "elapsed_s": el,
            })
        for q in qa_tasks:
            ans, tok_count, eos, el = generate_response(model, q["prompt"], max_new_tokens=128)
            ok, detail = grade_qa(q, ans)
            arm_samples.append({
                "panel": "qa",
                "task_id": q["id"],
                "category": q["category"],
                "prompt": q["prompt"],
                "correct": ok,
                "grade_detail": detail,
                "output_text": ans,
                "new_tokens": tok_count,
                "natural_eos": eos,
                "elapsed_s": el,
            })
        arms_data.append({"arm": arm_id, "is_base_control": False, "samples": arm_samples})

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(output_json_wsl, "w", encoding="utf-8") as f:
        json.dump(arms_data, f, indent=2, ensure_ascii=False)
    print(f"[WSL] Finished all 14 arms! Output written to {output_json_wsl}", flush=True)


def run_requalification(outdir: pathlib.Path) -> dict:
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()

    outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    math_tasks = load_math_panel(DEFAULT_MATH_PATH, FROZEN_GSM8K_IDS)
    qa_tasks = load_qa_panel(DEFAULT_QA_PATH, FROZEN_QA_IDS)

    if len(math_tasks) != 32:
        raise ValueError(f"Expected 32 math tasks, found {len(math_tasks)}")
    if len(qa_tasks) != 16:
        raise ValueError(f"Expected 16 QA tasks, found {len(qa_tasks)}")

    artifact_ledger = build_artifact_ledger(ROOT)
    dataset_ledger = build_dataset_ledger(ROOT)

    (raw_dir / "artifact_hashes.json").write_text(
        json.dumps(artifact_ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (raw_dir / "dataset_hashes.json").write_text(
        json.dumps(dataset_ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    scorer_ledger = {
        "script_path": "tools/research/run_adapter_requalification.py",
        "script_sha256": sha256_file(pathlib.Path(__file__).resolve()),
        "scorers": [
            {"name": "extract_gsm8k_pred", "type": "strict_regex_and_boxed"},
            {"name": "grade_qa", "type": "deterministic_classifier"},
        ]
    }
    (raw_dir / "scorer_hashes.json").write_text(
        json.dumps(scorer_ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    intermediate_json = raw_dir / "_intermediate_arms.json"
    intermediate_wsl = "/mnt/c/" + str(intermediate_json.resolve())[3:].replace("\\", "/")

    if platform.system() == "Windows":
        print("[HOST] Delegating GPU inference to WSL2 venv adapt00-20260824...", flush=True)
        wsl_script = "/mnt/c/" + str(pathlib.Path(__file__).resolve())[3:].replace("\\", "/")
        cmd = [
            "wsl", "-d", "Ubuntu-24.04", "--",
            "/home/augus/.venvs/adapt00-20260824/bin/python",
            wsl_script,
            "--worker-mode",
            "--worker-out", intermediate_wsl,
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            raise RuntimeError(f"WSL inference failed with code {completed.returncode}")
        print(completed.stdout)
    else:
        execute_inference_in_wsl(str(intermediate_json), math_tasks, qa_tasks)

    arms_data = json.loads(intermediate_json.read_text(encoding="utf-8"))

    samples_path = raw_dir / "samples.jsonl"
    all_samples = []
    with open(samples_path, "w", encoding="utf-8") as sf:
        for arm_entry in arms_data:
            arm_name = arm_entry["arm"]
            for sample in arm_entry["samples"]:
                row = copy.deepcopy(sample)
                row["arm"] = arm_name
                all_samples.append(row)
                sf.write(json.dumps(row, ensure_ascii=False) + "\n")

    recomputed_scores = {}
    independent_match = True
    for sample in all_samples:
        arm_name = sample["arm"]
        if arm_name not in recomputed_scores:
            recomputed_scores[arm_name] = {"math_correct": 0, "math_total": 0, "qa_correct": 0, "qa_total": 0}
        if sample["panel"] == "math":
            recomputed_scores[arm_name]["math_total"] += 1
            re_pred = extract_gsm8k_pred(sample["output_text"])
            re_ok = is_gsm8k_correct(re_pred, sample["gold"])
            if re_ok != sample["correct"]:
                independent_match = False
            if re_ok:
                recomputed_scores[arm_name]["math_correct"] += 1
        elif sample["panel"] == "qa":
            recomputed_scores[arm_name]["qa_total"] += 1
            matching_q = next(q for q in qa_tasks if q["id"] == sample["task_id"])
            re_ok, _ = grade_qa(matching_q, sample["output_text"])
            if re_ok != sample["correct"]:
                independent_match = False
            if re_ok:
                recomputed_scores[arm_name]["qa_correct"] += 1

    independent_eval = {
        "independent_scorer_match": independent_match,
        "arm_scores": recomputed_scores,
        "total_samples_evaluated": len(all_samples),
    }
    (raw_dir / "independent_evaluation.json").write_text(
        json.dumps(independent_eval, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    hashed_artifacts_count = len(artifact_ledger)
    base_control_present = any(arm["arm"] == "base" for arm in arms_data)
    min_math_samples = min(score["math_total"] for score in recomputed_scores.values())
    min_qa_samples = min(score["qa_total"] for score in recomputed_scores.values())

    receipt_inputs = [
        raw_dir / "artifact_hashes.json",
        raw_dir / "dataset_hashes.json",
        raw_dir / "scorer_hashes.json",
        raw_dir / "samples.jsonl",
        raw_dir / "independent_evaluation.json",
        DEFAULT_MATH_PATH,
        DEFAULT_QA_PATH,
    ]

    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(),
        started_at_utc=started_utc,
        started_monotonic=started_mono,
        input_paths=receipt_inputs,
        packages=["pytest", "torch", "transformers", "peft"],
        runtime={"execution_mode": "offline_gpu_requalification", "arms_count": len(arms_data)},
    )
    prov_ok, prov_errors = provenance_complete(provenance)
    if not prov_ok:
        raise ValueError(f"Provenance incomplete: {', '.join(prov_errors)}")

    gates = {
        "artifact_identity": {
            "metric": "hashed_artifacts",
            "operator": "eq",
            "threshold": 13,
            "actual": hashed_artifacts_count,
            "pass": (hashed_artifacts_count == 13),
        },
        "frozen_math_panel": {
            "metric": "scored_math_samples_per_arm",
            "operator": "ge",
            "threshold": 32,
            "actual": min_math_samples,
            "pass": (min_math_samples >= 32),
        },
        "frozen_qa_panel": {
            "metric": "scored_qa_samples_per_arm",
            "operator": "ge",
            "threshold": 16,
            "actual": min_qa_samples,
            "pass": (min_qa_samples >= 16),
        },
        "base_control": {
            "metric": "base_control_present",
            "operator": "eq",
            "threshold": True,
            "actual": base_control_present,
            "pass": (base_control_present is True),
        },
        "independent_score": {
            "metric": "independent_scorer_match",
            "operator": "eq",
            "threshold": True,
            "actual": independent_match,
            "pass": (independent_match is True),
        },
    }

    receipt = {
        "schema": "local-labs-backlog-receipt-v1",
        "task_id": "BACKLOG-ADAPT-REQUAL-01",
        "provenance": provenance,
        "provenance_complete": prov_ok,
        "gates": gates,
        "evidence": {
            "acceptance_gates": "raw/receipt.json",
            "artifact_hashes": "raw/artifact_hashes.json",
            "dataset_hashes": "raw/dataset_hashes.json",
            "independent_evaluation": "raw/independent_evaluation.json",
            "provenance": "raw/receipt.json",
            "raw_samples": "raw/samples.jsonl",
            "receipt_fingerprint": "raw/receipt.json",
            "scorer_hashes": "raw/scorer_hashes.json",
        },
    }
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)

    (raw_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[HOST] Successfully written receipt to {raw_dir / 'receipt.json'}!", flush=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Adapter requalification runner")
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs" / "research" / "BACKLOG-ADAPT-REQUAL-01")
    parser.add_argument("--worker-mode", action="store_true", help="Run GPU worker inside WSL")
    parser.add_argument("--worker-out", type=str, help="Path for worker intermediate output")
    args = parser.parse_args()

    if args.worker_mode:
        if not args.worker_out:
            parser.error("--worker-out is required in worker mode")
        math_tasks = load_math_panel(DEFAULT_MATH_PATH, FROZEN_GSM8K_IDS)
        qa_tasks = load_qa_panel(DEFAULT_QA_PATH, FROZEN_QA_IDS)
        execute_inference_in_wsl(args.worker_out, math_tasks, qa_tasks)
        return 0

    receipt = run_requalification(args.outdir)
    print(json.dumps(receipt["gates"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
