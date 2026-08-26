#!/usr/bin/env python3
"""Canonical host runner for BACKLOG-ADAPT01-640-EVAL-01."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file

TASK_ID = "BACKLOG-ADAPT01-640-EVAL-01"
SEED = 20260827
MODEL = "/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe"
ADAPTER = ROOT / "runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/mechanisms/adapt01/lokr_5ep/adapter"
SOURCES = [ROOT / "runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/receipt.json", ADAPTER / "adapter_model.safetensors", ADAPTER / "adapter_config.json", ROOT / "runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/mechanisms/adapt01/lokr_5ep/metrics.json", ROOT / "workloads/gsm8k.jsonl", ROOT / "runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json", ROOT / "runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl"]


def write(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def wsl_path(path: pathlib.Path) -> str:
    path = path.resolve(); return f"/mnt/{path.drive[0].lower()}/{path.as_posix()[3:]}"


def state() -> dict:
    completed = subprocess.run(["wsl", "-d", "Ubuntu-24.04", "-u", "root", "-e", "systemctl", "show", "llm-inference.service", "-p", "MainPID", "-p", "NRestarts", "-p", "ActiveState", "--no-pager"], capture_output=True, text=True, check=False)
    values = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
    health = {}
    for port in (8080, 8081):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as response: health[str(port)] = response.status
        except Exception: health[str(port)] = None
    return {"values": values, "health": health}


def execute(outdir: pathlib.Path) -> tuple[dict, dict]:
    raw = outdir / "raw"
    if any(raw.iterdir()): raise RuntimeError("raw directory is not empty")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); mono = time.monotonic()
    ledger = {path.relative_to(ROOT).as_posix(): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in SOURCES}
    before = state()
    worker = ROOT / "tools/research/adapt01_640_eval_worker.py"; worker_json = raw / "worker.json"
    command = ["wsl", "-d", "Ubuntu-24.04", "-e", "/home/augus/.venvs/adapt00-20260824/bin/python", wsl_path(worker), "--model", MODEL, "--adapter", wsl_path(ADAPTER), "--teacher", wsl_path(ROOT / "runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json"), "--prompts", wsl_path(ROOT / "workloads/gsm8k.jsonl"), "--qa", wsl_path(ROOT / "runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl"), "--seed", str(SEED), "--output", wsl_path(worker_json)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=1800)
    (raw / "worker.stdout.log").write_text(completed.stdout, encoding="utf-8"); (raw / "worker.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode: raise RuntimeError(f"worker failed: {completed.stderr[-4000:]}")
    result = json.loads(worker_json.read_text(encoding="utf-8")); after = state()
    summary = result["summary"]; rows = result["target_results"] + result["protected_results"]
    recomputed = {"target_correct": sum(row["correct"] for row in result["target_results"]), "target_n": len(result["target_results"]), "protected_pass": sum(row["pass"] for row in result["protected_results"]), "protected_n": len(result["protected_results"]), "natural_eos": sum(row["natural_eos"] for row in rows), "generation_n": len(rows)}
    reported = {key: summary[key] for key in recomputed}
    unchanged = int(before["values"].get("MainPID") == after["values"].get("MainPID") and after["values"].get("NRestarts") == "0" and after["health"] == {"8080": 200, "8081": 200})
    metrics = {"scored_generations": len(rows), "target_correct": recomputed["target_correct"], "target_gain_over_base": recomputed["target_correct"] - 8, "protected_pass": recomputed["protected_pass"], "natural_eos": recomputed["natural_eos"], "target_teacher_length_ratio": summary["target_teacher_length_ratio"], "independent_score_match": int(recomputed == reported), "serving_process_unchanged": unchanged}
    write(raw / "actual_scores.json", metrics); write(raw / "artifact_hashes.json", ledger); write(raw / "dataset_hashes.json", {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in SOURCES[-3:]}); write(raw / "independent_evaluation.json", {"reported": reported, "recomputed": recomputed, "match": recomputed == reported}); write(raw / "scorer_hashes.json", {"runner": sha256_file(pathlib.Path(__file__).resolve()), "worker": sha256_file(worker), "math": sha256_file(ROOT / "tools/analysis/a2_stats.py"), "qa": sha256_file(ROOT / "tools/benchmarks/normal_qa_ab.py")}); write(raw / "service_maintenance.json", {"before": before, "after": after, "service_untouched": bool(unchanged)}); write(raw / "source_execution_receipt.json", {"parent_receipt_sha256": ledger["runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/receipt.json"]["sha256"], "training_seed": SEED})
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in result["target_results"]: stream.write(json.dumps({"panel": "math", **row}, ensure_ascii=False) + "\n")
        for row in result["protected_results"]: stream.write(json.dumps({"panel": "qa", **row}, ensure_ascii=False) + "\n")
    defs = {"panel_coverage": ("scored_generations", "eq", 48), "target_absolute": ("target_correct", "ge", 16), "target_gain": ("target_gain_over_base", "ge", 3), "qa_retention": ("protected_pass", "ge", 2), "natural_eos": ("natural_eos", "ge", 40), "length_control": ("target_teacher_length_ratio", "le", 1.25), "independent_score": ("independent_score_match", "eq", 1), "runtime_unchanged": ("serving_process_unchanged", "eq", 1)}
    ops = {"eq": lambda a,b:a==b, "ge": lambda a,b:a>=b, "le": lambda a,b:a<=b}; gates = {name: {"metric": metric, "operator": op, "threshold": threshold, "actual": metrics[metric], "pass": ops[op](metrics[metric], threshold)} for name, (metric, op, threshold) in defs.items()}
    evidence = {"acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json", "artifact_hashes": "raw/artifact_hashes.json", "dataset_hashes": "raw/dataset_hashes.json", "independent_evaluation": "raw/independent_evaluation.json", "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl", "receipt_fingerprint": "raw/receipt.json", "scorer_hashes": "raw/scorer_hashes.json", "service_maintenance": "raw/service_maintenance.json", "source_execution_receipt": "raw/source_execution_receipt.json"}
    files = sorted({raw / value.removeprefix("raw/") for value in evidence.values() if value != "raw/receipt.json"}) + [worker_json]
    provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(), started_at_utc=started, started_monotonic=mono, input_paths=[*SOURCES, worker, *files], packages=["torch", "transformers", "peft"], runtime={"execution_mode": "fresh_adapter_behavioral_completion", "command": command}); ok, errors = provenance_complete(provenance)
    if not ok: raise ValueError(errors)
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}; receipt["receipt_fingerprint"] = canonical_json_sha256(receipt); write(raw / "receipt.json", receipt)
    return receipt, metrics


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--outdir", type=pathlib.Path, default=ROOT/"runs/research"/TASK_ID); args=parser.parse_args(); receipt, metrics=execute(args.outdir.resolve()); passed=all(g["pass"] for g in receipt["gates"].values()); claim="ADAPT01_640_ARM_PROMOTED_R1" if passed else "ADAPT01_640_ARM_REJECTED_R1"; failed=[name for name,g in receipt["gates"].items() if not g["pass"]]; (args.outdir/"RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent AGY review.\n\nThe omitted fresh 640-step arm scored {metrics['target_correct']}/32 math and {metrics['protected_pass']}/16 QA with {metrics['natural_eos']}/48 natural EOS. Failed gates: {', '.join(failed) if failed else 'none'}.\n", encoding="utf-8"); print(json.dumps({"claim": claim, "metrics": metrics, "gates": receipt["gates"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
