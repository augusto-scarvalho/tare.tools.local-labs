#!/usr/bin/env python3
"""Run the preregistered ADAPT-00B arms sequentially and preserve every outcome."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone


METHODS = ("dora", "loha", "lokr", "boft", "ia3", "trainable_tokens")
PREFLIGHT_PARAMETERS = {
    "dora": 5_811_264, "loha": 10_822_656, "lokr": 359_040,
    "boft": 1_505_856, "ia3": 239_616, "trainable_tokens": 2_026_496,
}


def pareto(rows: list[dict]) -> list[str]:
    """Maximize learning, minimize protected regression/parameters/VRAM."""
    passing = [row for row in rows if row.get("verdict") == "PASS"]
    winners = []
    for row in passing:
        a = row["metrics"]
        dominated = False
        for other in passing:
            if other is row:
                continue
            b = other["metrics"]
            no_worse = (
                b["target_improvement_fraction"] >= a["target_improvement_fraction"] and
                b["protected_regression_fraction"] <= a["protected_regression_fraction"] and
                other["trainable_parameters"] <= row["trainable_parameters"] and
                b["peak_allocated_vram_gib"] <= a["peak_allocated_vram_gib"])
            strictly = (
                b["target_improvement_fraction"] > a["target_improvement_fraction"] or
                b["protected_regression_fraction"] < a["protected_regression_fraction"] or
                other["trainable_parameters"] < row["trainable_parameters"] or
                b["peak_allocated_vram_gib"] < a["peak_allocated_vram_gib"])
            if no_worse and strictly:
                dominated = True
                break
        if not dominated:
            winners.append(row["method"])
    return winners


def result_row(method: str, path: pathlib.Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    return {
        "method": method,
        "verdict": result["verdict"],
        "trainable_parameters": result["parameters"]["trainable"],
        "metrics": result["metrics"],
        "gates": result["gates"],
        "receipt": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--include", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[2]
    arm_script = repo / "tools/probes/adapt00_lora_smoke.py"
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    execution_path = output_root / "execution.json"
    if args.summarize_only:
        execution = json.loads(execution_path.read_text(encoding="utf-8"))["arms"]
    else:
        execution = []
    for method in (() if args.summarize_only else args.include):
        arm_output = output_root / method
        command = [
            args.python, str(arm_script), "--method", method,
            "--model", "Qwen/Qwen3.5-0.8B-Base",
            "--model-path", args.model_path, "--revision", args.revision,
            "--teacher", str(repo / "runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json"),
            "--prompts", str(repo / "workloads/gsm8k.jsonl"),
            "--protected-file", str(repo / "README.md"),
            "--output", str(arm_output),
        ]
        print(f"=== ADAPT-00B {method} ===", flush=True)
        process = subprocess.run(command, text=True, capture_output=True, check=False)
        if process.stdout:
            print(process.stdout[-2500:], flush=True)
        if process.stderr:
            print(process.stderr[-2500:], file=sys.stderr, flush=True)
        metrics_path = arm_output / "metrics.json"
        execution.append({
            "method": method, "returncode": process.returncode,
            "metrics_written": metrics_path.exists(),
            "stdout_tail": process.stdout[-4000:], "stderr_tail": process.stderr[-4000:],
        })
        execution_path.write_text(json.dumps({
            "status": "RUNNING", "arms": execution}, indent=2) + "\n", encoding="utf-8")

    lora_path = output_root.parent.parent / "ADAPT-00A-MECHANICS-2026-08-24/raw/metrics.json"
    rows = []
    if lora_path.exists():
        rows.append(result_row("lora", lora_path))
    for method in args.include:
        path = output_root / method / "metrics.json"
        if path.exists():
            rows.append(result_row(method, path))
        else:
            attempted = next((row for row in execution if row["method"] == method), None)
            if attempted is not None:
                error = attempted.get("stderr_tail", "")
                verdict = ("FAIL_NONFINITE_LOSS_STEP_0" if
                           "non-finite loss at step 0" in error else "FAIL_EXECUTION")
                failure = {
                    "method": method, "verdict": verdict,
                    "trainable_parameters": PREFLIGHT_PARAMETERS[method],
                    "metrics": None,
                    "gates": {"finite_losses": False} if "NONFINITE" in verdict else {},
                    "error_tail": error,
                    "receipt": str(output_root / method / "failure.json"),
                }
                failure_path = output_root / method / "failure.json"
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                failure_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
                rows.append(failure)
    complete = len(rows) == 1 + len(args.include)
    matrix = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "common_budget": {"steps": 24, "batch_size": 1, "max_length": 384,
                          "learning_rate": 2e-4, "seed": 20260824},
        "pareto_frontier": pareto(rows),
        "arms": rows,
        "execution": execution,
    }
    (output_root / "matrix.json").write_text(
        json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    execution_path.write_text(json.dumps({
        "status": matrix["status"], "arms": execution}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": matrix["status"],
                      "pareto_frontier": matrix["pareto_frontier"],
                      "arm_verdicts": {row["method"]: row["verdict"] for row in rows}},
                     indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
