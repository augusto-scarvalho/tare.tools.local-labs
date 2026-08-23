#!/usr/bin/env python3
"""Run the frozen LAB-SERVE-001d paired closed-loop matrix from Windows Python."""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[2]
LMCTL = ROOT / "tools" / "benchmarks" / "lmctl.py"
BENCH = ROOT / "ops" / "serving-campaign" / "lab_serve_bench.py"
WSL_PYTHON = "/home/augus/sglang-venv/bin/python"
TOKENIZER = "/home/augus/models/fp16/base"
TARGET = "lab-qwen36-moe-nograph"
PORT = 8092
CONCURRENCIES = (1, 2, 4, 6, 8)
SCHEDULE = {
    1: ("on", "off"), 2: ("off", "on"), 3: ("off", "on"),
    4: ("on", "off"), 5: ("on", "off"),
}
COMMON = [
    "-fa", "on", "--n-cpu-moe", "8", "--ctx-size", "32768",
    "--parallel", "8", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
    "--batch-size", "2048", "--ubatch-size", "2048", "--jinja",
]
MTP = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]


def run(argv: list[str], timeout: float = 900) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def as_wsl(path: pathlib.Path) -> str:
    value = str(path.resolve())
    drive, rest = value[0].lower(), value[2:].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def stop() -> None:
    run([sys.executable, str(LMCTL), "stop", "--port", str(PORT)], timeout=60)


def serve(arm: str) -> tuple[bool, str, str]:
    extra = [*COMMON, *(MTP if arm == "on" else [])]
    result = run([sys.executable, str(LMCTL), "serve", TARGET, "--port", str(PORT),
                  "--timeout", "180", "--", *extra], timeout=240)
    combined = (result.stdout or "") + (result.stderr or "")
    argv = next((line.split("argv:", 1)[1].strip() for line in combined.splitlines()
                 if "argv:" in line), "")
    return result.returncode == 0 and "UP in" in result.stdout, argv, combined


def get_json(path: str) -> object:
    result = run(["wsl.exe", "-d", "Ubuntu-24.04", "--", "curl", "-fsS", path], timeout=30)
    return json.loads(result.stdout)


def topology() -> dict:
    try:
        props = get_json(f"http://127.0.0.1:{PORT}/props")
        slots = get_json(f"http://127.0.0.1:{PORT}/slots")
        return {"total_slots": props.get("total_slots"), "n_ctx": props.get("n_ctx"),
                "num_slots": len(slots),
                "slot_contexts": [slot.get("n_ctx") for slot in slots]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def acceptance() -> list[dict]:
    probe = "Summarize continuous batching and expert routing in detail. " * 16
    payload = json.dumps({"prompt": probe, "n_predict": 128, "temperature": 0,
                          "cache_prompt": False})
    rows = []
    for _ in range(3):
        result = run(["wsl.exe", "-d", "Ubuntu-24.04", "--", "curl", "-fsS",
                      f"http://127.0.0.1:{PORT}/completion", "-H",
                      "Content-Type: application/json", "-d", payload], timeout=300)
        try:
            timing = json.loads(result.stdout).get("timings", {})
            drafted = timing.get("draft_n") or 0
            accepted = timing.get("draft_n_accepted") or 0
            rows.append({"draft_n": drafted, "accepted": accepted,
                         "rate": round(accepted / drafted, 4) if drafted else None})
        except json.JSONDecodeError:
            rows.append({"error": result.stderr[-300:]})
    return rows


def bench(raw: pathlib.Path, tag: str, concurrency: int, output_len: int) -> dict:
    raw.mkdir(parents=True, exist_ok=True)
    num_prompts = concurrency * 8
    argv = ["wsl.exe", "-d", "Ubuntu-24.04", "--", WSL_PYTHON, as_wsl(BENCH),
            "--tag", tag, "--outdir", as_wsl(raw),
            "--base-url", f"http://127.0.0.1:{PORT}", "--model", "qwen36-35b",
            "--tokenizer", TOKENIZER, "--concurrency", str(concurrency),
            "--input-len", "1024", "--output-len", str(output_len),
            "--num-prompts", str(num_prompts), "--warmup", "2"]
    result = run(argv, timeout=1800)
    normalized = raw / f"{tag}.normalized.json"
    debug = raw / f"{tag}.orchestrator.txt"
    debug.write_text(" ".join(argv) + "\n" + result.stdout + "\n" + result.stderr,
                     encoding="utf-8")
    if not normalized.exists():
        return {"tag": tag, "error": "normalized output missing",
                "returncode": result.returncode}
    record = json.loads(normalized.read_text(encoding="utf-8"))
    validity = record.get("validity", {})
    summary = record.get("upstream_summary", {})
    return {
        "tag": tag, "returncode": result.returncode,
        "success_all": validity.get("success_all"),
        "token_accounting_sane": validity.get("token_accounting_sane"),
        "token_ratio": validity.get("token_accounting_ratio"),
        "requested": validity.get("requested"), "completed": validity.get("completed"),
        "output_throughput": summary.get("output_throughput"),
        "request_throughput": summary.get("request_throughput"),
        "median_tpot_ms": summary.get("median_tpot_ms"),
        "p95_tpot_ms": summary.get("p95_tpot_ms"),
        "median_ttft_ms": summary.get("median_ttft_ms"),
        "median_e2e_ms": summary.get("median_e2e_latency_ms"),
        "p95_e2e_ms": summary.get("p95_e2e_latency_ms"),
        "gpu": record.get("gpu", {}),
    }


def selfcheck() -> None:
    assert CONCURRENCIES == (1, 2, 4, 6, 8)
    assert len(SCHEDULE) == 5 and all(set(order) == {"on", "off"}
                                      for order in SCHEDULE.values())
    assert COMMON.count("--parallel") == 1 and MTP not in COMMON
    assert as_wsl(ROOT).startswith("/mnt/")
    print("LAB-SERVE-001d self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, required=True)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--cooldown", type=int, default=8)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    raw = args.outdir / "raw"
    args.outdir.mkdir(parents=True, exist_ok=True)
    blocks: list[dict] = []
    invalid = False
    try:
        for rep in range(1, args.reps + 1):
            for arm in SCHEDULE[rep]:
                stop()
                time.sleep(2)
                up, exact_argv, launch_log = serve(arm)
                block = {"rep": rep, "arm": arm, "up": up, "argv": exact_argv,
                         "launch_log_tail": launch_log[-1000:], "topology": None,
                         "acceptance": None, "cells": []}
                if not up:
                    invalid = True
                else:
                    block["topology"] = topology()
                    block["acceptance"] = acceptance() if arm == "on" else []
                    order = list(CONCURRENCIES)
                    random.Random(1000 + rep).shuffle(order)
                    for concurrency in order:
                        tag = f"rep{rep}_{arm}_n{concurrency}_o128"
                        cell = bench(raw, tag, concurrency, 128)
                        block["cells"].append(cell)
                        invalid |= not (cell.get("success_all") and
                                        cell.get("token_accounting_sane"))
                        print(f"{tag}: ok={cell.get('success_all')} "
                              f"tpot={cell.get('median_tpot_ms')} "
                              f"throughput={cell.get('output_throughput')}", flush=True)
                    if rep == 1:
                        for output_len in (32, 512):
                            tag = f"rep{rep}_{arm}_n4_o{output_len}"
                            cell = bench(raw, tag, 4, output_len)
                            block["cells"].append(cell)
                            invalid |= not (cell.get("success_all") and
                                            cell.get("token_accounting_sane"))
                            print(f"{tag}: ok={cell.get('success_all')} "
                                  f"tpot={cell.get('median_tpot_ms')} "
                                  f"throughput={cell.get('output_throughput')}", flush=True)
                blocks.append(block)
                (args.outdir / "blocks.json").write_text(
                    json.dumps(blocks, indent=2), encoding="utf-8")
                stop()
                time.sleep(args.cooldown)
    finally:
        stop()
    return 2 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
