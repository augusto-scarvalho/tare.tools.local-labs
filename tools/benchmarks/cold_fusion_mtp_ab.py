#!/usr/bin/env python3
"""Counterbalanced same-file Cold Fusion embedded-MTP A/B. Run inside WSL."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import signal
import statistics
import subprocess
import time
import urllib.request


BIN = "/home/augus/src/slop.cpp/build/bin/llama-server"
MODEL = ("/home/augus/models/qwen38-27b/cold-fusion-27a5cb2c/"
         "Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NM-DAU-NEO-MAX-NEO-MTP-IQ4_XS.gguf")
TEMPLATE = "/home/augus/models/templates/qwen-sharp.jinja"
PORT = 8092

TASKS = [
    {"id": "arithmetic", "prompt":
     "Question: What is 17 multiplied by 23? Answer with only the integer.\nAnswer:", "n": 384},
    {"id": "code", "prompt":
     "Write only Python code for def clamp(value, lower, upper). It must raise ValueError when lower "
     "is greater than upper, otherwise return value bounded inclusively to [lower, upper].\n", "n": 384},
    {"id": "red_black_tree", "prompt":
     "Explain how a red-black tree repairs an insertion that creates two consecutive red nodes. Cover "
     "the uncle-red recolor case, triangle rotation, line rotation, root handling, and why every case "
     "preserves black height. Use precise paragraphs and finish with a compact invariant checklist.\n\nAnswer:",
     "n": 384},
]
ORDERS = [
    ["off", "n2", "n3"],
    ["n3", "n2", "off"],
    ["n2", "off", "n3"],
]


def wait_health(timeout: float = 600) -> float | None:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3):
                return time.monotonic() - started
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return None


def gpu_used() -> int | None:
    p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    try:
        return int(p.stdout.strip().splitlines()[0]) if p.returncode == 0 else None
    except (ValueError, IndexError):
        return None


def complete(prompt: str, n_predict: int) -> dict:
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
                       "top_k": 1, "seed": 42, "cache_prompt": False}).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=1200) as response:
        raw = json.load(response)
    content = raw.get("content") or ""
    return {"wall_s": time.monotonic() - started, "content": content,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "timings": raw.get("timings") or {}, "stop": raw.get("stop"),
            "stopped_eos": raw.get("stopped_eos"), "tokens_predicted": raw.get("tokens_predicted")}


def correct(task: str, content: str) -> bool:
    text = content.lower()
    if task == "arithmetic":
        numbers = re.findall(r"-?\d+", content)
        return bool(numbers and numbers[0] == "391")
    if task == "code":
        return all(s in text for s in ("def clamp", "valueerror", "lower", "upper", "return"))
    if task == "red_black_tree":
        return all(s in text for s in ("uncle", "recolor", "rotate", "root", "black height"))
    return False


def argv(arm: str) -> list[str]:
    args = [BIN, "-m", MODEL, "--alias", f"cold-fusion-{arm}", "--host", "127.0.0.1",
            "--port", str(PORT), "--ctx-size", "32768", "-np", "1", "--gpu-layers", "all",
            "--flash-attn", "on", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
            "--jinja", "--chat-template-file", TEMPLATE]
    if arm != "off":
        args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", arm[1:]]
    return args


def run_arm(block: int, position: int, arm: str, out: pathlib.Path) -> dict:
    args = argv(arm)
    log_path = out / f"block{block + 1}-{position + 1}-{arm}.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, preexec_fn=os.setsid)
    load_s = wait_health()
    if load_s is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
        return {"block": block + 1, "position": position + 1, "arm": arm, "argv": args,
                "valid": False, "error": "server_not_ready", "log": str(log_path)}
    used_mib = gpu_used()
    try:
        complete("Warmup. Continue with one short sentence.\nAnswer:", 64)
        rows = []
        for task in TASKS:
            row = complete(task["prompt"], task["n"])
            row.update(task=task["id"], correct=correct(task["id"], row["content"]))
            rows.append(row)
        return {"block": block + 1, "position": position + 1, "arm": arm, "argv": args,
                "valid": all(r["content"] and r["correct"] for r in rows), "load_s": load_s,
                "used_mib": used_mib, "rows": rows, "log": str(log_path)}
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--reanalyze", action="store_true",
                        help="summarize the retained partial receipt without generation")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reanalyze:
        runs = json.loads((args.output_dir / "mtp-ab.partial.json").read_text(
            encoding="utf-8"))["runs"]
    else:
        runs = []
        for block, order in enumerate(ORDERS):
            for position, arm in enumerate(order):
                print(f"block {block + 1} position {position + 1}: {arm}", flush=True)
                cell = run_arm(block, position, arm, args.output_dir)
                runs.append(cell)
                (args.output_dir / "mtp-ab.partial.json").write_text(
                    json.dumps({"campaign": "LAB-COLD-FUSION-002", "runs": runs},
                               indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                print(f"  valid={cell.get('valid')} used={cell.get('used_mib')}", flush=True)

    summary = {}
    for arm in ("off", "n2", "n3"):
        # Runtime validity and task correctness are separate. A completed but incorrect cell is
        # model evidence, not a runtime failure, and must remain in the mechanism summary.
        cells = [r for r in runs if r["arm"] == arm and r.get("rows") and not r.get("error")]
        task_summary = {}
        for task in (t["id"] for t in TASKS):
            rows = [row for cell in cells for row in cell["rows"] if row["task"] == task]
            tps = [row["timings"].get("predicted_per_second") for row in rows]
            tps = [float(x) for x in tps if x]
            task_summary[task] = {"n": len(rows), "all_correct": all(row["correct"] for row in rows),
                                  "hash_stable": len({row["sha256"] for row in rows}) == 1,
                                  "median_decode_tps": statistics.median(tps) if tps else None}
        summary[arm] = {"runtime_valid_cells": len(cells), "tasks": task_summary}

    for arm in ("n2", "n3"):
        for task in summary[arm]["tasks"]:
            arm_rows = [row for cell in runs if cell["arm"] == arm and cell.get("rows")
                        for row in cell["rows"] if row["task"] == task]
            off_rows = [row for cell in runs if cell["arm"] == "off" and cell.get("rows")
                        for row in cell["rows"] if row["task"] == task]
            summary[arm]["tasks"][task]["byte_equal_to_off"] = (
                len(arm_rows) == len(off_rows) and
                all(a["sha256"] == b["sha256"] for a, b in zip(arm_rows, off_rows, strict=True)))
            base = summary["off"]["tasks"][task]["median_decode_tps"]
            value = summary[arm]["tasks"][task]["median_decode_tps"]
            summary[arm]["tasks"][task]["speedup_vs_off"] = value / base if value and base else None

    valid = all(s["runtime_valid_cells"] == 3 for s in summary.values())
    task_correct = all(t["all_correct"] for s in summary.values() for t in s["tasks"].values())
    equivalent = all(summary[a]["tasks"][t].get("byte_equal_to_off")
                     for a in ("n2", "n3") for t in summary[a]["tasks"])
    speed_gate = all(
        summary[a]["tasks"][t].get("speedup_vs_off", 0) >= 1.10
        for a in ("n2", "n3") for t in summary[a]["tasks"])
    qualified = valid and task_correct and equivalent and speed_gate
    report = {"campaign": "LAB-COLD-FUSION-002", "runs": runs, "summary": summary,
              "runtime_valid": valid, "all_tasks_correct": task_correct,
              "byte_equivalent": equivalent, "speed_gate": speed_gate,
              "decision": "MTP_QUALIFIED" if qualified else
                          ("BLOCKED_RUNTIME" if not valid else
                           ("MTP_REJECTED" if not task_correct or not equivalent else
                            "MTP_UNRESOLVED"))}
    (args.output_dir / "mtp-ab.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"runtime_valid": valid, "all_tasks_correct": task_correct,
                      "byte_equivalent": equivalent, "speed_gate": speed_gate,
                      "decision": report["decision"], "summary": summary}, indent=2))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
