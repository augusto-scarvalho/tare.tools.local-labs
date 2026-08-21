#!/usr/bin/env python3
"""LAB-CODE-001 Tier-1 generation runner for MBPP+ against a live endpoint."""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from benchmark_harness_qa import (benchmark_content_hash, extract_code, flag_truncated,
                                  run_identity, validate_samples)  # noqa: E402

SUBSET_SEED = 20260726
INSTRUCTION = ("Solve the following Python programming task. Reply with the complete executable "
               "solution inside one ```python code block, and nothing else. Include all required "
               "imports and the requested function. Do not include tests or example usage.\n\n{prompt}")


def load_problems(path: pathlib.Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["task_id"].split("/")[-1]))
    return rows


def pick_subset(problems: list[dict], n: int) -> list[dict]:
    if n >= len(problems):
        return problems
    order = list(range(len(problems)))
    random.Random(SUBSET_SEED).shuffle(order)
    return [problems[index] for index in sorted(order[:n])]


def generate(base_url: str, problem: dict, max_tokens: int, timeout_s: float) -> dict:
    payload = {"model": "local", "messages": [{"role": "user",
               "content": INSTRUCTION.format(prompt=problem["prompt"])}],
               "temperature": 0.0, "top_k": 1, "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}
    request = urllib.request.Request(f"{base_url.rstrip('/')}/v1/chat/completions",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = json.loads(response.read().decode("utf-8"))
        choice = raw["choices"][0]
        answer = choice["message"].get("content") or ""
        error = None
        finish_reason = choice.get("finish_reason")
    except urllib.error.HTTPError as exc:
        raw, answer, finish_reason = {}, "", None
        error = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:500]}"
    except Exception as exc:  # noqa: BLE001
        raw, answer, finish_reason = {}, "", None
        error = f"{type(exc).__name__}: {exc}"
    timings = raw.get("timings") or {}
    predicted_n = timings.get("predicted_n")
    truncation_probe = {"task_id": problem["task_id"], "finish_reason": finish_reason,
                        "answer_tokens": predicted_n}
    record = {"task_id": problem["task_id"], "entry_point": problem["entry_point"],
              "answered": bool(answer.strip()), "fenced": "```" in answer,
              "finish_reason": finish_reason, "predicted_n": predicted_n,
              "decode_tps": timings.get("predicted_per_second"),
              "wall_s": round(time.monotonic() - started, 3), "error": error,
              "truncated": bool(flag_truncated([truncation_probe], max_tokens=max_tokens)),
              "completion": answer, "solution": extract_code(answer)}
    return record


def selfcheck() -> None:
    fixture = [{"task_id": f"Mbpp/{i}", "prompt": "p", "entry_point": "f"} for i in range(20)]
    assert [x["task_id"] for x in pick_subset(fixture, 5)] == [
        x["task_id"] for x in pick_subset(fixture, 5)]
    assert len(pick_subset(fixture, 5)) == 5
    assert extract_code("```python\ndef f(): return 1\n```") == "def f(): return 1"
    good = [{"task_id": row["task_id"], "solution": "def f(): pass"}
            for row in pick_subset(fixture, 5)]
    assert validate_samples(good, [row["task_id"] for row in pick_subset(fixture, 5)]) == []
    print("MBPP+ generation harness self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--problems", type=pathlib.Path,
                        default=ROOT / "workloads" / "mbpp_plus.jsonl")
    parser.add_argument("--subset", type=int, default=50)
    parser.add_argument("--task-id", action="append", default=[],
                        help="run explicit task id(s), bypassing the seeded subset")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=ROOT / "runs" / "code" / "LAB-CODE-001-MBPP-plus")
    parser.add_argument("--model-sha256", default=None)
    parser.add_argument("--engine-commit", default="UNKNOWN")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    all_problems = load_problems(args.problems)
    if args.task_id:
        requested = set(args.task_id)
        problems = [problem for problem in all_problems if problem["task_id"] in requested]
        missing = requested - {problem["task_id"] for problem in problems}
        if missing:
            raise ValueError(f"unknown --task-id values: {sorted(missing)}")
    else:
        problems = pick_subset(all_problems, args.subset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    samples_path = args.output_dir / "samples.jsonl"
    records = []
    with records_path.open("w", encoding="utf-8") as records_handle, \
            samples_path.open("w", encoding="utf-8") as samples_handle:
        for index, problem in enumerate(problems, 1):
            record = generate(args.base_url, problem, args.max_tokens, args.timeout)
            records.append(record)
            records_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_handle.flush()
            samples_handle.write(json.dumps({"task_id": record["task_id"],
                                              "solution": record["solution"]}) + "\n")
            samples_handle.flush()
            print(f"{index:>3}/{len(problems)} {record['task_id']:<10} "
                  f"{'OK' if record['answered'] else 'EMPTY'} {record['wall_s']:.2f}s", flush=True)
    validation_errors = validate_samples(
        [{"task_id": r["task_id"], "solution": r["solution"]} for r in records],
        [problem["task_id"] for problem in problems])
    identity = run_identity(
        benchmark_name="mbpp-plus", benchmark_version="evalplus-mbpp-v0.2.0",
        dataset_version=str(args.problems), problems=all_problems,
        sampling={"temperature": 0.0, "top_k": 1, "max_tokens": args.max_tokens,
                  "enable_thinking": False, "subset_seed": SUBSET_SEED},
        model_id="qwen38-27b", model_path="/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-IQ4_XS.gguf",
        quant="IQ4_XS", engine_commit=args.engine_commit,
        timestamp=datetime.now(timezone.utc).isoformat(), repo_root=ROOT,
        model_sha256=args.model_sha256, model_bytes=15705861088,
        source_repo="unsloth/Qwen3.8-27B-GGUF", source_revision="UNKNOWN",
        quantizer="upstream artifact; exact quantizer revision UNKNOWN", imatrix="static/no imatrix",
        provenance_class="COMMUNITY_REQUANT")
    identity.update({"benchmark_content_hash": benchmark_content_hash(all_problems),
                     "subset_task_ids": [p["task_id"] for p in problems],
                     "sample_validation": {"ok": not validation_errors,
                                           "errors": validation_errors},
                     "answered": sum(r["answered"] for r in records),
                     "fenced": sum(r["fenced"] for r in records),
                     "truncated": sum(r["truncated"] for r in records)})
    (args.output_dir / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    print(json.dumps({k: identity[k] for k in ("answered", "fenced", "truncated")}, indent=2))
    return 0 if not validation_errors and identity["answered"] == len(problems) else 1


if __name__ == "__main__":
    raise SystemExit(main())
