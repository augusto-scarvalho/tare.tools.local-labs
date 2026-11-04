#!/usr/bin/env python3
"""Generate official BigCodeBench samples from a local OpenAI-compatible endpoint.

Dataset loading, prompt construction, and sanitization come from the revision-pinned
official package. This adapter only adds llama-server chat-template controls, resumable
per-task receipts, and deterministic index selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request
import warnings
from datetime import datetime, timezone

from bigcodebench.data import get_bigcodebench
from bigcodebench.provider.utility import make_raw_chat_prompt
from bigcodebench.sanitize import sanitize

warnings.filterwarnings("ignore", category=SyntaxWarning)


INSTRUCTION_PREFIX = ("Please provide a self-contained Python script that solves the following problem "
                      "in a markdown code block:")
RESPONSE_PREFIX = ("Below is a Python script with a self-contained function that solves the problem and "
                   "passes corresponding tests:")


def append_jsonl(path: pathlib.Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def completed(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()
    output = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                output.add(json.loads(line)["task_id"])
    return output


def post(base_url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:2000]}") from exc


def selfcheck() -> None:
    prompt = make_raw_chat_prompt("Write function f.", "hard", "instruct", INSTRUCTION_PREFIX,
                                  RESPONSE_PREFIX, tokenizer=None)
    assert prompt.startswith(INSTRUCTION_PREFIX) and "Write function f." in prompt
    code = sanitize("```python\ndef f():\n    return 1\n```", "f")
    assert "def f" in code and "return 1" in code
    print("BigCodeBench local generator self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--subset", choices=("hard", "full"), default="hard")
    parser.add_argument("--split", choices=("instruct", "complete"), default="instruct")
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--max-tokens", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--samples", type=pathlib.Path, required=True)
    parser.add_argument("--receipts", type=pathlib.Path, required=True)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0

    dataset = get_bigcodebench(subset=args.subset)
    items = list(dataset.items())
    selected = set(args.indices) if args.indices is not None else set(range(len(items)))
    if selected and (min(selected) < 0 or max(selected) >= len(items)):
        raise SystemExit(f"indices outside dataset of {len(items)} tasks")
    done = completed(args.samples)
    for index, (task_id, task) in enumerate(items):
        if index not in selected or task_id in done:
            continue
        task_prompt = task[f"{args.split}_prompt"]
        prompt = make_raw_chat_prompt(task_prompt, args.subset, args.split, INSTRUCTION_PREFIX,
                                      RESPONSE_PREFIX, tokenizer=None)
        payload = {"model": "qwen38-27b", "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.0, "top_p": 0.95, "seed": args.seed,
                   "max_tokens": args.max_tokens, "stream": False, "cache_prompt": False,
                   "chat_template_kwargs": {"enable_thinking": False}}
        started = time.monotonic()
        response = post(args.base_url, payload, args.timeout)
        wall_s = time.monotonic() - started
        choice = (response.get("choices") or [{}])[0]
        raw_solution = ((choice.get("message") or {}).get("content") or "")
        solution = sanitize(raw_solution, task["entry_point"])
        sample = {"task_id": task_id, "solution": solution, "raw_solution": raw_solution}
        append_jsonl(args.samples, sample)
        usage = response.get("usage") or {}
        receipt = {"task_id": task_id, "dataset_index": index,
                   "timestamp": datetime.now(timezone.utc).isoformat(),
                   "entry_point": task["entry_point"], "prompt_sha256":
                   hashlib.sha256(prompt.encode()).hexdigest(), "raw_sha256":
                   hashlib.sha256(raw_solution.encode()).hexdigest(), "solution_sha256":
                   hashlib.sha256(solution.encode()).hexdigest(), "finish_reason": choice.get("finish_reason"),
                   "usage": usage, "timings": response.get("timings") or {}, "wall_s": wall_s,
                   "raw_chars": len(raw_solution), "sanitized_chars": len(solution)}
        append_jsonl(args.receipts, receipt)
        print(f"{index:03}/{len(items)} {task_id:<20} finish={choice.get('finish_reason'):<10} "
              f"tokens={usage.get('completion_tokens')} chars={len(solution)} wall={wall_s:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
