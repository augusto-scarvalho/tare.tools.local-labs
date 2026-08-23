#!/usr/bin/env python3
"""Run revision-pinned official RULERv1 data through a local OpenAI endpoint.

Only transport and resumable receipts are local. Data generation, task definitions,
answer prefixes, token budgets, and string-match semantics remain NVIDIA RULER's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
MAX_TOKENS = {"vt": 30, "cwe": 120, "fwe": 50, "qa_1": 32, "qa_2": 32}


def post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def append_jsonl(path: pathlib.Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_done(path: pathlib.Path) -> set[tuple[int, str, int]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {
            (row["target_length"], row["task"], row["index"])
            for line in handle if line.strip() and (row := json.loads(line))
        }


def score(prediction: str, references: list[str], partial: bool) -> float:
    prediction = prediction.lower()
    matches = [float(reference.lower() in prediction) for reference in references]
    return max(matches) if partial else sum(matches) / len(matches)


def dataset_file(root: pathlib.Path, length: int, task: str) -> pathlib.Path:
    return root / str(length) / task / "test.jsonl"


def selfcheck() -> None:
    assert score("alpha BETA", ["alpha", "beta"], False) == 1.0
    assert score("alpha", ["alpha", "beta"], False) == 0.5
    assert score("the answer is beta", ["alpha", "beta"], True) == 1.0
    assert MAX_TOKENS["qa_2"] == 32 and len(TASKS) == 13
    print("RULER local endpoint adapter self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen38-27b")
    parser.add_argument("--data-root", type=pathlib.Path, required=False)
    parser.add_argument("--lengths", type=int, nargs="+", default=[65536, 131072])
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--receipts", type=pathlib.Path, required=False)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    if args.data_root is None or args.receipts is None:
        parser.error("--data-root and --receipts are required unless --selfcheck is used")

    base = args.base_url.rstrip("/")
    props = json.loads(urllib.request.urlopen(f"{base}/props", timeout=10).read().decode())
    server_ctx = props["default_generation_settings"]["n_ctx"]
    done = load_done(args.receipts)
    for length in args.lengths:
        for task in args.tasks:
            path = dataset_file(args.data_root, length, task)
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            for row in rows:
                key = (length, task, row["index"])
                if key in done:
                    continue
                prompt = row["input"] + row["answer_prefix"]
                template = post(
                    f"{base}/apply-template",
                    {"messages": [{"role": "user", "content": prompt}],
                     "chat_template_kwargs": {"enable_thinking": False},
                     "add_generation_prompt": True},
                    60,
                )["prompt"]
                actual_prompt_tokens = len(
                    post(f"{base}/tokenize", {"content": template, "add_special": False}, 180)["tokens"]
                )
                max_tokens = MAX_TOKENS.get(task, 128)
                if actual_prompt_tokens + max_tokens > server_ctx:
                    raise RuntimeError(
                        f"{length}/{task}/{row['index']} exceeds server context: "
                        f"{actual_prompt_tokens}+{max_tokens}>{server_ctx}"
                    )
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": args.seed,
                    "max_tokens": max_tokens,
                    "stream": False,
                    "cache_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                started = time.monotonic()
                response = post(f"{base}/v1/chat/completions", payload, args.timeout)
                wall_s = time.monotonic() - started
                choice = (response.get("choices") or [{}])[0]
                prediction = ((choice.get("message") or {}).get("content") or "")
                value = score(prediction, row["outputs"], task.startswith("qa_"))
                receipt = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "target_length": length,
                    "task": task,
                    "index": row["index"],
                    "dataset_reported_length": row.get("length_w_model_temp", row.get("length")),
                    "actual_prompt_tokens": actual_prompt_tokens,
                    "max_tokens": max_tokens,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "expected": row["outputs"],
                    "prediction": prediction,
                    "score": value,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": response.get("usage") or {},
                    "timings": response.get("timings") or {},
                    "wall_s": wall_s,
                }
                append_jsonl(args.receipts, receipt)
                print(
                    f"{length}/{task}/{row['index']} score={value:.3f} "
                    f"prompt={actual_prompt_tokens} wall={wall_s:.1f}s",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
