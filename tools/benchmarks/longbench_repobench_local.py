#!/usr/bin/env python3
"""Generate official LongBench RepoBench-P predictions via llama.cpp raw completion."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


PROMPT = "Please complete the code given below. \n{context}{input}Next line of code:"


def post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:2000]}") from exc


def append(path: pathlib.Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def completed(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["_id"] for line in path.read_text(encoding="utf-8").splitlines() if line}


def selfcheck() -> None:
    row = {"context": "class A:\n", "input": "    def f(self):\n"}
    assert PROMPT.format(**row).endswith("Next line of code:")
    print("LongBench RepoBench-P local generator self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--dataset", type=pathlib.Path)
    parser.add_argument("--predictions", type=pathlib.Path)
    parser.add_argument("--receipts", type=pathlib.Path)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--mode", choices=("raw", "chat"), default="raw")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    if args.dataset is None or args.predictions is None or args.receipts is None:
        parser.error("--dataset, --predictions, and --receipts are required")

    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line]
    selected = set(args.indices) if args.indices is not None else set(range(len(rows)))
    if selected and (min(selected) < 0 or max(selected) >= len(rows)):
        raise SystemExit(f"index outside dataset of {len(rows)} examples")
    done = completed(args.predictions)
    base = args.base_url.rstrip("/")
    props = json.loads(urllib.request.urlopen(f"{base}/props", timeout=10).read().decode())
    server_ctx = props["default_generation_settings"]["n_ctx"]
    for index, row in enumerate(rows):
        if index not in selected or row["_id"] in done:
            continue
        prompt = PROMPT.format(**row)
        prompt_tokens = len(post(f"{base}/tokenize", {"content": prompt, "add_special": False}, 180)["tokens"])
        if prompt_tokens + 64 > server_ctx:
            raise RuntimeError(f"index {index} exceeds context: {prompt_tokens}+64>{server_ctx}")
        payload = {"prompt": prompt, "n_predict": 64, "temperature": 0.0, "top_k": 1,
                   "top_p": 1.0, "seed": 42, "cache_prompt": False, "stream": False}
        endpoint = f"{base}/completion"
        if args.mode == "chat":
            payload = {"model": "qwen38-27b", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 64, "temperature": 0.0, "top_p": 1.0, "seed": 42,
                       "cache_prompt": False, "stream": False,
                       "chat_template_kwargs": {"enable_thinking": False}}
            endpoint = f"{base}/v1/chat/completions"
        started = time.monotonic()
        response = post(endpoint, payload, args.timeout)
        wall_s = time.monotonic() - started
        if args.mode == "raw":
            prediction = response.get("content") or ""
            stopped_eos = response.get("stopped_eos")
            stopped_limit = response.get("stopped_limit")
            tokens_predicted = response.get("tokens_predicted")
            tokens_evaluated = response.get("tokens_evaluated")
        else:
            choice = (response.get("choices") or [{}])[0]
            prediction = ((choice.get("message") or {}).get("content") or "")
            finish_reason = choice.get("finish_reason")
            stopped_eos = finish_reason == "stop"
            stopped_limit = finish_reason == "length"
            usage = response.get("usage") or {}
            tokens_predicted = usage.get("completion_tokens")
            tokens_evaluated = usage.get("prompt_tokens")
        append(args.predictions, {"_id": row["_id"], "pred": prediction, "answers": row["answers"],
                                  "all_classes": row["all_classes"], "length": row["length"]})
        append(args.receipts, {
            "timestamp": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
            "dataset_index": index, "_id": row["_id"],
            "language": row["language"], "reported_length": row["length"], "prompt_tokens": prompt_tokens,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prediction_sha256": hashlib.sha256(prediction.encode()).hexdigest(),
            "prediction": prediction, "nonempty": bool(prediction.strip()), "stopped_eos": stopped_eos,
            "stopped_limit": stopped_limit, "tokens_predicted": tokens_predicted,
            "tokens_evaluated": tokens_evaluated, "timings": response.get("timings") or {},
            "wall_s": wall_s,
        })
        print(f"{args.mode} {index:03d}/500 id={row['_id']} prompt={prompt_tokens} "
              f"chars={len(prediction)} wall={wall_s:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
