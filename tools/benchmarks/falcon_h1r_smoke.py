#!/usr/bin/env python3
"""Bounded official-sampling smoke panel for Falcon-H1R."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


PROMPTS = [
    "Reply with only the integer result of 17 + 25.",
    "Write a Python function named add that returns the sum of two numbers.",
    "In one sentence, explain why a hybrid Transformer-Mamba model can reduce KV-cache growth.",
    "Reply with exactly OK.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8092/v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for prompt in PROMPTS:
        payload = {
            "model": "falcon-h1r-7b", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6, "top_p": 0.95, "seed": 42, "max_tokens": 2048,
        }
        request = urllib.request.Request(
            args.base_url + "/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.load(response)
        choice = body["choices"][0]
        message = choice["message"]
        content = (message.get("content") or "").strip()
        reasoning = (message.get("reasoning_content") or "").strip()
        rows.append({
            "prompt": prompt, "content": content, "reasoning_content": reasoning,
            "finish_reason": choice.get("finish_reason"), "usage": body.get("usage", {}),
            "elapsed_seconds": time.perf_counter() - started,
            "nonempty": bool(content), "natural_stop": choice.get("finish_reason") == "stop",
        })
        print(f"finish={choice.get('finish_reason')} content={bool(content)} reasoning={bool(reasoning)}")
    report = {
        "schema_version": 1, "rows": rows,
        "nonempty": sum(row["nonempty"] for row in rows),
        "natural_stops": sum(row["natural_stop"] for row in rows),
    }
    report["pass"] = report["nonempty"] == 4 and report["natural_stops"] >= 3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

