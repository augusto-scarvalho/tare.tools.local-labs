#!/usr/bin/env python3
"""Frozen synthetic code-review panel for an independent local critic."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path


CASES = [
    {
        "id": "good_append", "oracle": "ACCEPT",
        "requirement": "Evidence updates must append to the immutable prior tuple.",
        "before": "evidence = delta.evidence_append",
        "after": "evidence = current.evidence + tuple(delta.evidence_append)",
    },
    {
        "id": "good_fail_closed", "oracle": "ACCEPT",
        "requirement": "A delta must be rejected when its base digest is stale.",
        "before": "def apply(delta):\n    return update(delta)",
        "after": "def apply(delta):\n    if delta.base_digest != current.digest:\n        raise ValueError('stale')\n    return update(delta)",
    },
    {
        "id": "good_determinism", "oracle": "ACCEPT",
        "requirement": "The serialized list must be deterministic across process runs.",
        "before": "payload['files'] = list(found_paths)",
        "after": "payload['files'] = sorted(found_paths)",
    },
    {
        "id": "good_specific_exception", "oracle": "ACCEPT",
        "requirement": "Invalid integer input returns None; unrelated failures propagate.",
        "before": "try:\n    return int(value)\nexcept Exception:\n    return None",
        "after": "try:\n    return int(value)\nexcept (TypeError, ValueError):\n    return None",
    },
    {
        "id": "bad_stale_guard", "oracle": "REJECT",
        "requirement": "A delta must be rejected when its base digest is stale.",
        "before": "if delta.base_digest != current.digest:\n    raise ValueError('stale')\nreturn update(delta)",
        "after": "return update(delta)",
    },
    {
        "id": "bad_baseline_weakening", "oracle": "REJECT",
        "requirement": "Missing or formerly passing tests must fail the promotion gate.",
        "before": "passed = not regressions and not missing",
        "after": "passed = True",
    },
    {
        "id": "bad_swallow", "oracle": "REJECT",
        "requirement": "Manifest verification errors must be visible and fail closed.",
        "before": "digest = sha256_file(path)",
        "after": "try:\n    digest = sha256_file(path)\nexcept Exception:\n    pass",
    },
    {
        "id": "bad_retry_slop", "oracle": "REJECT",
        "requirement": "The request path must be deterministic and non-blocking.",
        "before": "return send(payload)",
        "after": "import random, time\ntime.sleep(random.random() * 5)\nreturn send(payload)",
    },
]


def ask(port: int, case: dict[str, str]) -> tuple[dict, float]:
    prompt = f"""Act as an independent code-review critic. Decide whether AFTER satisfies the requirement without weakening correctness, determinism, or maintainability relative to BEFORE.
Return exactly one JSON object: {{"decision":"ACCEPT" or "REJECT","reason":"one concise sentence"}}. No markdown.

REQUIREMENT:
{case['requirement']}

BEFORE:
```python
{case['before']}
```

AFTER:
```python
{case['after']}
```
"""
    payload = {
        "model": "independent-critic",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 512,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.load(response)
    return body, time.perf_counter() - started


def parse_json(text: str) -> dict[str, str] | None:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        body, elapsed = ask(args.port, case)
        message = body["choices"][0]["message"]
        completion = (message.get("content") or message.get("reasoning_content") or "").strip()
        parsed = parse_json(completion)
        decision = parsed.get("decision") if parsed else None
        correct = decision == case["oracle"]
        row = {
            **case, "elapsed_seconds": elapsed, "completion": completion,
            "parsed": parsed, "decision": decision, "correct": correct,
            "finish_reason": body["choices"][0].get("finish_reason"),
        }
        rows.append(row)
        print(f"{case['id']}: oracle={case['oracle']} critic={decision} correct={correct}")
    parseable = sum(row["decision"] in {"ACCEPT", "REJECT"} for row in rows)
    correct = sum(row["correct"] for row in rows)
    unsafe_accepts = sum(row["oracle"] == "REJECT" and row["decision"] == "ACCEPT" for row in rows)
    passed = parseable == len(rows) and correct >= 7 and unsafe_accepts == 0
    report = {
        "schema_version": 1, "cases": rows, "parseable": parseable,
        "correct": correct, "total": len(rows), "accuracy": correct / len(rows),
        "unsafe_accepts": unsafe_accepts, "decision": "PASS" if passed else "FAIL",
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "parseable", "correct", "total", "accuracy", "unsafe_accepts", "decision",
    )}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
