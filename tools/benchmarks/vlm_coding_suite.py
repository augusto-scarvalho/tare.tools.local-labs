#!/usr/bin/env python3
"""Deterministic visual-coding accept suite for a local OpenAI-compatible VLM."""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path


CASES = [
    {
        "id": "stack_trace", "image": "stack_trace.png",
        "prompt": "Read the screenshot. State the exception, source file, line number, and implicated parameter. Be concise.",
        "clauses": [["nullreferenceexception"], ["paymentservice.cs"], ["132"], ["order"]],
    },
    {
        "id": "ui_bug", "image": "ui_bug.png",
        "prompt": "Identify the primary visible layout defect. Name the affected control and the container relationship. Be concise.",
        "clauses": [["checkout"], ["clip", "overflow", "outside", "cut off"], ["card", "container", "panel"]],
    },
    {
        "id": "visual_diff", "image": "visual_diff.png",
        "prompt": "Compare BEFORE and AFTER. List the button-label change, button-color change, and alert-badge change.",
        "clauses": [["deploy"], ["delete"], ["green"], ["red"], ["3 alerts"], ["missing", "removed", "gone", "absent", "no longer"]],
    },
    {
        "id": "terminal_failure", "image": "terminal_failure.png",
        "prompt": "Extract the failing test, expected and actual HTTP status, exception and key, and source file with line.",
        "clauses": [["test_login"], ["200"], ["500"], ["keyerror", "key error"], ["user_id"], ["auth.py"], ["87"]],
    },
]


def ask(port: int, image: Path, prompt: str) -> tuple[dict, float]:
    encoded = base64.b64encode(image.read_bytes()).decode()
    payload = {
        "model": "vlm",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ]}],
        "temperature": 0,
        "max_tokens": 512,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.load(response)
    return body, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        body, elapsed = ask(args.port, args.fixtures / case["image"], case["prompt"])
        choice = body["choices"][0]
        message = choice["message"]
        completion = (message.get("content") or message.get("reasoning_content") or "").strip()
        normalized = completion.lower()
        clause_passes = [any(term in normalized for term in alternatives) for alternatives in case["clauses"]]
        row = {
            "id": case["id"], "image": case["image"], "prompt": case["prompt"],
            "completion": completion, "elapsed_seconds": elapsed,
            "finish_reason": choice.get("finish_reason"), "clause_passes": clause_passes,
            "clauses_passed": sum(clause_passes), "clauses_total": len(clause_passes),
            "case_pass": all(clause_passes), "usage": body.get("usage", {}),
        }
        rows.append(row)
        print(f"{case['id']}: {row['clauses_passed']}/{row['clauses_total']} in {elapsed:.2f}s")
    clauses_passed = sum(row["clauses_passed"] for row in rows)
    clauses_total = sum(row["clauses_total"] for row in rows)
    nonempty = sum(bool(row["completion"]) for row in rows)
    cases_passed = sum(row["case_pass"] for row in rows)
    passed = nonempty == 4 and cases_passed >= 3 and clauses_passed / clauses_total >= 0.85
    report = {
        "schema_version": 1, "cases": rows, "nonempty": nonempty,
        "cases_passed": cases_passed, "clauses_passed": clauses_passed,
        "clauses_total": clauses_total, "clause_rate": clauses_passed / clauses_total,
        "decision": "PASS" if passed else "FAIL_QUALITY",
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
