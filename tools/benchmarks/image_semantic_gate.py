#!/usr/bin/env python3
"""Clause-based semantic gate for the frozen LAB-IMG-001 image panel."""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
import urllib.request
from pathlib import Path


CASES = [
    {
        "id": "typography",
        "prompt": (
            "Transcribe every visible word and number exactly. Then briefly state whether "
            "there is any other visible text."
        ),
        "clauses": [
            ["tare lab"], ["build 10161"], ["status ready"],
        ],
    },
    {
        "id": "dashboard",
        "prompt": "Transcribe the label and value of each status card exactly. Be concise.",
        "clauses": [
            ["queue 7"], ["gpu 68%", "gpu 68 %"], ["cache ok"],
        ],
    },
    {
        "id": "composition",
        "prompt": (
            "Describe each geometric object, its color, and its relative position. Also "
            "state the background color. Be concise."
        ),
        "clauses": [
            ["red cube"], ["left"], ["blue sphere"], ["right"],
            ["green triangle"], ["above", "top"],
            ["white background", "background white", "background off white"],
        ],
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
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.load(response)
    return body, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--case", action="append", choices=[case["id"] for case in CASES])
    args = parser.parse_args()
    rows = []
    selected = [case for case in CASES if not args.case or case["id"] in args.case]
    for case in selected:
        body, elapsed = ask(args.port, args.images / f"{case['id']}.png", case["prompt"])
        choice = body["choices"][0]
        message = choice["message"]
        completion = (message.get("content") or message.get("reasoning_content") or "").strip()
        normalized = " ".join(re.sub(r"[^a-z0-9%]+", " ", completion.lower()).split())
        clause_passes = [any(term in normalized for term in alternatives) for alternatives in case["clauses"]]
        rows.append({
            "id": case["id"], "prompt": case["prompt"], "completion": completion,
            "elapsed_seconds": elapsed, "finish_reason": choice.get("finish_reason"),
            "clause_passes": clause_passes, "clauses_passed": sum(clause_passes),
            "clauses_total": len(clause_passes), "case_pass": all(clause_passes),
            "usage": body.get("usage", {}),
        })
        print(f"{case['id']}: {sum(clause_passes)}/{len(clause_passes)} in {elapsed:.2f}s")
    passed = sum(row["clauses_passed"] for row in rows)
    total = sum(row["clauses_total"] for row in rows)
    cases_passed = sum(row["case_pass"] for row in rows)
    report = {
        "schema_version": 1, "cases": rows, "clauses_passed": passed,
        "clauses_total": total, "clause_rate": passed / total,
        "cases_passed": cases_passed,
        "decision": "PASS" if passed / total >= 0.85 and cases_passed >= 2 else "FAIL_QUALITY",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
    if report["decision"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
