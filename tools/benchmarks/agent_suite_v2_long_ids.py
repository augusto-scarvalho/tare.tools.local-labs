#!/usr/bin/env python3
"""Agent Suite V2 compatibility runner with semantically inert long tool-call IDs."""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import time
from datetime import datetime, timezone

from agent_suite_v2 import CASES, base_payload, calls, post, summarize, text


def normalize_ids(case: dict) -> dict:
    changed = copy.deepcopy(case)
    mapping: dict[str, str] = {}
    for message in changed["messages"]:
        for call in message.get("tool_calls") or []:
            old = call.get("id")
            if old:
                mapping[old] = f"fixture_{old}"
                call["id"] = mapping[old]
        old_result = message.get("tool_call_id")
        if old_result:
            mapping.setdefault(old_result, f"fixture_{old_result}")
            message["tool_call_id"] = mapping[old_result]
    return changed


def selfcheck() -> None:
    for case in CASES:
        changed = normalize_ids(case)
        ids = []
        for message in changed["messages"]:
            ids += [call["id"] for call in message.get("tool_calls") or []]
            if message.get("tool_call_id"):
                ids.append(message["tool_call_id"])
        assert all(len(value) >= 9 for value in ids)
        assert [tool["function"]["name"] for tool in changed["tools"]] == [
            tool["function"]["name"] for tool in case["tools"]
        ]
    print("agent suite long-ID compatibility self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0

    rows = []
    for original in CASES:
        case = normalize_ids(original)
        payload = base_payload(case["tools"], case["messages"])
        payload["seed"] = args.seed
        started = time.monotonic()
        response = post(args.base_url, payload, args.timeout)
        found, answer = calls(response), text(response)
        passed, expectation = original["validate"](found, answer)
        row = {
            "case": original["name"], "pass": passed, "expectation": expectation,
            "seconds": round(time.monotonic() - started, 3), "calls": found, "text": answer,
            "error": response.get("_error"), "raw_response": response,
        }
        rows.append(row)
        print(f"{original['name']:<30} {'PASS' if passed else 'FAIL'} {row['seconds']:>7.2f}s")
    report = {
        "campaign": "agent-suite-v2-long-ids",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.base_url,
        "semantic_delta": "tool-call fixture IDs lengthened only",
        "seed": args.seed,
        "summary": summarize(rows),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["dispatchable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
