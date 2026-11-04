#!/usr/bin/env python3
"""LAB-AGENT-003: bounded tool-calling stress and scale curves."""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from datetime import datetime, timezone

from agent_suite_v2 import STOCK, WEATHER, base_payload, calls, post, tool


DISTRACTOR_LEVELS = (2, 8, 16, 32)
PARALLEL_LEVELS = (2, 4, 8, 12)
SEQUENTIAL_LEVELS = (0, 2, 4, 8)
HISTORY_LEVELS = (0, 4, 8, 16)
CITIES = ("Lisbon", "Tokyo", "Nairobi", "Lima", "Oslo", "Seoul",
          "Cairo", "Perth", "Dublin", "Mumbai", "Quebec City", "Santiago")


def distractor(index: int) -> dict:
    return tool(f"catalog_lookup_{index:02d}",
                f"Look up a non-financial catalog item in namespace {index}; not for market prices.",
                {"query": {"type": "string"}}, ["query"])


def distractor_case(count: int) -> tuple[list[dict], list[dict], callable, dict]:
    tools = [distractor(i) for i in range(count - 1)]
    tools.insert(len(tools) // 2, STOCK)
    messages = [{"role": "user", "content": "What is NVDA trading at now? Use the appropriate tool."}]

    def validate(found):
        return (len(found) == 1 and found[0]["name"] == "get_stock_price"
                and found[0]["args_valid"]
                and str((found[0]["args"] or {}).get("ticker", "")).upper() == "NVDA")
    return tools, messages, validate, {"tool_count": count,
                                       "target_position": next(i for i, x in enumerate(tools)
                                                               if x["function"]["name"] == "get_stock_price")}


def parallel_case(fanout: int) -> tuple[list[dict], list[dict], callable, dict]:
    cities = CITIES[:fanout]
    joined = ", ".join(cities[:-1]) + f", and {cities[-1]}"
    messages = [{"role": "user", "content":
                 f"Get current weather for {joined}. Call the weather tool exactly once per city."}]

    def validate(found):
        observed = [str((call.get("args") or {}).get("city", "")).casefold()
                    for call in found if call.get("name") == "get_weather" and call.get("args_valid")]
        return len(found) == fanout and sorted(observed) == sorted(city.casefold() for city in cities)
    return [WEATHER], messages, validate, {"fanout": fanout, "cities": cities}


def sequential_case(depth: int) -> tuple[list[dict], list[dict], callable, dict]:
    advance = tool("advance_pipeline", "Advance exactly one workflow stage using the token from the prior result.", {
        "workflow_id": {"type": "string"}, "stage": {"type": "integer"},
        "token": {"type": "string"}}, ["workflow_id", "stage", "token"])
    target = depth + 1
    messages = [{"role": "user", "content":
                 f"Advance workflow wf-77 through stage {target}. Call advance_pipeline once per stage, "
                 "starting stage 1 with token root and then using each returned next_token. "
                 "Continue now without asking for confirmation."}]
    for stage in range(1, depth + 1):
        token_value = "root" if stage == 1 else f"tok-{stage}"
        call_id = f"stage_{stage}"
        messages.extend([
            {"role": "assistant", "content": None, "tool_calls": [{"id": call_id,
             "type": "function", "function": {"name": "advance_pipeline",
             "arguments": json.dumps({"workflow_id": "wf-77", "stage": stage,
                                      "token": token_value})}}]},
            {"role": "tool", "tool_call_id": call_id,
             "content": json.dumps({"completed_stage": stage, "next_stage": stage + 1,
                                    "next_token": f"tok-{stage + 1}"})},
        ])
    expected_token = "root" if target == 1 else f"tok-{target}"

    def validate(found):
        if len(found) != 1 or found[0]["name"] != "advance_pipeline" or not found[0]["args_valid"]:
            return False
        args = found[0]["args"] or {}
        return (args.get("workflow_id") == "wf-77" and args.get("stage") == target
                and args.get("token") == expected_token)
    return [advance], messages, validate, {"completed_depth": depth, "next_stage": target}


def history_case(turns: int) -> tuple[list[dict], list[dict], callable, dict]:
    messages = []
    for turn in range(turns):
        messages.extend([
            {"role": "user", "content": f"For context only, remember label note-{turn}. No tool needed."},
            {"role": "assistant", "content": f"Noted label note-{turn}."},
        ])
    messages.append({"role": "user", "content": "Now check the current AAPL share price."})

    def validate(found):
        return (len(found) == 1 and found[0]["name"] == "get_stock_price"
                and found[0]["args_valid"]
                and str((found[0]["args"] or {}).get("ticker", "")).upper() == "AAPL")
    return [WEATHER, STOCK], messages, validate, {"history_turns": turns}


AXES = {
    "distractors": (DISTRACTOR_LEVELS, distractor_case),
    "parallel": (PARALLEL_LEVELS, parallel_case),
    "sequential": (SEQUENTIAL_LEVELS, sequential_case),
    "history": (HISTORY_LEVELS, history_case),
}


def selfcheck() -> None:
    for axis, (levels, factory) in AXES.items():
        for level in levels:
            tools, messages, validator, metadata = factory(level)
            assert tools and messages and callable(validator) and metadata
    tools, _, _, metadata = distractor_case(32)
    assert len(tools) == 32 and metadata["target_position"] == 15
    _, messages, _, _ = sequential_case(8)
    assert len(messages) == 17
    print("agent stress/scale v2 self-check OK (16 cells)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--axes", nargs="+", choices=tuple(AXES), default=list(AXES))
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("runs/agent/LAB-AGENT-003-2026-08-22/results.json"))
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0

    rows = []
    for axis in args.axes:
        levels, factory = AXES[axis]
        for level in levels:
            tools, messages, validate, metadata = factory(level)
            payload = base_payload(tools, messages)
            payload["seed"] = args.seed
            started = time.monotonic()
            response = post(args.base_url, payload, args.timeout)
            found = calls(response)
            passed = validate(found)
            row = {"axis": axis, "level": level, "pass": passed,
                   "seconds": round(time.monotonic() - started, 3), "metadata": metadata,
                   "calls": found, "error": response.get("_error"), "raw_response": response}
            rows.append(row)
            print(f"{axis:<12} level={level:<3} {'PASS' if passed else 'FAIL'} "
                  f"calls={len(found):<2} {row['seconds']:>7.2f}s", flush=True)

    by_axis = {}
    for axis in args.axes:
        group = [row for row in rows if row["axis"] == axis]
        prefix = []
        for row in group:
            if not row["pass"]:
                break
            prefix.append(row["level"])
        by_axis[axis] = {"passed": sum(row["pass"] for row in group), "total": len(group),
                         "all_pass": all(row["pass"] for row in group),
                         "largest_contiguous_passing_level": prefix[-1] if prefix else None}
    report = {"campaign": "LAB-AGENT-003-v2", "timestamp": datetime.now(timezone.utc).isoformat(),
              "endpoint": args.base_url, "seed": args.seed, "axes": args.axes,
              "method": "bounded local stress curves; not BFCL-comparable",
              "summary": {"passed": sum(row["pass"] for row in rows), "total": len(rows),
                          "all_pass": all(row["pass"] for row in rows), "by_axis": by_axis},
              "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"evidence: {args.output}")
    return 0 if report["summary"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
