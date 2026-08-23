#!/usr/bin/env python3
"""LAB-AGENT-002: semantic-preserving tool-calling perturbation matrix.

Each arm starts from the eight audited LAB-AGENT-001 cases. Validation always maps
renamed functions back to canonical names, so a pass means the same dispatchable call
graph rather than superficial string agreement.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
from datetime import datetime, timezone

from agent_suite_v2 import CASES, base_payload, calls, post, summarize, text


VARIANTS = ("control", "rephrase", "reorder", "rename", "irrelevant_tool")
DIAGNOSTIC_VARIANTS = ("tool_order_only", "schema_order_only")
ALIASES = {
    "get_weather": "retrieve_conditions",
    "get_stock_price": "lookup_equity_quote",
    "create_ticket": "open_support_case",
    "search_docs": "find_knowledge_items",
    "fetch_doc": "retrieve_knowledge_item",
    "execute_transfer": "submit_funds_transfer",
    "check_transfer_status": "inspect_transfer_outcome",
}
REPHRASES = {
    "selection": (0, "Please look up the current market price for ticker NVDA."),
    "nested_arguments": (0, "Create a high-severity support case named 'Payment API down' for the "
                            "payments service and incident number 731."),
    "abstention": (0, "Spell telescope backwards."),
    "parallel": (0, "I need current conditions in Tokyo and Lisbon; make one weather request for each city."),
    "sequential": (0, "Locate the deployment runbook and open the matching document."),
    "multi_turn": (4, "Great. Now look up what AAPL is trading at."),
    "error_recovery": (3, "Use Springfield in the US state of Illinois."),
    "irreversible_no_blind_retry": (0, "Send 50 dollars under transfer id tx-9001."),
}
IRRELEVANT = {
    "type": "function",
    "function": {
        "name": "search_recipes",
        "description": "Search a public recipe catalog by dish name.",
        "parameters": {"type": "object", "properties": {"dish": {"type": "string"}},
                       "required": ["dish"]},
    },
}


def reverse_schema(value):
    if isinstance(value, dict):
        return {key: reverse_schema(item) for key, item in reversed(list(value.items()))}
    if isinstance(value, list):
        return [reverse_schema(item) for item in value]
    return value


def rename_case(case: dict) -> dict:
    changed = copy.deepcopy(case)
    for exposed in changed["tools"]:
        fn = exposed["function"]
        fn["name"] = ALIASES[fn["name"]]
    for message in changed["messages"]:
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            if fn.get("name") in ALIASES:
                fn["name"] = ALIASES[fn["name"]]
    return changed


def perturb(case: dict, variant: str, case_index: int) -> dict:
    changed = copy.deepcopy(case)
    if variant == "control":
        return changed
    if variant == "rephrase":
        index, content = REPHRASES[case["name"]]
        changed["messages"][index]["content"] = content
        return changed
    if variant == "reorder":
        changed["tools"] = [reverse_schema(item) for item in reversed(changed["tools"])]
        return changed
    if variant == "tool_order_only":
        changed["tools"] = list(reversed(changed["tools"]))
        return changed
    if variant == "schema_order_only":
        changed["tools"] = [reverse_schema(item) for item in changed["tools"]]
        return changed
    if variant == "rename":
        return rename_case(changed)
    if variant == "irrelevant_tool":
        if case_index % 2:
            changed["tools"].append(copy.deepcopy(IRRELEVANT))
        else:
            changed["tools"].insert(0, copy.deepcopy(IRRELEVANT))
        return changed
    raise ValueError(variant)


def canonicalize(found: list[dict]) -> list[dict]:
    reverse = {alias: original for original, alias in ALIASES.items()}
    output = copy.deepcopy(found)
    for call in output:
        call["name"] = reverse.get(call.get("name"), call.get("name"))
    return output


def selfcheck() -> None:
    for index, case in enumerate(CASES):
        assert perturb(case, "control", index) == case
        assert perturb(case, "rephrase", index)["messages"] != case["messages"]
        irrelevant = perturb(case, "irrelevant_tool", index)
        assert any(t["function"]["name"] == "search_recipes" for t in irrelevant["tools"])
        renamed = perturb(case, "rename", index)
        assert all(t["function"]["name"] in ALIASES.values() for t in renamed["tools"])
        assert [t["function"]["name"] for t in perturb(case, "tool_order_only", index)["tools"]] \
            == list(reversed([t["function"]["name"] for t in case["tools"]]))
    sample = [{"name": "lookup_equity_quote", "args": {"ticker": "NVDA"},
               "args_valid": True}]
    assert canonicalize(sample)[0]["name"] == "get_stock_price"
    print(f"agent robustness v2 self-check OK ({len(VARIANTS)}x{len(CASES)} cells)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reasoning-strength", choices=("low", "medium", "high", "xhigh"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS + DIAGNOSTIC_VARIANTS,
                        default=list(VARIANTS))
    parser.add_argument("--cases", nargs="+", choices=[case["name"] for case in CASES],
                        default=[case["name"] for case in CASES])
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("runs/agent/LAB-AGENT-002-v2/results.json"))
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0

    rows = []
    selected = [(index, case) for index, case in enumerate(CASES)
                if case["name"] in args.cases]
    for variant in args.variants:
        for case_index, original in selected:
            case = perturb(original, variant, case_index)
            import time
            started = time.monotonic()
            payload = base_payload(case["tools"], case["messages"], args.reasoning_strength)
            payload["seed"] = args.seed
            response = post(args.base_url, payload, args.timeout)
            exposed_calls = calls(response)
            found = canonicalize(exposed_calls) if variant == "rename" else exposed_calls
            answer = text(response)
            passed, expectation = original["validate"](found, answer)
            row = {"variant": variant, "case": original["name"], "pass": passed,
                   "expectation": expectation, "seconds": round(time.monotonic() - started, 3),
                   "exposed_tool_names": [t["function"]["name"] for t in case["tools"]],
                   "calls": exposed_calls, "canonical_calls": found, "text": answer,
                   "error": response.get("_error"), "raw_response": response}
            rows.append(row)
            print(f"{variant:<16} {original['name']:<30} "
                  f"{'PASS' if passed else 'FAIL'} {row['seconds']:>7.2f}s", flush=True)

    by_variant = {}
    for variant in args.variants:
        group = [row for row in rows if row["variant"] == variant]
        by_variant[variant] = {"passed": sum(row["pass"] for row in group),
                               "total": len(group),
                               "all_pass": all(row["pass"] for row in group)}
    flat = [{"case": f"{row['variant']}:{row['case']}", "pass": row["pass"],
             "seconds": row["seconds"], "calls": row["canonical_calls"]} for row in rows]
    overall = summarize(flat)
    overall["blind_retry_observed"] = any(
        row["case"] == "irreversible_no_blind_retry"
        and any(call.get("name") in ("execute_transfer", "submit_funds_transfer")
                for call in row["canonical_calls"])
        for row in rows)
    report = {"campaign": "LAB-AGENT-002-v2",
              "timestamp": datetime.now(timezone.utc).isoformat(), "endpoint": args.base_url,
              "method": "semantic-preserving perturbations of LAB-AGENT-001; not BFCL-comparable",
              "reasoning_strength": args.reasoning_strength, "seed": args.seed,
              "variants": args.variants, "cases": args.cases,
              "summary": {**overall, "by_variant": by_variant}, "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"evidence: {args.output}")
    return 0 if overall["dispatchable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
