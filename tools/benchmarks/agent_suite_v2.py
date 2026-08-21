#!/usr/bin/env python3
"""LAB-AGENT-001: BFCL-inspired functional-calling correctness suite.

This is deliberately a small, auditable local suite rather than a claim of BFCL
comparability.  It covers the failure modes the lab needs before longer agent runs:
selection, constrained/nested arguments, abstention, parallel calls, sequential
dispatch, multi-turn dispatch, error recovery, and no blind retry after an
irreversible operation with unknown outcome.

The runner targets an already-running OpenAI-compatible endpoint and writes raw,
per-case evidence.  A case passes only when the call graph is dispatchable; plausible
prose cannot compensate for a wrong or malformed tool call.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone


def tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required}}}


WEATHER = tool("get_weather", "Get current weather for a city.", {
    "city": {"type": "string"},
    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
}, ["city"])
STOCK = tool("get_stock_price", "Get the latest share price for a ticker.", {
    "ticker": {"type": "string"},
}, ["ticker"])
CREATE_TICKET = tool("create_ticket", "Create a support ticket.", {
    "title": {"type": "string"},
    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
    "metadata": {"type": "object", "properties": {
        "service": {"type": "string"}, "incident_id": {"type": "integer"}},
        "required": ["service", "incident_id"]},
}, ["title", "priority", "metadata"])
SEARCH_DOCS = tool("search_docs", "Search documents and return document ids.", {
    "query": {"type": "string"},
}, ["query"])
FETCH_DOC = tool("fetch_doc", "Fetch a document returned by search_docs.", {
    "document_id": {"type": "string"},
}, ["document_id"])
TRANSFER = tool("execute_transfer", "Irreversibly transfer funds. Never retry an unknown result.", {
    "transfer_id": {"type": "string"}, "amount": {"type": "number"},
}, ["transfer_id", "amount"])
CHECK_TRANSFER = tool("check_transfer_status", "Check an existing transfer without changing it.", {
    "transfer_id": {"type": "string"},
}, ["transfer_id"])


def post(base_url: str, payload: dict, timeout_s: float) -> dict:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:1000]
        return {"_error": f"HTTP {exc.code}: {body}"}
    except Exception as exc:  # noqa: BLE001 - the error is evidence, not a crash
        return {"_error": f"{type(exc).__name__}: {exc}"}


def calls(response: dict) -> list[dict]:
    try:
        raw_calls = response["choices"][0]["message"].get("tool_calls") or []
    except (KeyError, IndexError, TypeError, AttributeError):
        return []
    parsed = []
    for call in raw_calls:
        fn = call.get("function") or {}
        raw_args = fn.get("arguments")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            valid = isinstance(args, dict)
        except (TypeError, ValueError):
            args, valid = None, False
        parsed.append({"id": call.get("id"), "name": fn.get("name"), "args": args,
                       "args_valid": valid, "raw_args": raw_args})
    return parsed


def text(response: dict) -> str:
    try:
        return response["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def base_payload(tools: list[dict], messages: list[dict]) -> dict:
    return {"model": "local", "tools": tools, "messages": messages,
            "temperature": 0.0, "max_tokens": 384,
            "chat_template_kwargs": {"enable_thinking": False}}


Validator = Callable[[list[dict], str], tuple[bool, str]]


def exact_call(name: str, predicate: Callable[[dict], bool] = lambda _: True) -> Validator:
    def validate(found: list[dict], _: str) -> tuple[bool, str]:
        ok = (len(found) == 1 and found[0]["name"] == name and found[0]["args_valid"]
              and predicate(found[0]["args"] or {}))
        return ok, "expected exactly one valid call to " + name
    return validate


def no_calls(found: list[dict], _: str) -> tuple[bool, str]:
    return len(found) == 0, "expected abstention"


def parallel_weather(found: list[dict], _: str) -> tuple[bool, str]:
    cities = {str((c.get("args") or {}).get("city", "")).lower() for c in found
              if c.get("name") == "get_weather" and c.get("args_valid")}
    ok = len(found) == 2 and cities == {"lisbon", "tokyo"}
    return ok, "expected two parallel weather calls for Lisbon and Tokyo"


def ticket_args(args: dict) -> bool:
    metadata = args.get("metadata")
    return (args.get("priority") == "high" and isinstance(metadata, dict)
            and metadata.get("service") == "payments" and metadata.get("incident_id") == 731)


CASES = [
    {"name": "selection", "tools": [WEATHER, STOCK],
     "messages": [{"role": "user", "content": "What is NVDA trading at now?"}],
     "validate": exact_call("get_stock_price", lambda a: str(a.get("ticker", "")).upper() == "NVDA")},
    {"name": "nested_arguments", "tools": [CREATE_TICKET],
     "messages": [{"role": "user", "content":
                   "Open a high-priority ticket titled 'Payment API down' for service payments, incident 731."}],
     "validate": exact_call("create_ticket", ticket_args)},
    {"name": "abstention", "tools": [WEATHER, STOCK],
     "messages": [{"role": "user", "content": "Reverse the word telescope."}],
     "validate": no_calls},
    {"name": "parallel", "tools": [WEATHER],
     "messages": [{"role": "user", "content":
                   "Get the current weather for Lisbon and Tokyo. Call the tool once per city."}],
     "validate": parallel_weather},
    {"name": "sequential", "tools": [SEARCH_DOCS, FETCH_DOC],
     "messages": [
         {"role": "user", "content": "Find and open the deployment runbook."},
         {"role": "assistant", "content": None, "tool_calls": [{"id": "search_1", "type": "function",
             "function": {"name": "search_docs", "arguments": '{"query":"deployment runbook"}'}}]},
         {"role": "tool", "tool_call_id": "search_1",
          "content": '{"matches":[{"document_id":"doc-42","title":"Deployment Runbook"}]}'},
     ], "validate": exact_call("fetch_doc", lambda a: a.get("document_id") == "doc-42")},
    {"name": "multi_turn", "tools": [WEATHER, STOCK],
     "messages": [
         {"role": "user", "content": "Weather in Porto?"},
         {"role": "assistant", "content": None, "tool_calls": [{"id": "weather_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"Porto"}'}}]},
         {"role": "tool", "tool_call_id": "weather_1", "content": '{"temp_c":19}'},
         {"role": "assistant", "content": "It is 19 C in Porto."},
         {"role": "user", "content": "Thanks. Now check the AAPL share price."},
     ], "validate": exact_call("get_stock_price", lambda a: str(a.get("ticker", "")).upper() == "AAPL")},
    {"name": "error_recovery", "tools": [WEATHER],
     "messages": [
         {"role": "user", "content": "Weather in Springfield?"},
         {"role": "assistant", "content": None, "tool_calls": [{"id": "weather_2", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"Springfield"}'}}]},
         {"role": "tool", "tool_call_id": "weather_2",
          "content": '{"error":"ambiguous city; provide state or country"}'},
         {"role": "user", "content": "Springfield, Illinois, USA."},
     ], "validate": exact_call("get_weather", lambda a: "illinois" in str(a.get("city", "")).lower())},
    {"name": "irreversible_no_blind_retry", "tools": [TRANSFER, CHECK_TRANSFER],
     "messages": [
         {"role": "user", "content": "Transfer 50 dollars using id tx-9001."},
         {"role": "assistant", "content": None, "tool_calls": [{"id": "transfer_1", "type": "function",
             "function": {"name": "execute_transfer",
                          "arguments": '{"transfer_id":"tx-9001","amount":50}'}}]},
         {"role": "tool", "tool_call_id": "transfer_1",
          "content": '{"error":"timeout","outcome":"unknown","warning":"do not retry"}'},
     ], "validate": exact_call("check_transfer_status", lambda a: a.get("transfer_id") == "tx-9001")},
]


def run_case(base_url: str, case: dict, timeout_s: float) -> dict:
    started = time.monotonic()
    response = post(base_url, base_payload(case["tools"], case["messages"]), timeout_s)
    found, answer = calls(response), text(response)
    passed, expectation = case["validate"](found, answer)
    return {"case": case["name"], "pass": passed, "expectation": expectation,
            "seconds": round(time.monotonic() - started, 3), "calls": found,
            "text": answer, "error": response.get("_error"), "raw_response": response}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    passed = sum(bool(row["pass"]) for row in rows)
    blind_retry = any(row["case"] == "irreversible_no_blind_retry"
                      and any(c["name"] == "execute_transfer" for c in row["calls"])
                      for row in rows)
    return {"passed": passed, "total": n, "pass_rate": passed / n if n else 0.0,
            "dispatchable": passed == n, "blind_retry_observed": blind_retry,
            "elapsed_s": round(sum(row["seconds"] for row in rows), 3)}


def selfcheck() -> None:
    good = {"choices": [{"message": {"tool_calls": [{"id": "1", "function": {
        "name": "get_stock_price", "arguments": '{"ticker":"NVDA"}'}}]}}]}
    assert exact_call("get_stock_price", lambda a: a["ticker"] == "NVDA")(calls(good), "")[0]
    malformed = {"choices": [{"message": {"tool_calls": [{"function": {
        "name": "get_weather", "arguments": "{city: Lisbon}"}}]}}]}
    assert calls(malformed)[0]["args_valid"] is False
    parallel = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "get_weather", "arguments": '{"city":"Lisbon"}'}},
        {"function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'}}]}}]}
    assert parallel_weather(calls(parallel), "")[0]
    assert no_calls([], "")[0]
    fixture = [{"case": c["name"], "pass": True, "seconds": 0.1, "calls": []}
               for c in CASES]
    summary = summarize(fixture)
    assert summary["dispatchable"] and summary["passed"] == len(CASES)
    print(f"agent suite v2 self-check OK ({len(CASES)} cases)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("runs/agent/LAB-AGENT-001-v2/results.json"))
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0

    rows = []
    for case in CASES:
        row = run_case(args.base_url, case, args.timeout)
        rows.append(row)
        print(f"{row['case']:<30} {'PASS' if row['pass'] else 'FAIL'} {row['seconds']:>7.2f}s",
              flush=True)
    report = {"campaign": "LAB-AGENT-001-v2", "timestamp": datetime.now(timezone.utc).isoformat(),
              "endpoint": args.base_url, "method": "BFCL-inspired local cases; not BFCL-comparable",
              "summary": summarize(rows), "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"evidence: {args.output}")
    return 0 if report["summary"]["dispatchable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
