"""Rung 1 of the agentic ladder: can this model emit a valid tool call at all?

This is a GATE, not a score. A model that cannot produce a well-formed tool call fails
every agentic benchmark downstream for a reason that has nothing to do with how clever it
is -- the chat template baked into the GGUF, or the server's `--jinja` handling, decides
this before the weights get a say. Finding that out after fifteen hours of Aider polyglot
would be the expensive way.

Five cases, each covering a failure mode that actually kills agents in practice:

    single      one tool, obvious call            -- the floor
    enum        argument constrained to a set     -- models invent values here
    choose      two tools, only one is right      -- models call the first one they see
    followup    call -> result -> final answer    -- models loop, calling forever
    refrain     no tool is needed                 -- models call anyway, and agents burn
                                                     turns on nothing

`refrain` is the one people leave out, and over-calling is as fatal as under-calling: an
agent that invokes a tool for "hello" never finishes a task.

    python agentic_gate.py --model qwen36-35b
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from model_lifecycle.servers.llama_cpp import (                    # noqa: E402
    LlamaCppAdapter, ServerProfile)

LOCAL_BIN = "/home/augus/src/llama.cpp-local/build/bin/llama-server"

MODELS = {
    "qwen36-35b": ("/home/augus/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", 40),
    "qwen3-30b": ("/home/augus/models/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf", 48),
    "gpt-oss-20b": ("/home/augus/models/gpt-oss-20b/gpt-oss-20b-Q4_K_M.gguf", 24),
}

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

STOCK = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get the latest share price for a ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
}


def _post(base_url: str, payload: dict, timeout_s: float = 180.0) -> dict:
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"_error": f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"}
    except Exception as exc:                       # noqa: BLE001
        return {"_error": f"{type(exc).__name__}: {exc}"}


def _calls(resp: dict) -> list[dict]:
    """Tool calls from an OpenAI-shaped response, normalised. Returns [] when the model
    answered in prose instead -- which is a RESULT (it declined to call), not an error."""
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return []
    out = []
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        args = fn.get("arguments")
        # Arguments arrive as a JSON *string*. A model that emits invalid JSON here has
        # failed in the way that matters: the caller cannot dispatch it.
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
            ok = isinstance(parsed, dict)
        except (ValueError, TypeError):
            parsed, ok = None, False
        out.append({"name": fn.get("name"), "args": parsed, "args_valid": ok,
                    "raw": args})
    return out


def case_single(url: str) -> dict:
    r = _post(url, {"model": "local", "tools": [WEATHER], "max_tokens": 256,
                    "temperature": 0.0,
                    "messages": [{"role": "user",
                                  "content": "What is the weather in Lisbon right now?"}]})
    c = _calls(r)
    ok = (len(c) == 1 and c[0]["name"] == "get_weather" and c[0]["args_valid"]
          and isinstance((c[0]["args"] or {}).get("city"), str))
    return {"case": "single", "pass": ok, "calls": c, "error": r.get("_error")}


def case_enum(url: str) -> dict:
    r = _post(url, {"model": "local", "tools": [WEATHER], "max_tokens": 256,
                    "temperature": 0.0,
                    "messages": [{"role": "user",
                                  "content": "Weather in Tokyo, in fahrenheit please."}]})
    c = _calls(r)
    # The enum is the point: an invented unit is a dispatch failure downstream.
    ok = (len(c) == 1 and c[0]["args_valid"]
          and (c[0]["args"] or {}).get("unit") == "fahrenheit")
    return {"case": "enum", "pass": ok, "calls": c, "error": r.get("_error")}


def case_choose(url: str) -> dict:
    r = _post(url, {"model": "local", "tools": [WEATHER, STOCK], "max_tokens": 256,
                    "temperature": 0.0,
                    "messages": [{"role": "user",
                                  "content": "How much is NVDA trading at?"}]})
    c = _calls(r)
    ok = len(c) == 1 and c[0]["name"] == "get_stock_price" and c[0]["args_valid"]
    return {"case": "choose", "pass": ok, "calls": c, "error": r.get("_error")}


def case_followup(url: str) -> dict:
    """Call -> result -> answer. The failure mode is calling AGAIN instead of answering,
    which is how an agent spends a whole budget in a loop."""
    r = _post(url, {"model": "local", "tools": [WEATHER], "max_tokens": 256,
                    "temperature": 0.0, "messages": [
                        {"role": "user", "content": "Weather in Porto?"},
                        {"role": "assistant", "content": None, "tool_calls": [{
                            "id": "call_1", "type": "function",
                            "function": {"name": "get_weather",
                                         "arguments": '{"city": "Porto"}'}}]},
                        {"role": "tool", "tool_call_id": "call_1",
                         "content": '{"temp_c": 19, "sky": "clear"}'},
                    ]})
    c = _calls(r)
    text = ""
    try:
        text = (r["choices"][0]["message"].get("content") or "")
    except (KeyError, IndexError, TypeError):
        pass
    ok = len(c) == 0 and "19" in text
    return {"case": "followup", "pass": ok, "calls": c, "text": text[:200],
            "error": r.get("_error")}


def case_refrain(url: str) -> dict:
    """No tool applies. Over-calling is as fatal as under-calling: an agent that reaches
    for a tool on every turn never converges."""
    r = _post(url, {"model": "local", "tools": [WEATHER, STOCK], "max_tokens": 256,
                    "temperature": 0.0,
                    "messages": [{"role": "user",
                                  "content": "Write the word 'hello' backwards."}]})
    c = _calls(r)
    return {"case": "refrain", "pass": len(c) == 0, "calls": c, "error": r.get("_error")}


CASES = [case_single, case_enum, case_choose, case_followup, case_refrain]


def report(rows: list[dict], model: str, jinja: bool) -> bool:
    print("\n" + "=" * 68)
    print(f"TOOL-CALLING GATE  model={model}  --jinja={'on' if jinja else 'off'}")
    print("=" * 68)
    for r in rows:
        mark = "PASS" if r["pass"] else "FAIL"
        print(f"  {r['case']:<10} {mark}")
        if not r["pass"]:
            if r.get("error"):
                print(f"             error: {r['error'][:160]}")
            for c in r.get("calls") or []:
                print(f"             called {c['name']!r} args_valid={c['args_valid']} "
                      f"raw={str(c['raw'])[:100]!r}")
            if r.get("text"):
                print(f"             text: {r['text'][:120]!r}")
    n_pass = sum(1 for r in rows if r["pass"])
    print(f"\n  {n_pass}/{len(rows)} passed")

    # The verdict is a GATE decision, so it says what to do next rather than scoring.
    if n_pass == len(rows):
        print("  -> GATE OPEN: this model can be dispatched by an agent loop. Proceed to "
              "HumanEval+.")
    elif any(r["case"] == "single" and r["pass"] for r in rows):
        print("  -> PARTIAL: basic dispatch works, but a failing case above is a real "
              "agent failure mode.\n     Worth proceeding, with the failure recorded as a "
              "known handicap.")
    else:
        print("  -> GATE CLOSED: the model cannot emit a usable tool call. Every agentic "
              "benchmark below\n     would fail for this reason and not for capability. "
              "Check the GGUF's chat template and\n     that the server ran with --jinja "
              "BEFORE concluding anything about the model.")
    return n_pass == len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--ncmoe", type=int, default=None)
    ap.add_argument("--no-jinja", action="store_true",
                    help="run WITHOUT --jinja, to show the template is what decides this")
    args = ap.parse_args()

    gguf, blocks = MODELS[args.model]
    ncmoe = args.ncmoe if args.ncmoe is not None else max(1, round(blocks * 0.6))
    extra = () if args.no_jinja else ("--jinja",)

    adapter = LlamaCppAdapter(server_bin=LOCAL_BIN,
                              env={"GGML_CUDA_REGISTER_HOST": "1"})
    profile = ServerProfile(model_path=gguf, port=8080, n_cpu_moe=ncmoe, ctx_size=8192,
                            cache_type_k="q8_0", cache_type_v="q8_0", extra_args=extra)
    h = adapter.start(profile)
    try:
        if not adapter.wait_until_healthy(h, timeout_s=900):
            print("server never became healthy. argv:")
            print("  " + " ".join(adapter.argv(profile)))
            for ln in h.stderr_tail[-12:]:
                print(f"  | {ln}")
            return 2
        rows = []
        for fn in CASES:
            t0 = time.monotonic()
            r = fn(h.base_url)
            r["seconds"] = round(time.monotonic() - t0, 1)
            print(f"  ran {r['case']:<10} {'PASS' if r['pass'] else 'FAIL'} "
                  f"({r['seconds']}s)", flush=True)
            rows.append(r)
    finally:
        adapter.stop(h)
        adapter.force_stop(h)

    out = pathlib.Path(__file__).parent / "runs"
    out.mkdir(exist_ok=True)
    (out / f"agentic_gate_{args.model}.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    report(rows, args.model, not args.no_jinja)
    return 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        # Argument parsing is where a tool call is actually usable or not, so that is what
        # the self-check covers -- against the shapes servers really return.
        good = {"choices": [{"message": {"tool_calls": [
            {"function": {"name": "get_weather", "arguments": '{"city": "Lisbon"}'}}]}}]}
        c = _calls(good)
        assert len(c) == 1 and c[0]["args_valid"] and c[0]["args"]["city"] == "Lisbon"

        # Invalid JSON in `arguments` is the failure that matters: a caller cannot
        # dispatch it, however plausible the text looks.
        bad = {"choices": [{"message": {"tool_calls": [
            {"function": {"name": "get_weather", "arguments": '{city: Lisbon'}}]}}]}
        assert _calls(bad)[0]["args_valid"] is False

        # Answering in prose is a RESULT (declined to call), not a crash.
        prose = {"choices": [{"message": {"content": "It is sunny."}}]}
        assert _calls(prose) == []
        assert _calls({"_error": "HTTP 500"}) == []

        # Already-parsed dict arguments (some builds do this) must still validate.
        parsed = {"choices": [{"message": {"tool_calls": [
            {"function": {"name": "f", "arguments": {"a": 1}}}]}}]}
        assert _calls(parsed)[0]["args_valid"] is True

        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            report([{"case": "single", "pass": False, "calls": []}], "m", True)
        assert "GATE CLOSED" in buf.getvalue()
        buf = io.StringIO()
        with redirect_stdout(buf):
            report([{"case": c, "pass": True, "calls": []}
                    for c in ("single", "enum", "choose", "followup", "refrain")], "m", True)
        assert "GATE OPEN" in buf.getvalue()

        print("agentic_gate self-check OK")
        raise SystemExit(0)
    raise SystemExit(main())
