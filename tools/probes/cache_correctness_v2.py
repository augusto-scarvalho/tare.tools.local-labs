#!/usr/bin/env python3
"""LAB-CACHE-001: correctness probes for llama-server prompt-cache lifecycle.

The invariant is stronger than a latency win: a cached request must produce the same
greedy answer as a cold request, and both must satisfy a known-answer oracle.  Raw
responses and timing/cache counters are retained for audit.

This slice exercises exact reuse, divergent suffixes, partial removal, cancellation
followed by reuse, and long-context reuse.  Explicit slot file save/restore and
speculative rollback require different server launch conditions and remain separately
reported capabilities, never silently treated as covered.
"""
from __future__ import annotations

import argparse
import http.client
import json
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone


def post_json(url: str, payload: dict, timeout_s: float = 600.0) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"_error": f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:1000]}"}
    except Exception as exc:  # noqa: BLE001 - preserve failure as evidence
        return {"_error": f"{type(exc).__name__}: {exc}"}


def completion(base_url: str, prompt: str, cache_prompt: bool, n_predict: int = 64) -> dict:
    payload = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
               "seed": 0, "cache_prompt": cache_prompt, "stream": False}
    response = post_json(f"{base_url.rstrip('/')}/completion", payload)
    return {"content": response.get("content", ""), "timings": response.get("timings") or {},
            "error": response.get("_error"), "raw": response}


def normalize(answer: str) -> str:
    return " ".join(answer.strip().lower().replace("`", "").split()).strip(" .,!?:;\"'")


def compare_case(name: str, cold: dict, warm: dict, expected: str,
                 require_cache_hit: bool = True) -> dict:
    cold_norm, warm_norm, wanted = normalize(cold["content"]), normalize(warm["content"]), normalize(expected)
    cache_n = int((warm.get("timings") or {}).get("cache_n") or 0)
    exact = cold["content"] == warm["content"]
    oracle = wanted in cold_norm and wanted in warm_norm
    passed = not cold.get("error") and not warm.get("error") and exact and oracle
    if require_cache_hit:
        passed = passed and cache_n > 0
    return {"case": name, "pass": passed, "expected": expected,
            "exact_cold_warm": exact, "oracle_pass": oracle,
            "warm_cache_n": cache_n, "cold": cold, "warm": warm}


def prompt(shared: str, question: str) -> str:
    return (shared + "\n\n" + question +
            "\nReturn only the requested code word, with no explanation or punctuation.")


def wait_idle(base_url: str, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url.rstrip('/')}/slots", timeout=2) as response:
                slots = json.loads(response.read().decode("utf-8"))
            if slots and not any(bool(slot.get("is_processing")) for slot in slots):
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.1)
    return False


def cancel_stream(base_url: str, prompt_text: str) -> dict:
    """Start generation, consume a few SSE lines, then close the socket intentionally."""
    parsed = urllib.parse.urlsplit(base_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=30)
    payload = json.dumps({"prompt": prompt_text, "n_predict": 2048, "ignore_eos": True,
                          "temperature": 0.0, "top_k": 1, "seed": 0,
                          "cache_prompt": True, "stream": True})
    lines = []
    started = time.monotonic()
    try:
        conn.request("POST", "/completion", body=payload,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        while len(lines) < 3:
            line = response.readline().decode("utf-8", "replace").strip()
            if not line:
                continue
            lines.append(line[:500])
    finally:
        conn.close()
    idle = wait_idle(base_url)
    return {"lines_received": len(lines), "sample": lines,
            "idle_after_close": idle, "seconds": round(time.monotonic() - started, 3)}


def server_capabilities(base_url: str) -> dict:
    slots = post_json(f"{base_url.rstrip('/')}/slots/0?action=save",
                      {"filename": "lab-cache-capability-probe.bin"}, timeout_s=10)
    error = slots.get("_error")
    return {"slot_file_save_restore": "supported" if not error else "blocked_by_launch_config",
            "slot_probe_response": slots,
            "speculative_rollback": "not_exercised_server_reports_speculative_false"}


def run(base_url: str, n_predict: int = 64, nonce: str | None = None) -> dict:
    nonce = nonce or uuid.uuid4().hex
    rows = []

    shared = ((f"CACHE-{nonce}: routine telemetry is nominal. " * 320) +
              "The code word for this record is ORCHID.")
    q = "What is the code word for this record?"
    cold = completion(base_url, prompt(shared, q), False, n_predict=n_predict)
    completion(base_url, prompt(shared, "Is routine telemetry nominal? Answer YES."), True,
               n_predict=n_predict)
    warm = completion(base_url, prompt(shared, q), True, n_predict=n_predict)
    rows.append(compare_case("shared_prefix_divergent_suffix", cold, warm, "ORCHID"))

    shared2 = ((f"PARTIAL-{nonce}: archived line. " * 360) +
               "The recovery key is COBALT.")
    short = prompt(shared2, "What is the recovery key?")
    cold2 = completion(base_url, short, False, n_predict=n_predict)
    completion(base_url, shared2 + ("\nDisposable appendix." * 140) +
               "\nConfirm receipt with the single word RECEIVED.", True, n_predict=n_predict)
    warm2 = completion(base_url, short, True, n_predict=n_predict)
    rows.append(compare_case("partial_removal", cold2, warm2, "COBALT"))

    shared3 = ((f"CANCEL-{nonce}: stable ledger entry. " * 400) +
               "The post-cancel validation token is AMBER.")
    check3 = prompt(shared3, "What is the post-cancel validation token?")
    cold3 = completion(base_url, check3, False, n_predict=n_predict)
    cancelled = cancel_stream(base_url, shared3 +
                              "\nWrite an extremely long numbered account of every ledger entry.")
    warm3 = completion(base_url, check3, True, n_predict=n_predict)
    row3 = compare_case("cancel_then_reuse", cold3, warm3, "AMBER")
    row3["cancellation"] = cancelled
    row3["pass"] = row3["pass"] and cancelled["lines_received"] > 0 and cancelled["idle_after_close"]
    rows.append(row3)

    # Keep the nonce once. Repeating a high-entropy hex nonce in every filler line makes
    # token count explode and accidentally turns this into an over-context error probe.
    filler = "Ordinary archival material appears here without any code word. "
    shared4 = (f"LONG-{nonce}: archive begins. " + filler * 2250 +
               "The long-context code word is MAGNOLIA. " + filler * 250)
    check4 = prompt(shared4, "What is the long-context code word?")
    cold4 = completion(base_url, check4, False, n_predict=n_predict)
    completion(base_url, prompt(shared4, "Does this archive contain ordinary material? Answer YES."),
               True, n_predict=8)
    warm4 = completion(base_url, check4, True, n_predict=n_predict)
    rows.append(compare_case("long_context_reuse", cold4, warm4, "MAGNOLIA"))

    return {"campaign": "LAB-CACHE-001-v2", "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": base_url, "invariant": "cached output == cold output and both match oracle",
            "n_predict": n_predict, "nonce": nonce,
            "summary": {"passed": sum(bool(row["pass"]) for row in rows), "total": len(rows),
                        "all_pass": all(bool(row["pass"]) for row in rows)},
            "capabilities": server_capabilities(base_url), "results": rows}


def selfcheck() -> None:
    assert normalize(" `ORCHID`. ") == "orchid"
    base = {"content": "ORCHID", "timings": {}, "error": None, "raw": {}}
    cached = {"content": "ORCHID", "timings": {"cache_n": 100}, "error": None, "raw": {}}
    assert compare_case("x", base, cached, "ORCHID")["pass"]
    changed = dict(cached, content="ROSE")
    assert not compare_case("x", base, changed, "ORCHID")["pass"]
    no_hit = dict(cached, timings={"cache_n": 0})
    assert not compare_case("x", base, no_hit, "ORCHID")["pass"]
    print("cache correctness v2 self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("runs/cache/LAB-CACHE-001-v2/results.json"))
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--nonce", help="fixed prompt nonce for paired cross-arm comparisons")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    report = run(args.base_url, args.n_predict, args.nonce)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for row in report["results"]:
        print(f"{row['case']:<32} {'PASS' if row['pass'] else 'FAIL'} "
              f"cache_n={row['warm_cache_n']}")
    print(json.dumps(report["summary"], indent=2))
    print(f"evidence: {args.output}")
    return 0 if report["summary"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
