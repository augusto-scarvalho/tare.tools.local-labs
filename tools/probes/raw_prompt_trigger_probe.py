#!/usr/bin/env python3
"""Dissect the raw-prompt trigger observed in LAB-CACHE-001's long case."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
import urllib.request
from datetime import datetime, timezone


def post(base_url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/completion", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def build_prompt(nonce: str, answer_cue: bool) -> str:
    filler = "Ordinary archival material appears here without any code word. "
    shared = (f"LONG-{nonce}: archive begins. " + filler * 2250
              + "The long-context code word is MAGNOLIA. " + filler * 250)
    prompt = (shared + "\n\nWhat is the long-context code word?"
              + "\nReturn only the requested code word, with no explanation or punctuation.")
    return prompt + ("\nAnswer:" if answer_cue else "")


def normalize(value: str) -> str:
    return " ".join(value.strip().lower().replace("`", "").split()).strip(" .,!?:;\"'")


def selfcheck() -> None:
    a = build_prompt("fixed", False)
    b = build_prompt("fixed", True)
    assert b == a + "\nAnswer:"
    assert a.count("MAGNOLIA") == 1
    print("raw prompt trigger probe self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8092")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--nonce", default="mtp-rollback-20260822-a")
    parser.add_argument("--budgets", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0

    rows = []
    for budget in args.budgets:
        for variant, cue in (("original", False), ("answer_cue", True)):
            prompt = build_prompt(args.nonce, cue)
            started = time.monotonic()
            raw = post(args.base_url, {"prompt": prompt, "n_predict": budget,
                       "temperature": 0.0, "top_k": 1, "seed": 0,
                       "cache_prompt": False, "stream": False}, args.timeout)
            wall_s = time.monotonic() - started
            content = raw.get("content", "")
            timings = raw.get("timings") or {}
            passed = "magnolia" in normalize(content)
            rows.append({"arm": args.arm, "variant": variant, "n_predict": budget,
                         "pass": passed, "response": content, "timings": timings,
                         "wall_s": wall_s,
                         "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                         "raw": raw})
            print(f"{args.arm:<8} budget={budget:<3} {variant:<10} "
                  f"{'PASS' if passed else 'FAIL'} prompt_n={timings.get('prompt_n')} "
                  f"draft={timings.get('draft_n')}/{timings.get('draft_n_accepted')}", flush=True)
    report = {"campaign": "LAB-CACHE-001-raw-trigger", "arm": args.arm,
              "timestamp": datetime.now(timezone.utc).isoformat(), "endpoint": args.base_url,
              "nonce": args.nonce, "budgets": args.budgets,
              "summary": {"passed": sum(row["pass"] for row in rows), "total": len(rows),
                          "all_pass": all(row["pass"] for row in rows)}, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"evidence: {args.output}")
    return 0 if report["summary"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
