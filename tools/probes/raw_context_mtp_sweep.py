#!/usr/bin/env python3
"""Localize raw ``/completion`` correctness by prompt shape and context depth.

This is deliberately separate from the chat context suite: it sends an untemplated
prompt, calibrates that exact string with the live server tokenizer, and retains the
full response plus speculative timing counters for audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.request
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.benchmarks.qwen38_requal import niah_units  # noqa: E402


def post(base_url: str, path: str, payload: dict, timeout: float = 900.0) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def token_count(base_url: str, content: str) -> int:
    result = post(base_url, "/tokenize", {"content": content, "add_special": False}, 60)
    return len(result.get("tokens") or [])


def assemble(shape: str, units: list[str]) -> str:
    fact = "AUTHORITATIVE FACT: the archival verification code is MAGNOLIA."
    question = ("What is the archival verification code? Return only the exact code word, "
                "with no explanation or punctuation.\nAnswer:")
    position = int(round(0.9 * len(units)))
    material = units[:position] + [fact] + units[position:]
    return f"RAW CONTEXT SHAPE: {shape}\n" + "\n".join(material) + "\n\n" + question


def unit_pool(shape: str, seed: int, target: int) -> list[str]:
    if shape == "repetitive":
        return ["Ordinary archival material appears here without any code word."] * max(2048, target)
    if shape == "diverse":
        return niah_units(seed, max(2048, target // 3))
    raise ValueError(shape)


def calibrate(base_url: str, target: int, shape: str, seed: int) -> tuple[str, int]:
    units = unit_pool(shape, seed, target)
    low, high, best, best_n = 0, len(units), "", 0
    while low <= high:
        middle = (low + high) // 2
        candidate = assemble(shape, units[:middle])
        count = token_count(base_url, candidate)
        if count <= target:
            best, best_n = candidate, count
            low = middle + 1
        else:
            high = middle - 1
    if not best:
        raise RuntimeError(f"could not calibrate {shape} at {target}")
    return best, best_n


def normalize(value: str) -> str:
    return " ".join(value.strip().lower().replace("`", "").split()).strip(" .,!?:;\"'")


def selfcheck() -> None:
    assert "MAGNOLIA" in assemble("repetitive", ["filler"] * 10)
    assert normalize(" `MAGNOLIA`. ") == "magnolia"
    assert len(unit_pool("diverse", 7, 100)) >= 100
    print("raw context MTP sweep self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8092")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=[8000, 16000, 20000, 24000, 32000])
    parser.add_argument("--shapes", nargs="+", choices=("repetitive", "diverse"),
                        default=["repetitive", "diverse"])
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0

    rows = []
    for target in args.targets:
        for shape_index, shape in enumerate(args.shapes):
            prompt, actual = calibrate(args.base_url, target, shape, args.seed + shape_index)
            started = time.monotonic()
            raw = post(args.base_url, "/completion", {
                "prompt": prompt, "n_predict": args.n_predict, "temperature": 0.0,
                "top_k": 1, "seed": 0, "cache_prompt": False, "stream": False,
            }, args.timeout)
            wall_s = time.monotonic() - started
            content = raw.get("content", "")
            timings = raw.get("timings") or {}
            passed = "magnolia" in normalize(content)
            row = {"arm": args.arm, "target_tokens": target, "actual_tokens": actual,
                   "shape": shape, "pass": passed, "expected": "MAGNOLIA",
                   "response": content, "finish_reason": raw.get("stop_type"),
                   "timings": timings, "wall_s": wall_s,
                   "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "raw": raw}
            rows.append(row)
            print(f"{args.arm:<8} t={target:<6} actual={actual:<6} {shape:<10} "
                  f"{'PASS' if passed else 'FAIL'} draft="
                  f"{timings.get('draft_n')}/{timings.get('draft_n_accepted')}", flush=True)

    report = {"campaign": "LAB-CACHE-001-raw-context-mtp", "arm": args.arm,
              "timestamp": datetime.now(timezone.utc).isoformat(), "endpoint": args.base_url,
              "targets": args.targets, "shapes": args.shapes, "n_predict": args.n_predict,
              "summary": {"passed": sum(row["pass"] for row in rows), "total": len(rows),
                          "all_pass": all(row["pass"] for row in rows)}, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"evidence: {args.output}")
    return 0 if report["summary"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
