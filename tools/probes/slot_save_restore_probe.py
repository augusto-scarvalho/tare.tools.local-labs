#!/usr/bin/env python3
"""LAB-CACHE-001 explicit slot file save/erase/restore correctness probe.

Requires llama-server launched with ``--slot-save-path``.  The saved recurrent/KV state
must survive erase+restore, produce the same greedy known answer, and expose a cache hit.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone


def post(url: str, payload: dict, timeout_s: float = 600.0) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return {"_error": f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:1000]}"}


def action(base_url: str, slot: int, name: str, filename: str | None = None) -> dict:
    payload = {"filename": filename} if filename is not None else {}
    return post(f"{base_url.rstrip('/')}/slots/{slot}?action={name}", payload)


def complete(base_url: str, slot: int, prompt: str) -> dict:
    response = post(f"{base_url.rstrip('/')}/completion", {
        "prompt": prompt, "n_predict": 64, "temperature": 0.0, "top_k": 1,
        "seed": 0, "cache_prompt": True, "id_slot": slot, "stream": False})
    return {"content": response.get("content", ""), "timings": response.get("timings") or {},
            "error": response.get("_error"), "raw": response}


def normalize(value: str) -> str:
    return " ".join(value.strip().lower().split()).strip(" .,!?:;`\"'")


def selfcheck() -> None:
    assert normalize(" `SAFFRON`. ") == "saffron"
    print("slot save/restore probe self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("runs/cache/LAB-CACHE-001-slot-save/results.json"))
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    filename = f"lab-cache-{uuid.uuid4().hex}.bin"
    nonce = uuid.uuid4().hex
    shared = ((f"SLOT-{nonce}: routine state remains stable. " * 500) +
              "The persistent code word is SAFFRON.")
    prompt = shared + ("\nWhat is the persistent code word? Reply with ONLY the exact code word, "
                       "with no explanation or punctuation.")
    erased_before = action(args.base_url, args.slot, "erase")
    cold = complete(args.base_url, args.slot, prompt)
    saved = action(args.base_url, args.slot, "save", filename)
    erased = action(args.base_url, args.slot, "erase")
    restored = action(args.base_url, args.slot, "restore", filename)
    warm = complete(args.base_url, args.slot, prompt)
    erased_after = action(args.base_url, args.slot, "erase")
    expected = "saffron"
    exact = cold["content"] == warm["content"]
    oracle = expected in normalize(cold["content"]) and expected in normalize(warm["content"])
    cache_n = int(warm["timings"].get("cache_n") or 0)
    lifecycle_ok = (not saved.get("_error") and not erased.get("_error")
                    and not restored.get("_error") and int(saved.get("n_saved") or 0) > 0
                    and int(restored.get("n_restored") or 0) > 0)
    report = {"campaign": "LAB-CACHE-001-slot-save", "timestamp": datetime.now(timezone.utc).isoformat(),
              "endpoint": args.base_url, "slot": args.slot, "filename": filename,
              "pass": exact and oracle and cache_n > 0 and lifecycle_ok,
              "exact_cold_restored": exact, "oracle_pass": oracle,
              "restored_cache_n": cache_n, "lifecycle_ok": lifecycle_ok,
              "erased_before": erased_before, "cold": cold, "saved": saved,
              "erased": erased, "restored": restored, "warm": warm,
              "erased_after": erased_after,
              "note": "saved file remains under server slot-save-path for evidence/cleanup"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "pass", "exact_cold_restored", "oracle_pass", "restored_cache_n", "lifecycle_ok")}, indent=2))
    print(f"evidence: {args.output}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
