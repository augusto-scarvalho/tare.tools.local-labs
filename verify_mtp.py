#!/usr/bin/env python3
"""Greedy verification of MTP spec-decode: token-identity + accept rate + timings.

Runs inside WSL. For each mode (base = no spec, mtp = draft-mtp), launches the
master llama-server on the SAME MTP GGUF, sends ONE greedy (temp=0) completion, and
records the emitted text + server timings. Spec-decode is exact, so base and mtp must
emit byte-identical text; the timings expose the draft accept rate and the real t/s.
"""
import json
import subprocess
import time
import urllib.request

BIN = "/home/augus/src/llama.cpp-master/build/bin/llama-server"
MODEL = "/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
PORT = 8095
PROMPT = ("Write a Python function `fib(n)` that returns the nth Fibonacci number "
          "using iteration, then list three properties of the Fibonacci sequence.")

COMMON = ["-m", MODEL, "--host", "127.0.0.1", "--port", str(PORT), "-fa", "on",
          "--n-cpu-moe", "8", "--ctx-size", "8192"]
MODES = {
    "base": COMMON,
    "mtp":  COMMON + ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"],
}


def wait_ready(timeout=90):
    for _ in range(timeout):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def complete():
    body = json.dumps({
        "prompt": PROMPT, "n_predict": 256, "temperature": 0.0, "top_k": 1,
        "cache_prompt": False, "seed": 42, "timings_per_token": True,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


results = {}
for mode, args in MODES.items():
    print(f"=== launching {mode} ===", flush=True)
    proc = subprocess.Popen([BIN] + args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        if not wait_ready():
            print(f"{mode}: server never became ready"); proc.kill(); continue
        resp = complete()
        results[mode] = {"content": resp.get("content", ""),
                         "timings": resp.get("timings", {})}
        print(f"{mode}: got {len(results[mode]['content'])} chars", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        time.sleep(3)

print("\n===== TIMINGS =====")
for mode, r in results.items():
    t = r["timings"]
    tps = t.get("predicted_per_second")
    print(f"\n[{mode}] predicted_per_second = {tps}")
    # Surface every draft/accept-related field the build exposes.
    for k, v in sorted(t.items()):
        if any(x in k.lower() for x in ("draft", "accept", "predicted_n", "spec")):
            print(f"    {k} = {v}")

if {"base", "mtp"} <= set(results):
    a, b = results["base"]["content"], results["mtp"]["content"]
    print("\n===== TOKEN IDENTITY =====")
    print(f"base chars={len(a)}  mtp chars={len(b)}  IDENTICAL={a == b}")
    if a != b:
        # First divergence, for diagnosis.
        i = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        print(f"  first diff at char {i}:")
        print(f"    base: ...{a[max(0,i-40):i+40]!r}")
        print(f"    mtp : ...{b[max(0,i-40):i+40]!r}")
