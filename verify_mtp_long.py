#!/usr/bin/env python3
"""Characterise WHERE draft-mtp diverges from base over a LONG (thinking) generation.

§E4/gate G2 verified token-identity on SHORT (256-tok) output. The levers x quality matrix
showed base != mtp on 24/40 long thinking generations (base itself is deterministic: base==base2,
40/40). This pins the onset: base and mtp share an identical PREFIX, then diverge at some token
position D (batched-verify numerics flip a greedy near-tie; likely #23658 KV-slot boundary).
Reports D (char + approx token) for a few coding prompts on the deploy config.
"""
import json, os, subprocess, sys, time, urllib.request

BIN = os.environ.get("MTP_BIN", "/home/augus/src/llama.cpp-master/build/bin/llama-server")
MODEL = "/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
PORT = 8098
COMMON = ["-m", MODEL, "--host", "127.0.0.1", "--port", str(PORT), "-fa", "on",
          "--n-cpu-moe", "8", "--ctx-size", "8192", "--jinja",
          "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
MODES = {"base": COMMON,
         "mtp":  COMMON + ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]}
PROMPTS = [
    "Complete this Python function. Reply with the complete function in one ```python block:\n\n"
    "def sort_third(l: list):\n    \"\"\"Return a list where every third element is sorted.\"\"\"",
    "Complete this Python function. Reply with the complete function in one ```python block:\n\n"
    "def decode_cyclic(s: str):\n    \"\"\"Takes a string encoded by cyclic groups of three and decodes it.\"\"\"",
]


def wait(timeout=180):
    for _ in range(timeout):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


def gen(prompt, npred=3000):
    body = json.dumps({"prompt": prompt, "n_predict": npred, "temperature": 0.0, "top_k": 1,
                       "cache_prompt": False, "seed": 42}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return d.get("content", ""), d.get("timings", {}).get("predicted_n", 0)


out = {}
for mode, args in MODES.items():
    p = subprocess.Popen([BIN] + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait():
            print(f"{mode}: never healthy"); p.kill(); continue
        out[mode] = [gen(pr) for pr in PROMPTS]
    finally:
        p.terminate()
        try: p.wait(timeout=15)
        except Exception: p.kill()
        time.sleep(3)

print("\n===== draft-mtp vs base, LONG generation =====")
for i in range(len(PROMPTS)):
    (ba, bn), (ma, mn) = out["base"][i], out["mtp"][i]
    if ba == ma:
        print(f"prompt {i}: IDENTICAL ({len(ba)} chars, ~{bn} tok)")
        continue
    d = next((j for j in range(min(len(ba), len(ma))) if ba[j] != ma[j]), min(len(ba), len(ma)))
    frac = d / max(1, len(ba))
    print(f"prompt {i}: DIVERGE  base={len(ba)}ch(~{bn}tok) mtp={len(ma)}ch(~{mn}tok)  "
          f"first-diff @char {d} ({frac:.0%} through base)")
    print(f"    base: ...{ba[max(0,d-30):d+30]!r}")
    print(f"    mtp : ...{ma[max(0,d-30):d+30]!r}")
