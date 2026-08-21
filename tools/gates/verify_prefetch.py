"""Prove the fork's feature is ON before spending hours measuring it.

Two gates must both open: the env var (else prefetch_experts stays false) and the batch
size (ids count >= 2*n_expert, i.e. >=64 tokens for this model). Neither is visible on
the command line, which is how three A/B runs measured a disabled feature.
"""
import json, subprocess, sys, time, urllib.request
sys.path.insert(0, r"C:\projects\tare.tools.local-labs\src")

BIN = "/home/augus/src/slop.cpp/build/bin/llama-server"
MODEL = "/home/augus/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
from ab_compare import PROMPT  # the long prompt, so this tests the real thing

argv = ["wsl", "-d", "Ubuntu-24.04", "--", "env", "GGML_SCHED_PREFETCH_EXPERTS=3",
        BIN, "-m", MODEL, "--host", "127.0.0.1", "--port", "8097",
        "-ncmoe", "24", "-c", "8192", "-ctk", "q8_0", "-ctv", "q8_0"]
p = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
lines = []
try:
    for _ in range(180):
        try:
            urllib.request.urlopen("http://127.0.0.1:8097/health", timeout=3).read()
            break
        except Exception:
            time.sleep(2)
    else:
        print("server never healthy"); sys.exit(1)

    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": 8, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request("http://127.0.0.1:8097/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    timings = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if ln.startswith("data:") and ln[5:].strip() not in ("[DONE]", ""):
                try:
                    ev = json.loads(ln[5:].strip())
                except ValueError:
                    continue
                if "timings" in ev:
                    timings = ev["timings"]
    n_expert, n_used = 256, 8
    need = 2 * n_expert // n_used
    got = timings.get("prompt_n") if timings else None
    print(f"prompt_n            : {got}")
    print(f"batch gate needs    : >= {need} tokens")
    print(f"BATCH GATE OPEN     : {bool(got and got >= need)}")
finally:
    p.kill()
    try:
        out = p.stderr.read()
    except Exception:
        out = ""
    subprocess.run(["wsl", "-d", "Ubuntu-24.04", "--", "pkill", "-9", "-f", "llama-server"])
    pinned = [l for l in out.splitlines() if "pinned" in l.lower()]
    print(f"PINNING ACTIVE      : {bool(pinned)}")
    for l in pinned[:2]:
        print("   ", l.strip())
