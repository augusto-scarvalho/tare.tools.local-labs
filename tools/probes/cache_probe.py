#!/usr/bin/env python3
"""Prompt-cache reuse: the biggest agentic-latency win. Prefill a long shared prefix ONCE, then
each follow-up query that shares it skips re-prefilling (KV reused in the slot). Measures query 1
(cold, full prefill) vs query 2 (same long prefix, new question -> only the new tokens prefill).
"""
import json, subprocess, time, urllib.request

MODEL = "/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
BIN = "/home/augus/src/slop.cpp-main/build/bin/llama-server"
PORT = 8101
PREFIX = (("The maintenance log records routine checks with nominal readings across all monitored "
           "subsystems this cycle and no anomalies were detected anywhere. ") * 1800)  # ~40k tok


def wait(t=240):
    for _ in range(t):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


def ask(text):
    body = json.dumps({"prompt": text, "n_predict": 8, "temperature": 0.0,
                       "cache_prompt": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    t = json.loads(urllib.request.urlopen(req, timeout=600).read()).get("timings", {})
    return t


proc = subprocess.Popen([BIN, "-m", MODEL, "-fa", "on", "--n-cpu-moe", "8",
                         "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "-c", "65536",
                         "-ub", "2048", "-b", "2048", "--host", "127.0.0.1", "--port", str(PORT)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    if not wait():
        print("server never healthy"); raise SystemExit(1)
    print("== prompt-cache reuse (shared long prefix, ub2048) ==")
    t1 = ask(PREFIX + "\n\nQuestion 1: summarize the log in one word.")
    print(f"query 1 (cold): prompt_n={t1.get('prompt_n')} cache_n={t1.get('cache_n',0)} "
          f"prompt_ms={t1.get('prompt_ms',0):.0f}  ({t1.get('prompt_ms',0)/1000:.1f}s TTFT)")
    t2 = ask(PREFIX + "\n\nQuestion 2: is there any anomaly? answer yes or no.")
    print(f"query 2 (reuse): prompt_n={t2.get('prompt_n')} cache_n={t2.get('cache_n',0)} "
          f"prompt_ms={t2.get('prompt_ms',0):.0f}  ({t2.get('prompt_ms',0)/1000:.2f}s TTFT)")
    r = (t1.get('prompt_ms', 1) or 1) / (t2.get('prompt_ms', 1) or 1)
    print(f"\n-> follow-up TTFT is {r:.0f}x faster: the {t2.get('cache_n',0)}-token prefix was "
          f"REUSED, only {t2.get('prompt_n',0)} new tokens prefilled.")
finally:
    proc.terminate()
    try: proc.wait(timeout=15)
    except Exception: proc.kill()
