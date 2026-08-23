#!/usr/bin/env python3
"""Quant sweep (speed/placement/headroom side). On a 24 GB card, bigger weight-quant = better
quality but must OFFLOAD more experts to CPU (higher ncmoe) to fit -> slower decode + less context
headroom. For each quant, find the MIN ncmoe that fits the 4 GB envelope, measure decode t/s there,
VRAM, and derive the context headroom. (Quality side = quality_bench, run separately.)"""
import json, subprocess, time, urllib.request

D = "/home/augus/models/qwen36-35b-a3b/"
QUANTS = [("Q4_K_M", D + "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
          ("Q5_K_M", D + "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"),
          ("Q6_K",   D + "Qwen3.6-35B-A3B-UD-Q6_K.gguf"),
          ("Q8_0",   D + "Qwen3.6-35B-A3B-Q8_0.gguf")]
BIN = "/home/augus/src/slop.cpp-main/build/bin/llama-server"
PORT = 8103
NCMOE_TRY = [8, 16, 24, 32, 40]


def wait(t=200):
    for _ in range(t):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


def decode():
    body = json.dumps({"prompt": "Explain memory bandwidth limits in one paragraph.",
                       "n_predict": 100, "temperature": 0.0, "cache_prompt": False,
                       "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    t = json.loads(urllib.request.urlopen(req, timeout=300).read()).get("timings", {})
    return t.get("predicted_per_second", 0)


print(f"{'quant':<8} {'min-fit ncmoe':>13} {'decode':>8} {'vram_used':>10} {'vram_free':>10} "
      f"{'ctx_headroom_q8':>16}", flush=True)
for qn, path in QUANTS:
    done = False
    for nc in NCMOE_TRY:
        subprocess.run(["pkill", "-9", "-f", f"port {PORT}"], capture_output=True); time.sleep(2)
        proc = subprocess.Popen([BIN, "-m", path, "-fa", "on", "--n-cpu-moe", str(nc),
                                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "-c", "8192",
                                 "-ub", "2048", "-b", "2048", "--host", "127.0.0.1",
                                 "--port", str(PORT)], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        try:
            if not wait():
                proc.terminate(); proc.wait(timeout=10); continue
            used, free = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.free",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True
            ).stdout.split("\n")[0].split(",")
            used, free = int(used), int(free)
            if free < 4096:  # doesn't fit the envelope at this ncmoe -> try higher
                continue
            decode()  # warm
            dt = decode()
            # context headroom: usable KV budget / ~13 MiB per 1k (q8, from Phase A)
            hdr = (free - 4096) / 13.0  # in "k tokens" of extra ctx beyond 8k
            print(f"{qn:<8} {nc:>13} {dt:>7.1f}t {used:>9}M {free:>9}M {8 + max(0,hdr):>13.0f}k",
                  flush=True)
            done = True
            break
        finally:
            proc.terminate()
            try: proc.wait(timeout=15)
            except Exception: proc.kill()
            time.sleep(2)
    if not done:
        print(f"{qn:<8} did not fit even at ncmoe=40", flush=True)
