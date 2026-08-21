#!/usr/bin/env python3
"""Concurrency / multi-slot throughput. A real deploy serves N concurrent requests; llama.cpp
BATCHES their decode (reads the weights once per batch), so aggregate t/s should scale sub-linearly
up to compute/bandwidth saturation while per-stream t/s drops. Measures aggregate vs per-stream vs
VRAM across N parallel slots. MoE, ncmoe=8, q4 KV, ub2048; each stream decodes NPRED tokens.
  python3 concurrency_probe.py [mtp]
"""
import json, subprocess, sys, time, urllib.request, concurrent.futures

MODEL = "/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
BIN = "/home/augus/src/slop.cpp-main/build/bin/llama-server"
PORT, NC, KV, NPRED = 8102, "8", "q4_0", 200
NS = [1, 2, 4, 8]
MTP = len(sys.argv) > 1 and sys.argv[1] == "mtp"
SPEC = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"] if MTP else []
TOPICS = ["memory bandwidth", "mixture of experts routing", "flash attention", "KV cache growth",
          "PCIe transfers", "quantization tradeoffs", "speculative decoding", "context length"]


def one(i):
    body = json.dumps({"prompt": f"[stream {i}] Explain in detail how {TOPICS[i % 8]} affects "
                       "local LLM inference on a consumer GPU. Be thorough.",
                       "n_predict": NPRED, "temperature": 0.0, "cache_prompt": False,
                       "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    t = d.get("timings", {})
    return t.get("predicted_n", 0), t.get("predicted_per_second", 0)


def wait(t=240):
    for _ in range(t):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


print(f"== concurrency: MoE ncmoe={NC} kv={KV} ub2048 MTP={'on' if MTP else 'off'} ==", flush=True)
print(f"{'N':>3} {'agg_tps':>9} {'per_stream':>11} {'scaling':>8} {'vram_MiB':>9}", flush=True)
rows = []
for N in NS:
    subprocess.run(["pkill", "-9", "-f", f"port {PORT}"], capture_output=True); time.sleep(2)
    proc = subprocess.Popen([BIN, "-m", MODEL, "-fa", "on", "--n-cpu-moe", NC, "--cache-type-k", KV,
                             "--cache-type-v", KV, "-c", str(N * 4096), "-np", str(N),
                             "-ub", "2048", "-b", "2048", *SPEC,
                             "--host", "127.0.0.1", "--port", str(PORT)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait():
            print(f"{N:>3}  server never healthy"); continue
        one(0)  # warm-up
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
            res = list(ex.map(one, range(N)))
        wall = time.monotonic() - t0
        toks = sum(r[0] for r in res)
        agg = toks / wall
        per = sum(r[1] for r in res) / len(res)
        vram = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                               "--format=csv,noheader,nounits"], capture_output=True,
                              text=True).stdout.split("\n")[0].strip()
        scal = agg / rows[0][1] if rows else 1.0
        rows.append((N, agg, per, vram))
        print(f"{N:>3} {agg:>9.1f} {per:>11.2f} {scal:>7.2f}x {vram:>9}", flush=True)
    finally:
        proc.terminate()
        try: proc.wait(timeout=15)
        except Exception: proc.kill()
        time.sleep(2)

if rows:
    print("\n== throughput scaling ==")
    b = rows[0][1]
    for N, agg, per, vram in rows:
        print(f"  N={N}: aggregate {agg:>7.1f} t/s ({agg/b:.2f}x)  per-stream {per:>6.1f} t/s  "
              f"vram {vram} MiB")
