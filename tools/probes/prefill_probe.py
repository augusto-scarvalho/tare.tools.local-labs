#!/usr/bin/env python3
"""Prefill (prompt-processing) speed at long context. TTFT for a 128k prompt is prefill-bound
(~87s at ~1500 t/s), the real latency wall for long-context agentic use. Two levers:
  * --ubatch (-ub): tokens processed per forward pass -> GPU parallelism (costs VRAM).
  * Fable prefetch (GGML_SCHED_PREFETCH_EXPERTS + pinning): overlap the per-batch expert H2D on a
    2nd CUDA stream. Built FOR large-batch prefill (its gate needs >=64 tok; prefill has >=512).
Measures prompt_per_second on a ~fixed long prompt; reports est. 128k prefill time.

  python3 prefill_probe.py   (MoE, ncmoe=8, q4 KV)
"""
import json, os, subprocess, sys, time, urllib.request

MODEL = "/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
BIN = "/home/augus/src/slop.cpp-main/build/bin/llama-server"
PORT, NCMOE, KV, CTX = 8101, "8", "q4_0", 65536
PROMPT_WORDS = 36000  # ~48k tokens
PF = {"GGML_SCHED_PREFETCH_EXPERTS": "3", "GGML_CUDA_REGISTER_HOST": "1"}
ARMS = [
    ("ub512",      {}, ["-ub", "512",  "-b", "2048"]),
    ("ub1024",     {}, ["-ub", "1024", "-b", "2048"]),
    ("ub2048",     {}, ["-ub", "2048", "-b", "2048"]),
    ("ub512+pf",   PF, ["-ub", "512",  "-b", "2048"]),
    ("ub2048+pf",  PF, ["-ub", "2048", "-b", "2048"]),
]
PROMPT = (("The maintenance log records routine checks with nominal readings across all "
           "monitored subsystems this cycle and no anomalies were detected anywhere. ")
          * (PROMPT_WORDS // 22 + 2))


def wait(t=240):
    for _ in range(t):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


def prefill():
    body = json.dumps({"prompt": PROMPT, "n_predict": 1, "temperature": 0.0,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    t = json.loads(urllib.request.urlopen(req, timeout=600).read()).get("timings", {})
    return t.get("prompt_n", 0), t.get("prompt_per_second", 0)


print(f"== prefill_probe: MoE ncmoe={NCMOE} kv={KV} ==", flush=True)
print(f"{'arm':<12} {'prompt_n':>9} {'prefill_tps':>12} {'128k_prefill_s':>15}", flush=True)
results = []
for label, env, extra in ARMS:
    subprocess.run(["pkill", "-9", "-f", f"port {PORT}"], capture_output=True); time.sleep(2)
    e = dict(os.environ, **env)
    proc = subprocess.Popen([BIN, "-m", MODEL, "-fa", "on", "--n-cpu-moe", NCMOE,
                             "--cache-type-k", KV, "--cache-type-v", KV, "-c", str(CTX),
                             *extra, "--host", "127.0.0.1", "--port", str(PORT)],
                            env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait():
            print(f"{label:<12} server never healthy"); continue
        prefill()  # warm-up (discard)
        pn, tps = prefill()
        t128 = 131072 / tps if tps else 0
        results.append((label, pn, tps, t128))
        print(f"{label:<12} {pn:>9} {tps:>12.1f} {t128:>14.1f}s", flush=True)
    finally:
        proc.terminate()
        try: proc.wait(timeout=15)
        except Exception: proc.kill()
        time.sleep(2)

if results:
    base = results[0][2]
    print("\n== vs ub512 baseline ==")
    for label, pn, tps, t128 in results:
        print(f"  {label:<12} {tps:>8.1f} t/s  ({(tps/base-1)*100:+6.1f}%)  128k prefill {t128:.1f}s")
