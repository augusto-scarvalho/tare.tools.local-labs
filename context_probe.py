#!/usr/bin/env python3
"""CONTEXT_PLAN Phases B+C in one harness (share the expensive deep prefill).

For each context depth L and needle depth d: build a needle-in-haystack prompt of ~L tokens
(filler + a unique passcode at fraction d), ask the model to retrieve it, and record BOTH:
  * Phase C: retrieved? (usable context / effective-vs-advertised)
  * Phase B: prefill_tps and decode_tps AT depth L (speed at depth)
temp=0 (deterministic). One server launch per (model,kv); loops L x d against it.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 -- /home/augus/evalplus-venv/bin/python \
    /mnt/c/projects/local-model-lifecycle/context_probe.py <moe|dense> <ncmoe> <kv> <ctx> <L...>
"""
import json, subprocess, sys, time, urllib.request

MODELS = {"moe": "/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
          "dense": "/home/augus/models/qwen36-27b-mtp/Qwen3.6-27B-Q4_K_M.gguf"}
BIN = "/home/augus/src/llama.cpp-master/build/bin/llama-server"
MK, NC, KV, CTX = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
LS = [int(x) for x in sys.argv[5:]] or [8000, 32000, 65000, 131000]
DS = [0.25, 0.75]
PORT = 8100
PLACE = ["--n-cpu-moe", NC] if MK == "moe" else ["-ngl", "99"]

FILLER = ("The quarterly maintenance log for sector {i} records routine checks with no anomalies "
          "and nominal readings across all subsystems. ")  # ~22 tokens/unit


def niah(L_tok, depth, code):
    n_words = int(L_tok / 1.35)
    pool = (FILLER.replace("{i}", "7") * (n_words // 22 + 2)).split()
    words = pool[:n_words]
    needle = f"REMEMBER: the special access code for the Grover facility is {code}.".split()
    pos = max(1, min(len(words) - 1, int(len(words) * depth)))
    words = words[:pos] + needle + words[pos:]
    return (" ".join(words) +
            "\n\nQuestion: what is the special access code for the Grover facility? "
            "Answer with only the number.")


def wait(t=600):
    for _ in range(t):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


def ask(prompt):
    body = json.dumps({"prompt": prompt, "n_predict": 24, "temperature": 0.0, "top_k": 1,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=1200).read())
    t = d.get("timings", {})
    return d.get("content", ""), t


print(f"== context_probe: {MK} placement={PLACE} kv={KV} ctx={CTX} ==", flush=True)
proc = subprocess.Popen([BIN, "-m", MODELS[MK], "-fa", "on", *PLACE,
                         "--cache-type-k", KV, "--cache-type-v", KV, "-c", str(CTX),
                         "--host", "127.0.0.1", "--port", str(PORT)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
rows = []
try:
    if not wait():
        print("server never healthy"); sys.exit(1)
    print(f"{'L_target':>9} {'prompt_n':>9} {'d':>5} {'found':>6} "
          f"{'prefill_tps':>11} {'decode_tps':>10}", flush=True)
    for L in LS:
        for d in DS:
            code = 40000 + (L // 1000) * 7 + int(d * 100) * 13
            content, t = ask(niah(L, d, code))
            found = str(code) in content
            pn = t.get("prompt_n", 0)
            ptps = t.get("prompt_per_second", 0)
            dtps = t.get("predicted_per_second", 0)
            rows.append((L, pn, d, found, ptps, dtps))
            print(f"{L:>9} {pn:>9} {d:>5} {str(found):>6} {ptps:>11.1f} {dtps:>10.2f}", flush=True)
finally:
    proc.terminate()
    try: proc.wait(timeout=15)
    except Exception: proc.kill()

print("\n== summary ==")
for L in LS:
    r = [x for x in rows if x[0] == L]
    if not r:
        continue
    acc = sum(1 for x in r if x[3]) / len(r)
    dt = sum(x[5] for x in r) / len(r)
    pn = r[0][1]
    print(f"L~{L:>7} (prompt_n {pn:>7}): NIAH {acc*100:>5.0f}%  decode {dt:>6.2f} t/s")
