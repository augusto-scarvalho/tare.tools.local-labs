#!/usr/bin/env python3
"""CONTEXT_PLAN Phase C — the HARD probe: multi-hop / aggregation over long context.

Single-needle NIAH saturates (100%) and hides the real limit. This scatters N distinct facts
("the access code for facility <X> is <NNNN>") EVENLY across ~L tokens, then asks an aggregation
question that requires attending to ALL of them: "what is the LARGEST access code?". Score = the
true max appears in the answer. As L grows the model must track N facts over more context -> this
is what degrades (RULER-style), revealing the USABLE ceiling per KV-format.

  python3 multihop_probe.py <moe|dense> <ncmoe> <kv> <ctx> <L...>
"""
import json, subprocess, sys, time, urllib.request

MODELS = {"moe": "/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
          "dense": "/home/augus/models/qwen36-27b-mtp/Qwen3.6-27B-Q4_K_M.gguf"}
BIN = "/home/augus/src/llama.cpp-master/build/bin/llama-server"
MK, NC, KV, CTX = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
LS = [int(x) for x in sys.argv[5:]] or [8000, 32000, 65000, 128000, 175000]
PLACE = ["--n-cpu-moe", NC] if MK == "moe" else ["-ngl", "99"]
PORT, NFACTS, TRIALS = 8100, 8, 3
NAMES = ["Aster", "Brill", "Cove", "Dune", "Ember", "Frost", "Gale", "Haven",
         "Iris", "Jade", "Kite", "Lyra", "Mesa", "Nova", "Onyx", "Pike"]
FILLER = ("The quarterly maintenance log for the north wing records routine checks with no "
          "anomalies and nominal readings across all monitored subsystems this cycle. ")


def build(L_tok, trial):
    # deterministic-per-(L,trial) codes/names, evenly scattered through the filler
    base = 1000 + (L_tok // 1000 + trial * 97) % 8000
    facts = [(NAMES[(trial * 3 + i) % len(NAMES)], base + i * 137 + trial * 11)
             for i in range(NFACTS)]
    codes = [c for _, c in facts]
    n_words = int(L_tok / 1.35)
    pool = (FILLER * (n_words // 24 + 2)).split()[:n_words]
    step = max(1, len(pool) // (NFACTS + 1))
    for i, (nm, c) in enumerate(facts):
        ins = f"IMPORTANT: the access code for facility {nm} is {c}."
        p = min(len(pool), (i + 1) * step)
        pool[p:p] = ins.split()
    prompt = (" ".join(pool) + "\n\nQuestion: among ALL facilities mentioned above, what is the "
              "LARGEST access code? Answer with only that number.")
    return prompt, max(codes)


def wait(t=700):
    for _ in range(t):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


def ask(prompt):
    body = json.dumps({"prompt": prompt, "n_predict": 16, "temperature": 0.0, "top_k": 1,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return d.get("content", ""), d.get("timings", {})


print(f"== multihop: {MK} {PLACE} kv={KV} ctx={CTX} N={NFACTS} facts, aggregation(max) ==", flush=True)
proc = subprocess.Popen([BIN, "-m", MODELS[MK], "-fa", "on", *PLACE, "--cache-type-k", KV,
                         "--cache-type-v", KV, "-c", str(CTX), "--host", "127.0.0.1",
                         "--port", str(PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
summary = []
try:
    if not wait():
        print("server never healthy"); sys.exit(1)
    for L in LS:
        ok = 0; pn = 0; dtps = []
        for tr in range(TRIALS):
            prompt, ans = build(L, tr)
            content, t = ask(prompt)
            hit = str(ans) in content
            ok += hit; pn = t.get("prompt_n", pn); dtps.append(t.get("predicted_per_second", 0))
            print(f"  L~{L:>7} pn={t.get('prompt_n',0):>7} trial{tr} exp={ans} "
                  f"{'HIT' if hit else 'MISS:'+content.strip()[:24]!r}", flush=True)
        summary.append((L, pn, ok, TRIALS, sum(dtps) / len(dtps)))
finally:
    proc.terminate()
    try: proc.wait(timeout=15)
    except Exception: proc.kill()

print("\n== summary (multi-hop aggregation accuracy vs context) ==")
for L, pn, ok, n, dt in summary:
    print(f"L~{L:>7} (prompt_n {pn:>7}): {ok}/{n} correct  decode {dt:>6.2f} t/s")
