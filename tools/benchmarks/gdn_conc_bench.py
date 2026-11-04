#!/usr/bin/env python3
"""GDN concurrent-serving A/B driver (M4b) — measures whether the chunk-parallel GDN
prefill kernel's win surfaces under co-batched multi-slot serving.

Fires K simultaneous /completion requests (gated on a Barrier) at an already-running
llama-server, with UNIQUE random token-id prompts (defeats prompt cache), n_predict=1
(measure prefill, not decode). Primary signal = aggregate prefill throughput =
sum(prompt_n) / burst_wall_time. Compare ON vs OFF per K.

Run the server headless first (see GDN_M4_RESUME.md for the exact ON/OFF launch lines),
then from Windows:  python gdn_conc_bench.py <K> <PROMPT_LEN>  > runs/gdn/armB__on__k8.json

Stdlib only. Same random seed => identical prompt sets across ON and OFF for a given (K, rep).
"""
import json, random, threading, time, urllib.request, statistics, sys

URL   = "http://127.0.0.1:8080/completion"
K     = int(sys.argv[1]) if len(sys.argv) > 1 else 8
L     = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
REPS  = int(sys.argv[3]) if len(sys.argv) > 3 else 12   # first 2 = warmup, discarded
PAUSE = 10  # seconds between bursts (thermal / clock settle)

random.seed(1234)  # deterministic prompt sets => identical across ON/OFF


def burst(k, plen, rep):
    prompts = [[random.randrange(1000, 100000) for _ in range(plen)] for _ in range(k)]
    barrier = threading.Barrier(k)
    out = [None] * k

    def fire(i):
        body = json.dumps({
            "prompt": prompts[i], "n_predict": 1, "stream": False,
            "cache_prompt": False, "temperature": 0, "seed": 42,
        }).encode()
        req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
        barrier.wait()  # sub-ms simultaneity so all k occupy slots 0..k-1 at once
        out[i] = json.load(urllib.request.urlopen(req, timeout=600))

    ths = [threading.Thread(target=fire, args=(i,)) for i in range(k)]
    t0 = time.perf_counter()
    [t.start() for t in ths]
    [t.join() for t in ths]
    wall = time.perf_counter() - t0
    tm = [r["timings"] for r in out]
    # cache_n must be 0 on every request or the measurement is invalid (prompt-cache hit)
    assert all(t.get("cache_n", 0) == 0 for t in tm), "CACHE HIT -> measurement invalid"
    tot = sum(t["prompt_n"] for t in tm)
    return {
        "rep": rep, "k": k, "plen": plen, "wall_s": wall,
        "agg_tps": tot / wall,
        "prompt_ms": sorted(t["prompt_ms"] for t in tm),
    }


def main():
    res = []
    for r in range(REPS):
        res.append(burst(K, L, r))
        if r < REPS - 1:
            time.sleep(PAUSE)
    meas = res[2:]  # drop warmups
    tps = [x["agg_tps"] for x in meas]
    all_pms = sorted(pm for x in meas for pm in x["prompt_ms"])
    p50 = all_pms[len(all_pms) // 2]
    p95 = all_pms[int(len(all_pms) * 0.95)]
    print(json.dumps({
        "k": K, "plen": L, "reps_measured": len(meas),
        "agg_tps_median": statistics.median(tps),
        "agg_tps_iqr": (min(tps), max(tps)),
        "prompt_ms_p50": p50, "prompt_ms_p95": p95,
        "runs": res,
    }, indent=1))


if __name__ == "__main__":
    main()
