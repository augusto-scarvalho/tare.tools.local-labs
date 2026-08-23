#!/usr/bin/env python3
"""A5 step-2: gemma-4-12b-vision DSpark-vs-incumbent-MTP A/B (batch=1, greedy).

Three isolated arms, one server each, on the DSpark-enabled probe binary:
  nospec  - floor (no speculation)
  mtp     - INCUMBENT: QAT-MTP assistant (--spec-type draft-mtp, ~0.68 accept on MMStar)
  dspark  - CANDIDATE: DeepSeek DSpark block7 draft (--spec-type draft-dspark)

Measures decode t/s + draft acceptance (draft_n_accepted/draft_n) + a content hash to
sanity-check greedy losslessness. A4 defenses: cache_prompt=False + unique per-rep prefix
(avoids #26100 repeated-prompt accept inflation). Text reasoning prompt = clean RELATIVE
dspark-vs-mtp read on identical conditions (absolute != the 0.68 MMStar-VQA number; a
vlm_vqa_bench rerun is the confirmation if dspark looks promising).

Env: PORT=8092 REPS=3 NTOK=512 COOLDOWN=12 ARMS=nospec,mtp,dspark NMAX_DSPARK=7 NMAX_MTP=4
"""
import json, os, subprocess, time, urllib.request, statistics, math, signal, tempfile, hashlib

BIN = "/home/augus/src/slop.cpp-main/build/bin/llama-server"
PORT = int(os.environ.get("PORT", "8092"))
REPS = int(os.environ.get("REPS", "3"))
NTOK = int(os.environ.get("NTOK", "512"))
COOLDOWN = int(os.environ.get("COOLDOWN", "12"))
NMAX_DSPARK = os.environ.get("NMAX_DSPARK", "7")
NMAX_MTP = os.environ.get("NMAX_MTP", "4")
ARMS = os.environ.get("ARMS", "nospec,mtp,dspark").split(",")

G = "/home/augus/models/gemma-4-12b-vision"
MODEL = f"{G}/gemma-4-12B-it-Q4_0.gguf"
MMPROJ = f"{G}/mmproj-gemma-4-12B-it-Q8_0.gguf"
QAT = f"{G}/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf"
DSPARK = f"{G}/dspark_gemma4_12b_q4pure.gguf"

COMMON = ["-m", MODEL, "--mmproj", MMPROJ, "-ngl", "99", "-fa", "on", "--ctx-size", "8192",
          "--reasoning-budget", "256", "--host", "127.0.0.1", "--port", str(PORT)]
ARM_FLAGS = {
    "nospec": [],
    "mtp":    ["--spec-type", "draft-mtp",    "--spec-draft-n-max", NMAX_MTP,    "--model-draft", QAT],
    "dspark": ["--spec-type", "draft-dspark", "--spec-draft-n-max", NMAX_DSPARK, "--model-draft", DSPARK],
}
PROMPT = ("Explain step by step, in several paragraphs, how a red-black tree rebalances after "
          "an insertion that creates two consecutive red nodes. Cover the uncle-red recolor case, "
          "the triangle rotation, and the line rotation, and explain why each preserves the "
          "black-height invariant.\n\nAnswer:")

def wait_health(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3).read(); return True
        except Exception:
            time.sleep(2)
    return False

def complete(nonce):
    body = json.dumps({"prompt": f"[req {nonce}] " + PROMPT, "n_predict": NTOK,
                       "temperature": 0.0, "top_k": 1, "cache_prompt": False, "seed": 42}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r.get("timings", {}), r.get("content", "")

def run_arm(arm):
    flags = [BIN, *COMMON, *ARM_FLAGS[arm]]
    log = tempfile.NamedTemporaryFile(delete=False, suffix=f".{arm}.log").name
    lf = open(log, "w")
    proc = subprocess.Popen(flags, stdout=lf, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, preexec_fn=os.setsid)
    try:
        if not wait_health():
            print(f"   !! {arm} server never healthy; log {log}\n{open(log).read()[-1800:]}"); return None
        complete("warmup")
        tps, accepts, hashes = [], [], []
        for rep in range(REPS):
            t, content = complete(f"{arm}-{rep}")
            if t.get("predicted_per_second"): tps.append(t["predicted_per_second"])
            dn = t.get("draft_n"); da = t.get("draft_n_accepted")
            if dn and da is not None: accepts.append(da / dn)
            hashes.append(hashlib.sha1(content.encode()).hexdigest()[:12])
        return {"tps": tps, "accept": (statistics.mean(accepts) if accepts else None),
                "hash": hashes[0], "hash_stable": len(set(hashes)) == 1, "log": log}
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM); proc.wait(timeout=30)
        except Exception:
            try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception: pass
        lf.close(); time.sleep(COOLDOWN)

def ci95(xs):
    if len(xs) < 2: return 0.0
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))

def main():
    res = {}
    for arm in ARMS:
        print(f"\n=== {arm} ===", flush=True)
        r = run_arm(arm)
        if not r: continue
        res[arm] = r
        m = statistics.mean(r["tps"]) if r["tps"] else 0
        print(f"   tps={m:.1f} ±{ci95(r['tps']):.1f}  accept={r['accept'] if r['accept'] is not None else '-'}  "
              f"hash={r['hash']} stable={r['hash_stable']}")
    base = res.get("nospec", {}).get("tps")
    b = statistics.mean(base) if base else None
    print("\n===== A5 GEMMA DSPARK A/B =====")
    print(f"{'arm':8} {'tps':>10} {'±ci':>6} {'accept':>8} {'edge%':>8} {'lossless-vs-nospec':>20}")
    nh = res.get("nospec", {}).get("hash")
    for arm in ARMS:
        if arm not in res: continue
        r = res[arm]; m = statistics.mean(r["tps"]) if r["tps"] else 0
        edge = f"{100*(m-b)/b:+.1f}" if (b and m) else "-"
        acc = f"{r['accept']:.3f}" if r["accept"] is not None else "-"
        loss = "(floor)" if arm == "nospec" else ("MATCH" if (nh and r["hash"] == nh) else "DIFFERS(tie-flips)")
        print(f"{arm:8} {m:>10.1f} {ci95(r['tps']):>6.1f} {acc:>8} {edge:>8} {loss:>20}")

if __name__ == "__main__":
    main()
