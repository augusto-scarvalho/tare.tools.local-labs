#!/usr/bin/env python3
"""A1-0 de-risk (IDEAS_BACKLOG A1): does the MTP decode-t/s edge shrink with context depth?

The deploy MoE and the dense-27B are BOTH GDN hybrids; only ~25% of base layers bear a KV cache,
but the `nextn`/MTP head (blk 40) IS a full-attention KV-bearing layer -> the MTP draft re-attends
the full-context KV each speculated token. So MTP's edge over no-spec MAY shrink at long context.
This measures it: per (model, depth) launch llama-server for no-spec vs draft-mtp on ONE fixed
high-accept generation task, varying only the amount of preceding context. Record decode t/s +
draft accept. Isolated arms + cooldown (the GPU-A/B variance rule).

Runs inside WSL (native binary/model paths). Invoke via:
  wsl.exe -d Ubuntu-24.04 -- bash -lc 'MSYS_NO_PATHCONV=1 MODELSET=moe \
    /home/augus/evalplus-venv/bin/python3 /mnt/c/projects/local-model-lifecycle/ops/a1_mtp_depth_bench.py'
Env: MODELSET=moe|dense (default both), REPS=3, NTOK=256, COOLDOWN=20, PORT=8080, OUT=<csv path>.
"""
import json, os, subprocess, sys, time, urllib.request, statistics, signal, tempfile

BIN = "/home/augus/src/llama.cpp-master/build/bin/llama-server"
PORT = int(os.environ.get("PORT", "8080"))
REPS = int(os.environ.get("REPS", "3"))
NTOK = int(os.environ.get("NTOK", "256"))
COOLDOWN = int(os.environ.get("COOLDOWN", "20"))
MODELSET = os.environ.get("MODELSET", "moe,dense").split(",")
DEPTHS_FILTER = set(x for x in os.environ.get("DEPTHS", "").split(",") if x)  # e.g. "8k" for smoke test
OUT = os.environ.get("OUT", "/mnt/c/projects/local-model-lifecycle/runs/a1-mtp-depth/a1_depth.csv")

# model -> (gguf, common flags, [(depth_label, ctx_size, kv_type, target_tokens)])
MODELS = {
    "moe": (
        "/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        ["-fa", "on", "--n-cpu-moe", "8", "-np", "1", "--batch-size", "2048", "--ubatch-size", "2048"],
        [("8k", 8192, "q8_0", 6600), ("128k", 131072, "q4_0", 122000)],
    ),
    "dense": (  # 27B GDN hybrid; -ngl 65; long-ctx ceiling ~48-64k (CONTEXT_PLAN), match -c to use
        "/home/augus/models/qwen36-27b-mtp/Qwen3.6-27B-Q4_K_M.gguf",
        ["-fa", "on", "-ngl", "65", "-np", "1", "--batch-size", "2048", "--ubatch-size", "2048"],
        [("8k", 8192, "q8_0", 6600), ("48k", 49152, "q4_0", 45000)],
    ),
}

# One coherent-code padding unit (~60 tok). Repeat to reach a target depth; then a FIXED task whose
# continuation is highly predictable from the pattern -> high MTP accept = where the edge is largest
# = the best place to detect degradation.
def make_prompt(n_units):
    head = "# Auto-generated data-processing module. Each section follows the same pattern.\n\n"
    unit = ("# ---- module section {i} ----\n"
            "def process_batch_{i}(records, config):\n"
            "    results = []\n"
            "    for r in records:\n"
            "        if r.get('active') and r['score'] > {i}:\n"
            "            results.append({{'id': r['id'], 'value': r['score'] * {i}, 'tag': 'sec{i}'}})\n"
            "    return results\n\n")
    body = "".join(unit.format(i=i) for i in range(1, n_units + 1))
    # Sustained, highly predictable generation (matches the S3 GEN regime / §E4 structured-code, ~400 tok):
    # a 20-field validated class. Same task at every depth -> only context depth varies.
    task = ("# Given all the sections above, now write ONE Python class UserProfileValidated with @property\n"
            "# getters and setters for these 20 fields, each setter validating the type and raising TypeError\n"
            "# on mismatch: user_id:int, username:str, email:str, full_name:str, age:int, is_active:bool,\n"
            "# is_admin:bool, created_at:str, updated_at:str, last_login:str, login_count:int, bio:str,\n"
            "# avatar_url:str, phone_number:str, country_code:str, timezone:str, locale:str,\n"
            "# email_verified:bool, phone_verified:bool, two_factor_enabled:bool. Output only the code.\n"
            "class UserProfileValidated:\n")
    return head + body + task

def size_prompt(target_tokens):
    # ~91 tok/unit measured at depth (4-digit section indices inflate it vs the 8k case); target is the
    # desired PROMPT token count, kept a few k under n_ctx so prompt + n_predict fits.
    return make_prompt(max(1, round(target_tokens / 91)))

def wait_health(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3).read()
            return True
        except Exception:
            time.sleep(2)
    return False

def complete(prompt):
    body = json.dumps({
        "prompt": prompt, "n_predict": NTOK, "temperature": 0.0, "top_k": 1,
        "cache_prompt": True, "seed": 42,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=1200))
    except urllib.error.HTTPError as e:
        print(f"   !! HTTP {e.code}: {e.read().decode()[:400]}")
        raise
    return r.get("timings", {}), r.get("content", "")

def draft_fields(t):
    # discover accept accounting regardless of exact key naming across builds
    d = {k: v for k, v in t.items() if any(x in k.lower() for x in ("draft", "accept", "spec"))}
    return d

def run_cell(model_key, gguf, common, depth_label, ctx, kv, target, spec):
    prompt = size_prompt(target)
    flags = [BIN, "-m", gguf, *common, "--ctx-size", str(ctx),
             "--cache-type-k", kv, "--cache-type-v", kv,
             "--host", "127.0.0.1", "--port", str(PORT)]
    if spec:
        flags += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]
    log = tempfile.NamedTemporaryFile(delete=False, suffix=".log").name
    lf = open(log, "w")
    proc = subprocess.Popen(flags, stdout=lf, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, preexec_fn=os.setsid)
    try:
        if not wait_health():
            print(f"!! {model_key}/{depth_label}/{'mtp' if spec else 'nospec'}: server never healthy; log {log}")
            print(open(log).read()[-2000:]); return None
        # warmup (rep-0 pays the prefill; fills KV to depth)
        t0, _ = complete(prompt)
        prompt_n = t0.get("prompt_n")
        rows = []
        for rep in range(REPS):
            t, content = complete(prompt)
            tps = t.get("predicted_per_second")
            df = draft_fields(t)
            rows.append((tps, df, t.get("prompt_n"), content))
            print(f"   rep{rep}: tps={tps} draft={df}")
        return {"prompt_n": prompt_n, "rows": rows, "sample": rows[-1][3][:120]}
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception: pass
        lf.close()
        time.sleep(COOLDOWN)  # cooldown so back-to-back arms don't heat-soak the GPU

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    new = not os.path.exists(OUT)
    csv = open(OUT, "a")
    if new:
        csv.write("model,depth,ctx,kv,spec,rep,prompt_n,tps,draft_n,draft_accepted,accept_rate\n")
    summary = {}
    for mk in MODELSET:
        gguf, common, depths = MODELS[mk]
        for depth_label, ctx, kv, target in depths:
            if DEPTHS_FILTER and depth_label not in DEPTHS_FILTER:
                continue
            for spec in (False, True):
                tag = f"{mk}/{depth_label}/{'mtp' if spec else 'nospec'}"
                print(f"\n=== {tag} (ctx={ctx} kv={kv}) ===", flush=True)
                res = run_cell(mk, gguf, common, depth_label, ctx, kv, target, spec)
                if not res:
                    continue
                tpss = []
                for rep, (tps, df, pn, _) in enumerate(res["rows"]):
                    dn = df.get("draft_n") or df.get("n_draft")
                    da = df.get("draft_n_accepted") or df.get("n_accept") or df.get("draft_accepted")
                    ar = (da / dn) if (dn and da is not None) else ""
                    csv.write(f"{mk},{depth_label},{ctx},{kv},{'mtp' if spec else 'nospec'},{rep},{pn},{tps},{dn or ''},{da if da is not None else ''},{ar}\n")
                    if tps: tpss.append(tps)
                csv.flush()
                summary[(mk, depth_label, spec)] = (statistics.median(tpss) if tpss else None, res["prompt_n"])
                print(f"   -> median tps={summary[(mk, depth_label, spec)][0]}  prompt_n={res['prompt_n']}  sample={res['sample']!r}")
    # deltas
    print("\n\n===== A1-0 SUMMARY: MTP edge vs depth =====")
    print(f"{'model':6} {'depth':6} {'prompt_n':>9} {'nospec':>8} {'mtp':>8} {'delta%':>8}")
    for mk in MODELSET:
        for depth_label, ctx, kv, target in MODELS[mk][2]:
            ns = summary.get((mk, depth_label, False), (None, None))
            mt = summary.get((mk, depth_label, True), (None, None))
            if ns[0] and mt[0]:
                d = 100 * (mt[0] - ns[0]) / ns[0]
                print(f"{mk:6} {depth_label:6} {str(mt[1]):>9} {ns[0]:8.1f} {mt[0]:8.1f} {d:+8.1f}")
    csv.close()

if __name__ == "__main__":
    main()
