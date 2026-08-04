#!/usr/bin/env python3
"""A1-0 / A1-0b (IDEAS_BACKLOG A1): MTP decode edge & draft acceptance vs context depth.

Deploy MoE and dense-27B are GDN hybrids; the `nextn`/MTP head (blk 40) is a full-attention KV-bearing
layer (build_attn_inp_kv + build_attn in src/models/qwen35moe.cpp), so the draft DOES re-attend the full
KV each speculated token. Literature (Leviathan c-ratio, MagicDec, EAGLE-3.1, DeepSeek-V3 MTP) predicts
that in the 8k-128k range the *target*-forward O(S) cost dominates and speculation amortizes it -> the
MTP edge GROWS with depth at ~stable acceptance. The draft's OWN O(S) attention tax (arXiv:2607.21535
"Windowed-MTP") only surfaces at >=256k. This bench measures decode t/s + draft accept across depth to
(a) confirm the edge-grows/accept-stable behavior and (b) rule out the llama.cpp slot-boundary
acceptance-oscillation bug (#23658) / hybrid-cache accept collapse (#23322) on OUR pinned base.

Two generation tasks (only depth varies within a task):
  T1 'class'  — a 20-field validated class: context-INDEPENDENT, ~99% accept. Flat accept across depth
                here isolates a pure slot/cache BUG (drafts are identical regardless of padding).
  T2 'reason' — free-form prose reasoning: realistic ~80% accept (the §E4 reasoning regime, headroom to
                fall). Tests whether the LOW-accept regime stays depth-stable.

Runs inside WSL. Invoke, e.g.:
  wsl.exe -d Ubuntu-24.04 -- bash -lc 'MSYS_NO_PATHCONV=1 MODELSET=moe TASK=class SPECS=mtp \
    DEPTHS=8k,32k,64k,96k,128k REPS=5 \
    /home/augus/evalplus-venv/bin/python3 /mnt/c/projects/local-model-lifecycle/ops/a1_mtp_depth_bench.py'
Env: MODELSET=moe|dense (csv), TASK=class|reason, SPECS=nospec,mtp (csv, default both),
     DEPTHS=<csv of labels> (default all for the model), REPS=5, NTOK=450, COOLDOWN=15, PORT=8080, OUT=<csv>.
"""
import json, os, subprocess, sys, time, urllib.request, statistics, math, signal, tempfile

BIN = "/home/augus/src/llama.cpp-master/build/bin/llama-server"
PORT = int(os.environ.get("PORT", "8080"))
REPS = int(os.environ.get("REPS", "5"))
NTOK = int(os.environ.get("NTOK", "450"))
COOLDOWN = int(os.environ.get("COOLDOWN", "15"))
MODELSET = os.environ.get("MODELSET", "moe,dense").split(",")
TASK = os.environ.get("TASK", "class")            # class (T1) | reason (T2)
SPECS = os.environ.get("SPECS", "nospec,mtp").split(",")
DEPTHS_FILTER = set(x for x in os.environ.get("DEPTHS", "").split(",") if x)
OUT = os.environ.get("OUT", "/mnt/c/projects/local-model-lifecycle/runs/a1-mtp-depth/a1_depth.csv")

# model -> (gguf, common flags, [(depth_label, ctx_size, kv_type, target_prompt_tokens)])
MODELS = {
    "moe": (
        "/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        ["-fa", "on", "--n-cpu-moe", "8", "-np", "1", "--batch-size", "2048", "--ubatch-size", "2048"],
        [("8k", 8192, "q8_0", 6600), ("32k", 32768, "q4_0", 30000),
         ("64k", 65536, "q4_0", 62000), ("96k", 98304, "q4_0", 94000),
         ("128k", 131072, "q4_0", 122000)],
    ),
    "dense": (  # 27B GDN hybrid; -ngl 65; long-ctx ceiling ~48-64k (CONTEXT_PLAN); match -c to use
        "/home/augus/models/qwen36-27b-mtp/Qwen3.6-27B-Q4_K_M.gguf",
        ["-fa", "on", "-ngl", "65", "-np", "1", "--batch-size", "2048", "--ubatch-size", "2048"],
        [("8k", 8192, "q8_0", 6600), ("32k", 32768, "q4_0", 30000), ("48k", 49152, "q4_0", 45000)],
    ),
}

def make_prompt(n_units, task):
    head = "# Auto-generated data-processing module. Each section follows the same pattern.\n\n"
    unit = ("# ---- module section {i} ----\n"
            "def process_batch_{i}(records, config):\n"
            "    results = []\n"
            "    for r in records:\n"
            "        if r.get('active') and r['score'] > {i}:\n"
            "            results.append({{'id': r['id'], 'value': r['score'] * {i}, 'tag': 'sec{i}'}})\n"
            "    return results\n\n")
    body = "".join(unit.format(i=i) for i in range(1, n_units + 1))
    if task == "class":  # T1: context-independent, ~99% accept
        task_txt = ("# Given all the sections above, now write ONE Python class UserProfileValidated with @property\n"
                    "# getters and setters for these 20 fields, each setter validating the type and raising TypeError\n"
                    "# on mismatch: user_id:int, username:str, email:str, full_name:str, age:int, is_active:bool,\n"
                    "# is_admin:bool, created_at:str, updated_at:str, last_login:str, login_count:int, bio:str,\n"
                    "# avatar_url:str, phone_number:str, country_code:str, timezone:str, locale:str,\n"
                    "# email_verified:bool, phone_verified:bool, two_factor_enabled:bool. Output only the code.\n"
                    "class UserProfileValidated:\n")
    elif task == "reason":  # T2: free-form prose reasoning, realistic lower accept
        task_txt = ('# Ignore the code above. Now answer in prose (no code): Explain step by step how a red-black\n'
                    '# tree rebalances after inserting a node that creates two consecutive red nodes. Cover every\n'
                    '# case: uncle red (recolor and recurse up), uncle black with a triangle configuration (rotate\n'
                    '# the parent), and uncle black with a line configuration (rotate the grandparent and recolor).\n'
                    '# Explain why each case preserves the black-height invariant. Write several paragraphs.\n\n'
                    'Answer: When we insert a new red node into a red-black tree and its parent is also red,')
    else:
        raise SystemExit(f"unknown TASK={task}")
    return head + body + task_txt

def size_prompt(target_tokens, task):
    # ~91 tok/unit at depth (4-digit section indices); target = desired PROMPT tokens, kept under n_ctx.
    return make_prompt(max(1, round(target_tokens / 91)), task)

def wait_health(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3).read(); return True
        except Exception:
            time.sleep(2)
    return False

def complete(prompt):
    body = json.dumps({"prompt": prompt, "n_predict": NTOK, "temperature": 0.0, "top_k": 1,
                       "cache_prompt": True, "seed": 42}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=1800))
    except urllib.error.HTTPError as e:
        print(f"   !! HTTP {e.code}: {e.read().decode()[:400]}"); raise
    return r.get("timings", {}), r.get("content", "")

def draft_fields(t):
    return {k: v for k, v in t.items() if any(x in k.lower() for x in ("draft", "accept", "spec"))}

def ci95(xs):
    if len(xs) < 2: return 0.0
    sd = statistics.stdev(xs); return 1.96 * sd / math.sqrt(len(xs))

def run_cell(gguf, common, ctx, kv, target, spec, task):
    prompt = size_prompt(target, task)
    flags = [BIN, "-m", gguf, *common, "--ctx-size", str(ctx),
             "--cache-type-k", kv, "--cache-type-v", kv, "--host", "127.0.0.1", "--port", str(PORT)]
    if spec:
        flags += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]
    log = tempfile.NamedTemporaryFile(delete=False, suffix=".log").name
    lf = open(log, "w")
    proc = subprocess.Popen(flags, stdout=lf, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, preexec_fn=os.setsid)
    try:
        if not wait_health():
            print(f"   !! server never healthy; log {log}\n{open(log).read()[-1500:]}"); return None
        t0, _ = complete(prompt)                      # warmup: rep-0 pays the prefill, fills KV to depth
        prompt_n = t0.get("prompt_n")
        rows = []
        for rep in range(REPS):
            t, content = complete(prompt)
            rows.append((t.get("predicted_per_second"), draft_fields(t), content))
        return {"prompt_n": prompt_n, "rows": rows}
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM); proc.wait(timeout=30)
        except Exception:
            try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception: pass
        lf.close(); time.sleep(COOLDOWN)

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    new = not os.path.exists(OUT)
    csv = open(OUT, "a")
    if new:
        csv.write("model,task,depth,ctx,kv,spec,rep,prompt_n,tps,draft_n,draft_accepted,accept_rate\n")
    summ = {}  # (mk,depth,spec) -> (tps_mean, tps_ci, prompt_n, accept)
    for mk in MODELSET:
        gguf, common, depths = MODELS[mk]
        for depth_label, ctx, kv, target in depths:
            if DEPTHS_FILTER and depth_label not in DEPTHS_FILTER:
                continue
            for spec in [s == "mtp" for s in SPECS]:
                tag = f"{mk}/{TASK}/{depth_label}/{'mtp' if spec else 'nospec'}"
                print(f"\n=== {tag} (ctx={ctx} kv={kv}) ===", flush=True)
                res = run_cell(gguf, common, ctx, kv, target, spec, TASK)
                if not res: continue
                tpss, accept = [], None
                for rep, (tps, df, _) in enumerate(res["rows"]):
                    dn = df.get("draft_n") or df.get("n_draft")
                    da = df.get("draft_n_accepted") or df.get("n_accept") or df.get("draft_accepted")
                    ar = (da / dn) if (dn and da is not None) else ""
                    if ar != "": accept = ar
                    csv.write(f"{mk},{TASK},{depth_label},{ctx},{kv},{'mtp' if spec else 'nospec'},{rep},"
                              f"{res['prompt_n']},{tps},{dn or ''},{da if da is not None else ''},{ar}\n")
                    if tps: tpss.append(tps)
                csv.flush()
                mean = statistics.mean(tpss) if tpss else None
                summ[(mk, depth_label, spec)] = (mean, ci95(tpss), res["prompt_n"], accept)
                print(f"   -> tps={mean:.1f} ±{ci95(tpss):.1f} (n={len(tpss)})  prompt_n={res['prompt_n']}  "
                      f"accept={accept if accept is not None else '-'}")
    print(f"\n\n===== A1 SUMMARY (task={TASK}): edge & accept vs depth =====")
    print(f"{'model':6} {'depth':6} {'prompt_n':>9} {'nospec':>10} {'mtp':>12} {'edge%':>8} {'accept':>8}")
    for mk in MODELSET:
        for depth_label, ctx, kv, target in MODELS[mk][2]:
            if DEPTHS_FILTER and depth_label not in DEPTHS_FILTER: continue
            ns = summ.get((mk, depth_label, False)); mt = summ.get((mk, depth_label, True))
            nsm = f"{ns[0]:.1f}" if ns and ns[0] else "-"
            mtm = f"{mt[0]:.1f}±{mt[1]:.1f}" if mt and mt[0] else "-"
            edge = f"{100*(mt[0]-ns[0])/ns[0]:+.1f}" if (ns and mt and ns[0] and mt[0]) else "-"
            acc = f"{mt[3]:.3f}" if (mt and mt[3] is not None) else "-"
            pn = (mt or ns or (None,None,'?',None))[2]
            print(f"{mk:6} {depth_label:6} {str(pn):>9} {nsm:>10} {mtm:>12} {edge:>8} {acc:>8}")
    csv.close()

if __name__ == "__main__":
    main()
