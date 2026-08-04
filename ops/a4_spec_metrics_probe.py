#!/usr/bin/env python3
"""A4 validation probe: prove the harness's spec-decode metrics match the server's own.

The harness (collectors/request.py) derives, from the JSON `timings` alone:
    alpha = draft_n_accepted / draft_n                      (== server's draft_ratio)
    tau   = 1 + draft_n_accepted / (predicted_n - draft_n_accepted - 1)   (== mean_acc_len)

The '-1' is the single non-speculative first token; that term is the ONLY thing this
derivation could get wrong, and n_draft_verif_steps is not in the JSON to check it
directly. So this probe launches the server with stderr KEPT (verify_mtp.py throws it
away), reads the authoritative `draft acceptance = A (.. / ..), mean len = M` the server
prints (upstream #24536), and asserts the harness numbers reproduce A and M. It doubles
as the standing A4 gate: run it after any pin bump to confirm the fields still flow.

    python3 a4_spec_metrics_probe.py            # deploy MoE MTP, ncmoe=8
"""
import json, os, re, subprocess, sys, time, urllib.request

BIN   = os.environ.get("MTP_BIN", "/home/augus/src/llama.cpp-master/build/bin/llama-server")
MODEL = os.environ.get("MODEL",
    "/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
PORT  = 8096
LOG   = "/tmp/a4_spec_probe.stderr.log"
# A real, mixed prompt -- code + prose, the workload §63.4 says acceptance is sensitive to.
PROMPT = ("Write a Python function `merge_intervals(intervals)` that merges overlapping "
          "intervals and returns the sorted result, then explain its time complexity and "
          "give two edge cases worth a unit test.")
COMMON = [BIN, "-m", MODEL, "--host", "127.0.0.1", "--port", str(PORT), "-fa", "on",
          "--n-cpu-moe", "8", "--ctx-size", "8192"]
MODES = {
    "nospec": COMMON,
    "mtp":    COMMON + ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"],
}


def wait_ready(timeout=120):
    for _ in range(timeout):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def complete():
    body = json.dumps({"prompt": PROMPT, "n_predict": 256, "temperature": 0.0, "top_k": 1,
                       "cache_prompt": False, "seed": 42, "timings_per_token": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


GAMMA = 4   # matches --spec-draft-n-max below; tau = 1 + GAMMA*alpha must reproduce the log


def harness_alpha(t):
    """Exactly the request.py derivation, duplicated here so the probe tests the FORMULA,
    not an import (a bug in the shared code should make the probe fail, not vanish).
    alpha is the ONLY thing derivable from the JSON alone; tau needs gamma (below)."""
    dn, da = t.get("draft_n"), t.get("draft_n_accepted")
    if not dn or da is None:
        return None
    return da / dn


ok = True
for mode, args in MODES.items():
    print(f"\n===== {mode} =====", flush=True)
    with open(LOG, "w") as errf:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=errf)
    try:
        if not wait_ready():
            print(f"{mode}: server never became ready"); proc.kill(); ok = False; continue
        resp = complete()
        # print_timings only fires for n_decoded >= 100; give it a beat to flush.
        time.sleep(1)
        proc.terminate()
        try: proc.wait(timeout=15)
        except Exception: proc.kill()
    finally:
        time.sleep(2)

    t = resp.get("timings", {})
    dn, da, pn = t.get("draft_n"), t.get("draft_n_accepted"), t.get("predicted_n")
    tps = t.get("predicted_per_second")
    tpot = t.get("predicted_per_token_ms")
    print(f"  JSON: predicted_n={pn} draft_n={dn} draft_n_accepted={da} "
          f"tps={tps:.1f} tpot_ms={tpot:.2f}")

    log = open(LOG).read()
    # NB the server pads with multiple spaces ("mean len =  3.79") -> \s+, not a literal
    # space, or the value's leading space defeats [0-9.]+ (the bug the first run hit).
    m = re.search(r"draft acceptance\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)"
                  r"\s*generated\s*\)\s*,\s*mean len\s*=\s*([0-9.]+)", log)

    if mode == "nospec":
        # The whole point of the None-not-0 rule: no draft keys at all.
        if dn is None and da is None:
            print("  OK: no draft fields in JSON (spec was off) -> accept_rate stays None")
        else:
            print(f"  FAIL: nospec arm leaked draft fields dn={dn} da={da}"); ok = False
        if m:
            print(f"  FAIL: nospec arm logged a draft-acceptance line"); ok = False
        continue

    alpha = harness_alpha(t)
    if alpha is None:
        print("  FAIL: harness derived no alpha on a spec arm"); ok = False; continue
    if not m:
        print("  FAIL: server logged no 'draft acceptance' line (raise n_predict?)")
        ok = False; continue
    s_ratio, s_acc, s_gen, s_meanlen = float(m[1]), int(m[2]), int(m[3]), float(m[4])
    tau_identity = 1.0 + GAMMA * alpha          # the exact relation the report uses
    print(f"  harness: alpha={alpha:.5f}  tau(=1+{GAMMA}*alpha)={tau_identity:.4f}")
    print(f"  server : ratio={s_ratio:.5f}  mean_len={s_meanlen:.4f}  "
          f"({s_acc} accepted / {s_gen} generated)")

    # 1) alpha must equal the server's draft_ratio to the log's 5 decimals.
    if abs(alpha - s_ratio) > 5e-5:
        print(f"  FAIL: alpha {alpha:.5f} != server ratio {s_ratio:.5f}"); ok = False
    else:
        print("  PASS: alpha == server draft_ratio")
    # 2) the JSON counts must equal the counts the server logged.
    if (da, dn) != (s_acc, s_gen):
        print(f"  FAIL: JSON (acc={da},gen={dn}) != log (acc={s_acc},gen={s_gen})"); ok = False
    else:
        print("  PASS: JSON draft counts == logged counts")
    # 3) THE tau test: the report's tau = 1 + gamma*alpha must reproduce the server's own
    #    mean_acc_len (which the JSON cannot give directly -- n_draft_verif_steps is
    #    log-only). This is what makes tau trustworthy without a fragile predicted_n term;
    #    a mismatch means the drafter is NOT proposing gamma per step (partial last steps
    #    beyond rounding), so re-derive before trusting tau. Tolerance = the server's %.2f
    #    rounding plus a hair.
    if abs(tau_identity - s_meanlen) > 0.02:
        print(f"  FAIL: tau 1+{GAMMA}*alpha={tau_identity:.4f} != server mean_len "
              f"{s_meanlen:.4f} -> drafter not proposing gamma/step; re-derive"); ok = False
    else:
        print(f"  PASS: tau == server mean_acc_len (the 1+gamma*alpha identity holds)")

print("\n" + ("A4 PROBE OK" if ok else "A4 PROBE FAILED"))
sys.exit(0 if ok else 1)
