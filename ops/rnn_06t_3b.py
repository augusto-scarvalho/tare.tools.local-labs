#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06T 3B — wide-target generalization. mode=calib picks BEST_FIXED_SNAPSHOT; mode=qual tests
MAX_CONFIDENCE (frozen) vs the frozen BEST_FIXED_SNAPSHOT on a fresh disjoint set.

Wide target band [8,144] over region strata: no single fixed snapshot has observed every target, so
the adaptive confidence selector faces a genuinely harder problem than in 3A. Single-pass capture on
the official fast path; fixed batch size.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_06t_lib as L      # noqa: E402
import rnn_06d_lib as D6     # noqa: E402

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06T")
TOK_ID = "EleutherAI/gpt-neox-20b"
POOL_SEED = 20260817
BATCH = 16
M = 192
SCHEDULE = [38, 76, 115, 153]
CTX_LEN = 4 * M
CAP_BOUNDS = [4 * (s + 1) for s in SCHEDULE] + [CTX_LEN]
SESOI_RECOVERY = 0.15
SESOI_ADAPTIVE = 0.05
CI_LB_RECOVERY = 0.05
ROBUST_MIN = 3       # of 4 region strata


def git(*a):
    try:
        return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"<git-error:{e}>"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    import math
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def boot_ci(delta, strat, s_strata, seed):
    rb = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 0x3B])))
    idxs = {s: np.where(strat == s)[0] for s in range(s_strata)}
    b = []
    for _ in range(2000):
        idx = np.concatenate([idxs[s][rb.integers(0, len(idxs[s]), size=len(idxs[s]))] for s in idxs])
        b.append(float(delta[idx].mean()))
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def capture_readouts(model, examples, pools, vt, V):
    N = len(examples); K = len(SCHEDULE)
    ctx_all, query_all, golds, tslots = [], [], [], []
    for e in examples:
        toks, gold = D6.materialize_d0(e, M, pools)
        ctx_all.append(toks[:CTX_LEN]); query_all.append(toks[-2:]); golds.append(gold); tslots.append(e["target_slot"])
    pool_logits = np.zeros((N, K, V), np.float32); final_logits = np.zeros((N, V), np.float32)
    counters = {"singlePassRuns": 0, "snapshotsCapturedInRun": 0, "snapshotsRestored": 0,
                "candidateSnapshotsScored": 0, "queriesEvaluated": 0, "snapshotBoundaryFailures": 0}
    L.reset_counters()
    for b in range(0, N, BATCH):
        rows = list(range(b, min(b + BATCH, N)))
        ctx = torch.tensor([ctx_all[r] for r in rows], device=L.DEVICE, dtype=torch.long)
        q = torch.tensor([query_all[r] for r in rows], device=L.DEVICE, dtype=torch.long)
        snaps = L.run_trajectory(model, ctx, CAP_BOUNDS)
        counters["singlePassRuns"] += 1; counters["snapshotsCapturedInRun"] += len(rows) * (K + 1)
        for ki, s in enumerate(SCHEDULE):
            pos = 4 * (s + 1)
            _, sub = L.readout(model, snaps[pos], q, pos, vt)
            counters["snapshotsRestored"] += len(rows); counters["candidateSnapshotsScored"] += len(rows)
            counters["queriesEvaluated"] += len(rows)
            for li, r in enumerate(rows):
                pool_logits[r, ki] = sub[li].cpu().numpy()
        _, sub = L.readout(model, snaps[CTX_LEN], q, CTX_LEN, vt)
        counters["snapshotsRestored"] += len(rows); counters["queriesEvaluated"] += len(rows)
        for li, r in enumerate(rows):
            final_logits[r] = sub[li].cpu().numpy()
        del snaps; torch.cuda.empty_cache()
    return pool_logits, final_logits, np.array(golds), np.array(tslots), counters


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "qual"
    t0 = time.time(); runner = os.path.abspath(__file__)
    L.install_counters()
    import mamba_ssm  # noqa
    tok = AutoTokenizer.from_pretrained(TOK_ID)
    pools, _ = L.build_official_pools(tok, POOL_SEED)
    vset = sorted(set(pools["scored_vals"])); vt = torch.tensor(vset, device=L.DEVICE, dtype=torch.long); V = len(vset)
    model = L.load_model(); w0 = L.weights_identity(model)
    K = len(SCHEDULE)

    def softmax(x):
        e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

    if mode == "calib":
        spec = json.load(open(os.path.join(OUTDIR, "T1_3B_CALIBRATION_SPEC.json")))
        ex = spec["examples"]; strat = np.array([e["stratum"] for e in ex])
        pl, fl, golds, tslots, counters = capture_readouts(model, ex, pools, vt, V)
        gcol = np.array([vset.index(int(g)) for g in golds])
        pool_correct = (pl.argmax(-1) == gcol[:, None])
        fixed_acc = {int(SCHEDULE[k]): float(pool_correct[:, k].mean()) for k in range(K)}
        best_slot = max(fixed_acc, key=fixed_acc.get)
        maxconf_sel = softmax(pl).max(-1).argmax(1)
        maxconf_acc = float((pl.argmax(-1)[np.arange(len(ex)), maxconf_sel] == gcol).mean())
        out = {"packet": "RNN-06T-3B-CALIB", "calibrationSetSha256": spec["setSha256"],
               "fixed_slot_acc": fixed_acc, "BEST_FIXED_SNAPSHOT": best_slot,
               "maxconf_acc_calib": maxconf_acc, "final_acc_calib": float((fl.argmax(-1) == gcol).mean()),
               "counters": counters, "note": "BEST_FIXED_SNAPSHOT frozen from calibration before qual"}
        json.dump(out, open(os.path.join(OUTDIR, "T1_3B_CALIBRATION.json"), "w"), indent=2, default=str)
        print(f"fixed_slot_acc={fixed_acc}  BEST_FIXED_SNAPSHOT={best_slot}  maxconf_calib={maxconf_acc:.3f}")
        return

    # ---- qualification ----
    calib = json.load(open(os.path.join(OUTDIR, "T1_3B_CALIBRATION.json")))
    best_slot = int(calib["BEST_FIXED_SNAPSHOT"]); best_k = SCHEDULE.index(best_slot)
    spec = json.load(open(os.path.join(OUTDIR, "T1_3B_QUALIFICATION_SPEC.json")))
    ex = spec["examples"]; N = len(ex); strat = np.array([e["stratum"] for e in ex]); s_strata = spec["s_strata"]
    res = {"packet": "RNN-06T-3B", "kind": "wide_target_generalization",
           "qualificationSetSha256": spec["setSha256"], "disjointness_proof": spec["disjointness_proof"],
           "BEST_FIXED_SNAPSHOT": best_slot, "band": spec["band"], "regions": spec["regions"],
           "executed_source_identity": {"runner_git_blob": git("hash-object", runner),
                                        "runner_dirty": git("status", "--porcelain", "--", runner),
                                        "git_head": git("rev-parse", "HEAD"), "repo_id": L.REPO_ID,
                                        "revision": L.REVISION, "mamba_ssm": mamba_ssm.__version__,
                                        "model_weights_identity": w0,
                                        "protocol_sha256": sha256_file(os.path.join(OUTDIR, "T1_3B_PRE_REGISTRATION.md"))}}
    pl, fl, golds, tslots, counters = capture_readouts(model, ex, pools, vt, V)
    gcol = np.array([vset.index(int(g)) for g in golds])
    pool_pred = pl.argmax(-1); pool_correct = (pool_pred == gcol[:, None]); pool_top1 = softmax(pl).max(-1)
    final_correct = (fl.argmax(-1) == gcol); accF = float(final_correct.mean())
    best_fixed = pool_correct[:, best_k]
    maxconf_sel = pool_top1.argmax(1); maxconf = (pool_pred[np.arange(N), maxconf_sel] == gcol)
    oracle_best = pool_correct.any(1)
    prox = [D6.proximal_snapshot_index(SCHEDULE, int(t)) for t in tslots]
    proximal = pool_correct[np.arange(N), np.array([p[0] for p in prox])]

    def rep(correct, ref, refname):
        d = correct.astype(float) - ref.astype(float)
        per = {int(s): float(correct[strat == s].mean() - ref[strat == s].mean()) for s in range(s_strata)}
        return {"delta": round(float(d.mean()), 4), "ci95": boot_ci(d, strat, s_strata, spec["master_seed"]),
                "per_stratum": per, "robust": sum(1 for s in per if per[s] >= 0), "vs": refname}

    fixed_all = {int(SCHEDULE[k]): float(pool_correct[:, k].mean()) for k in range(K)}
    res["arms"] = {"FINAL": {"acc": accF, "ci95": wilson(int(final_correct.sum()), N)},
                   "ORACLE_BEST_GOLD": {"acc": float(oracle_best.mean()), "note": "diag"},
                   "ORACLE_TARGET_PROXIMAL": {"acc": float(proximal.mean()), "note": "diag"},
                   "BEST_FIXED_SNAPSHOT": {"slot": best_slot, "acc": float(best_fixed.mean()),
                                           "vs_final": rep(best_fixed, final_correct, "FINAL")},
                   "MAX_CONFIDENCE": {"acc": float(maxconf.mean()),
                                      "vs_final": rep(maxconf, final_correct, "FINAL"),
                                      "vs_best_fixed": rep(maxconf, best_fixed, "BEST_FIXED"),
                                      "selectedSnapshotHistogram": [int((maxconf_sel == k).sum()) for k in range(K)],
                                      "n_recovered_vs_final": int((~final_correct & maxconf).sum()),
                                      "n_harmed_vs_final": int((final_correct & ~maxconf).sum()),
                                      "n_better_than_bestfixed": int((maxconf & ~best_fixed).sum()),
                                      "n_worse_than_bestfixed": int((~maxconf & best_fixed).sum())},
                   "all_fixed_slot_acc": fixed_all,
                   "per_region_fixed_and_maxconf": {
                       int(s): {"final": float(final_correct[strat == s].mean()),
                                "maxconf": float(maxconf[strat == s].mean()),
                                "best_fixed": float(best_fixed[strat == s].mean()),
                                **{f"slot{SCHEDULE[k]}": float(pool_correct[strat == s, k].mean()) for k in range(K)}}
                       for s in range(s_strata)}}

    rec = res["arms"]["MAX_CONFIDENCE"]["vs_final"]
    wide_recovery = ("QUALIFIED" if (rec["delta"] >= SESOI_RECOVERY and rec["ci95"][0] > CI_LB_RECOVERY
                     and rec["robust"] >= ROBUST_MIN) else
                     ("PARTIAL" if rec["delta"] >= SESOI_RECOVERY else "NOT_REPLICATED"))
    adv = res["arms"]["MAX_CONFIDENCE"]["vs_best_fixed"]
    if adv["delta"] >= SESOI_ADAPTIVE and adv["ci95"][0] > 0 and adv["robust"] >= ROBUST_MIN:
        adaptive = "QUALIFIED"
    elif adv["delta"] > 0:
        adaptive = "DIRECTIONAL"
    else:
        adaptive = "NOT_QUALIFIED"
    res["mechanism_activation"] = counters
    res["fast_path_active"] = bool(not any(L.fallback_reachable().values()) and L.KCOUNT["selective_state_update"] > 0)
    res["weights_immutable"] = bool(w0 == L.weights_identity(model))
    res["WIDE_TARGET_RECOVERY"] = wide_recovery
    res["ADAPTIVE_SELECTION"] = adaptive
    res["total_runtime_s"] = round(time.time() - t0, 1)
    res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    json.dump(res, open(os.path.join(OUTDIR, "T1_3B_RESULTS.json"), "w"), indent=2, default=str)
    np.savez_compressed(os.path.join(OUTDIR, "T1_3B_READOUTS.npz"), pool_logits=pl, final_logits=fl,
                        golds=golds, target_slots=tslots, strata=strat, schedule=np.array(SCHEDULE))

    print(f"\nBEST_FIXED_SNAPSHOT(calib)=slot{best_slot}")
    print(f"FINAL={accF:.3f} BEST_FIXED={res['arms']['BEST_FIXED_SNAPSHOT']['acc']:.3f} "
          f"MAX_CONF={res['arms']['MAX_CONFIDENCE']['acc']:.3f} ORACLE={res['arms']['ORACLE_BEST_GOLD']['acc']:.3f}")
    print(f"all_fixed_slot_acc={fixed_all}")
    print(f"per_region: {json.dumps(res['arms']['per_region_fixed_and_maxconf'])}")
    print(f"MAX_CONF vs FINAL: {rec['delta']:+.3f}{rec['ci95']} robust={rec['robust']}/4")
    print(f"MAX_CONF vs BEST_FIXED: {adv['delta']:+.4f}{adv['ci95']} robust={adv['robust']}/4 "
          f"histogram={res['arms']['MAX_CONFIDENCE']['selectedSnapshotHistogram']}")
    print(f"WIDE_TARGET_RECOVERY={wide_recovery}  ADAPTIVE_SELECTION={adaptive}")
    print(f"fast_path_active={res['fast_path_active']} runtime={res['total_runtime_s']}s")


if __name__ == "__main__":
    main()
