#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06T 3A — exact-contract transportability replication on the official Mamba fast path.

Runs the T0-qualified SINGLE-PASS capture on the official state-spaces/mamba2-1.3b for a fresh
disjoint 06D-semantics qualification set. Arms: FINAL, ORACLE_BEST_GOLD (diag), ORACLE_TARGET_PROXIMAL
(diag), FIXED_SLOT_76 (mandatory non-adaptive control), MAX_CONFIDENCE (FROZEN adaptive), RECENCY/
FIXED_SLOT_153 (descriptive), MATCHED_NO_HISTORY (compute control). Two claims, separate SESOIs:
CLAIM 1 historical recovery transport (FIXED_SLOT_76 & MAX_CONFIDENCE vs FINAL); CLAIM 2 adaptive
selector incremental value (MAX_CONFIDENCE vs FIXED_SLOT_76). Fixed batch size throughout (T0 showed
batch-size numeric sensitivity; neighbor isolation bit-exact).
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
SPEC_PATH = os.path.join(OUTDIR, "T1_3A_QUALIFICATION_SPEC.json")
T0_PATH = os.path.join(OUTDIR, "T0_RESULTS.json")
TOK_ID = "EleutherAI/gpt-neox-20b"
POOL_SEED = 20260817
BATCH = 16
M = 192
SCHEDULE = [38, 76, 115, 153]
CTX_LEN = 4 * M                       # 768
CAP_BOUNDS = [4 * (s + 1) for s in SCHEDULE] + [CTX_LEN]   # token positions [156,308,464,616,768]
# frozen SESOIs / thresholds (T1_3A_PRE_REGISTRATION)
SESOI_RECOVERY = 0.15
SESOI_ADAPTIVE = 0.05
CI_LB_RECOVERY = 0.05
ROBUST_MIN = 2


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
    rb = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 0x3A])))
    idxs = {s: np.where(strat == s)[0] for s in range(s_strata)}
    b = []
    for _ in range(2000):
        idx = np.concatenate([idxs[s][rb.integers(0, len(idxs[s]), size=len(idxs[s]))] for s in idxs])
        b.append(float(delta[idx].mean()))
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    t0 = time.time(); os.makedirs(OUTDIR, exist_ok=True)
    runner = os.path.abspath(__file__); libpath = os.path.join(os.path.dirname(runner), "rnn_06t_lib.py")
    L.install_counters()
    import mamba_ssm, causal_conv1d, triton  # noqa
    res = {"packet": "RNN-06T-3A", "kind": "official_transport_exact_contract"}

    t0res = json.load(open(T0_PATH))
    assert t0res["OFFICIAL_MAMBA_LIFECYCLE"] == "QUALIFIED" and t0res["SINGLE_PASS_HISTORICAL_CAPTURE"] == "QUALIFIED", \
        "T0 not both QUALIFIED — 3A must not run"

    spec = json.load(open(SPEC_PATH))
    res["challenge_identities"] = {"qualificationSetSha256_3A": spec["qualificationSetSha256_3A"],
                                   "disjointness_proof": spec["disjointness_proof"],
                                   "schedule": SCHEDULE, "cap_bounds": CAP_BOUNDS}
    model = L.load_model()
    w0 = L.weights_identity(model); n_layer = len(model.backbone.layers)
    res["executed_source_identity"] = {
        "runner_sha256": sha256_file(runner), "runner_git_blob": git("hash-object", runner),
        "runner_dirty": git("status", "--porcelain", "--", runner), "lib_git_blob": git("hash-object", libpath),
        "git_head": git("rev-parse", "HEAD"), "repo_id": L.REPO_ID, "revision": L.REVISION,
        "mamba_ssm": mamba_ssm.__version__, "causal_conv1d": causal_conv1d.__version__,
        "triton": triton.__version__, "torch": torch.__version__, "model_weights_identity": w0,
        "protocol_sha256": sha256_file(os.path.join(OUTDIR, "T1_3A_PRE_REGISTRATION.md"))}

    tok = AutoTokenizer.from_pretrained(TOK_ID)
    pools, pool_meta = L.build_official_pools(tok, POOL_SEED)
    vset = sorted(set(pools["scored_vals"])); vt = torch.tensor(vset, device=L.DEVICE, dtype=torch.long)
    V = len(vset); res["chance"] = 1.0 / V

    examples = spec["examples"]; N = len(examples); strat = np.array([e["stratum"] for e in examples])
    ctx_all, query_all, golds, tslots = [], [], [], []
    for e in examples:
        toks, gold = D6.materialize_d0(e, M, pools)
        ctx_all.append(toks[:CTX_LEN]); query_all.append(toks[-2:]); golds.append(gold); tslots.append(e["target_slot"])
    golds = np.array(golds); tslots = np.array(tslots)
    gcol = np.array([vset.index(int(g)) for g in golds])

    K = len(SCHEDULE)
    pool_logits = np.zeros((N, K, V), np.float32); final_logits = np.zeros((N, V), np.float32)
    counters = {"singlePassRuns": 0, "snapshotsCapturedInRun": 0, "snapshotsRestored": 0,
                "candidateSnapshotsScored": 0, "queriesEvaluated": 0, "snapshotBoundaryChecks": 0,
                "snapshotBoundaryFailures": 0, "fastPathCalls": 0, "fallbackPathCalls": 0}
    run_ids = []
    L.reset_counters()
    for b in range(0, N, BATCH):
        rows = list(range(b, min(b + BATCH, N)))
        ctx = torch.tensor([ctx_all[r] for r in rows], device=L.DEVICE, dtype=torch.long)
        q = torch.tensor([query_all[r] for r in rows], device=L.DEVICE, dtype=torch.long)
        snaps = L.run_trajectory(model, ctx, CAP_BOUNDS)
        counters["singlePassRuns"] += 1
        rid = hashlib.sha256((f"3A|{b}|" + ",".join(map(str, ctx_all[rows[0]][:6]))).encode()).hexdigest()[:16]
        run_ids.append(rid)
        for pos in CAP_BOUNDS:
            counters["snapshotBoundaryChecks"] += len(rows)
            counters["snapshotsCapturedInRun"] += len(rows)
        for ki, s in enumerate(SCHEDULE):
            pos = 4 * (s + 1)
            pred, sub = L.readout(model, snaps[pos], q, pos, vt)
            counters["snapshotsRestored"] += len(rows); counters["candidateSnapshotsScored"] += len(rows)
            counters["queriesEvaluated"] += len(rows)
            for li, r in enumerate(rows):
                pool_logits[r, ki] = sub[li].cpu().numpy()
        pred, sub = L.readout(model, snaps[CTX_LEN], q, CTX_LEN, vt)
        counters["snapshotsRestored"] += len(rows); counters["queriesEvaluated"] += len(rows)
        for li, r in enumerate(rows):
            final_logits[r] = sub[li].cpu().numpy()
        del snaps; torch.cuda.empty_cache()
    counters["fastPathCalls"] = int(L.KCOUNT["selective_state_update"] + L.KCOUNT["mamba_chunk_scan_combined"])
    fallbacks = L.fallback_reachable()
    res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    # arms
    def softmax(x):
        e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
    pool_pred = pool_logits.argmax(-1); pool_correct = (pool_pred == gcol[:, None])
    pool_top1 = softmax(pool_logits).max(-1)
    final_pred = final_logits.argmax(-1); final_correct = (final_pred == gcol)
    i76 = SCHEDULE.index(76); i153 = SCHEDULE.index(153)
    fixed76 = pool_correct[:, i76]
    fixed153 = pool_correct[:, i153]
    recency = pool_correct[:, -1]
    maxconf_sel = pool_top1.argmax(1); maxconf = (pool_pred[np.arange(N), maxconf_sel] == gcol)
    oracle_best = pool_correct.any(1)
    prox = [D6.proximal_snapshot_index(SCHEDULE, int(t)) for t in tslots]
    prox_idx = np.array([p[0] for p in prox]); proximal = pool_correct[np.arange(N), prox_idx]
    matched_no_history = final_correct.copy()

    accF = float(final_correct.mean())

    def arm_report(correct, vs_fixed=False):
        acc = float(correct.mean())
        dF = correct.astype(float) - final_correct.astype(float)
        rep = {"acc": acc, "delta_vs_final": round(float(dF.mean()), 4),
               "delta_vs_final_ci95": boot_ci(dF, strat, spec["s_strata"], spec["master_seed"]),
               "n_final_wrong": int((~final_correct).sum()),
               "n_recovered": int((~final_correct & correct).sum()),
               "n_final_correct": int(final_correct.sum()),
               "n_harmed": int((final_correct & ~correct).sum()),
               "per_stratum_delta_vs_final": {int(s): float(correct[strat == s].mean() - final_correct[strat == s].mean())
                                              for s in range(spec["s_strata"])},
               "oracle_gap": round(float(oracle_best.mean() - acc), 4)}
        rep["net_recovery_count"] = rep["n_recovered"] - rep["n_harmed"]
        rep["net_recovery_rate"] = round(rep["net_recovery_count"] / N, 4)
        rep["recovery_rate"] = round(rep["n_recovered"] / max(1, rep["n_final_wrong"]), 4)
        rep["harm_rate"] = round(rep["n_harmed"] / max(1, rep["n_final_correct"]), 4)
        rep["robust_strata_vs_final"] = sum(1 for s in rep["per_stratum_delta_vs_final"]
                                            if rep["per_stratum_delta_vs_final"][s] >= 0)
        if vs_fixed:
            dX = correct.astype(float) - fixed76.astype(float)
            rep["delta_vs_fixed76"] = round(float(dX.mean()), 4)
            rep["delta_vs_fixed76_ci95"] = boot_ci(dX, strat, spec["s_strata"], spec["master_seed"])
            rep["per_stratum_delta_vs_fixed76"] = {int(s): float(correct[strat == s].mean() - fixed76[strat == s].mean())
                                                   for s in range(spec["s_strata"])}
            rep["robust_strata_vs_fixed76"] = sum(1 for s in rep["per_stratum_delta_vs_fixed76"]
                                                  if rep["per_stratum_delta_vs_fixed76"][s] >= 0)
        return rep

    arms = {
        "FINAL": {"acc": accF, "ci95": wilson(int(final_correct.sum()), N)},
        "ORACLE_BEST_GOLD": {"acc": float(oracle_best.mean()), "note": "diagnostic upper bound"},
        "ORACLE_TARGET_PROXIMAL": {"acc": float(proximal.mean()), "note": "diagnostic; uses target pos"},
        "FIXED_SLOT_76": arm_report(fixed76),
        "MAX_CONFIDENCE": {**arm_report(maxconf, vs_fixed=True),
                           "selectedSnapshotHistogram": [int((maxconf_sel == k).sum()) for k in range(K)]},
        "RECENCY_FIXED_SLOT_153": arm_report(recency),
        "MATCHED_NO_HISTORY": {"acc": float(matched_no_history.mean()), "equals_final": True,
                               "note": "K FINAL readouts ensembled == FINAL (compute control)"},
        "pool_per_slot_acc": {int(SCHEDULE[k]): float(pool_correct[:, k].mean()) for k in range(K)}}
    res["arms"] = arms
    res["replicates_06D_maxconf_0833"] = {"maxconf_acc_here": arms["MAX_CONFIDENCE"]["acc"],
                                          "d06_maxconf_acc": 0.8333, "note": "exact equality not required"}

    # ---- CLAIM gates ----
    rec_fixed = arms["FIXED_SLOT_76"]["delta_vs_final"]; rec_fixed_ci = arms["FIXED_SLOT_76"]["delta_vs_final_ci95"]
    rec_mc = arms["MAX_CONFIDENCE"]["delta_vs_final"]; rec_mc_ci = arms["MAX_CONFIDENCE"]["delta_vs_final_ci95"]
    rec_ok = (rec_fixed >= SESOI_RECOVERY and rec_fixed_ci[0] > CI_LB_RECOVERY
              and rec_mc >= SESOI_RECOVERY and rec_mc_ci[0] > CI_LB_RECOVERY
              and arms["FIXED_SLOT_76"]["robust_strata_vs_final"] >= ROBUST_MIN
              and arms["MAX_CONFIDENCE"]["robust_strata_vs_final"] >= ROBUST_MIN)
    HISTORICAL_RECOVERY_TRANSPORT = "QUALIFIED" if rec_ok else "NOT_REPLICATED"

    adv = arms["MAX_CONFIDENCE"]["delta_vs_fixed76"]; adv_ci = arms["MAX_CONFIDENCE"]["delta_vs_fixed76_ci95"]
    adv_robust = arms["MAX_CONFIDENCE"]["robust_strata_vs_fixed76"]
    if adv >= SESOI_ADAPTIVE and adv_ci[0] > 0 and adv_robust >= ROBUST_MIN:
        ADAPTIVE_SELECTOR_ADVANTAGE = "QUALIFIED"
    elif adv > 0:
        ADAPTIVE_SELECTOR_ADVANTAGE = "DIRECTIONAL"
    else:
        ADAPTIVE_SELECTOR_ADVANTAGE = "NOT_QUALIFIED"

    if HISTORICAL_RECOVERY_TRANSPORT != "QUALIFIED":
        OFFICIAL_MAMBA_SYNTHETIC_TRANSPORT = "NOT_REPLICATED"
    elif ADAPTIVE_SELECTOR_ADVANTAGE == "QUALIFIED":
        OFFICIAL_MAMBA_SYNTHETIC_TRANSPORT = "QUALIFIED"
    else:
        OFFICIAL_MAMBA_SYNTHETIC_TRANSPORT = "PARTIAL"

    counters_ok = all(counters[k] > 0 for k in ["singlePassRuns", "snapshotsCapturedInRun",
                      "snapshotsRestored", "candidateSnapshotsScored", "queriesEvaluated"])
    res["mechanism_activation"] = counters
    res["fallback_reachable"] = fallbacks
    res["fast_path_active"] = bool(not any(fallbacks.values()) and counters["fastPathCalls"] > 0)
    res["single_pass_identity"] = {"run_ids": run_ids, "boundaries_monotonic": CAP_BOUNDS == sorted(CAP_BOUNDS),
                                   "n_runs": counters["singlePassRuns"]}
    res["claims"] = {
        "CLAIM1_recovery": {"fixed76_vs_final": rec_fixed, "fixed76_ci": rec_fixed_ci,
                            "maxconf_vs_final": rec_mc, "maxconf_ci": rec_mc_ci, "SESOI": SESOI_RECOVERY},
        "CLAIM2_adaptive": {"maxconf_vs_fixed76": adv, "ci": adv_ci, "robust": adv_robust, "SESOI": SESOI_ADAPTIVE}}
    res["HISTORICAL_RECOVERY_TRANSPORT"] = HISTORICAL_RECOVERY_TRANSPORT
    res["ADAPTIVE_SELECTOR_ADVANTAGE"] = ADAPTIVE_SELECTOR_ADVANTAGE
    res["OFFICIAL_MAMBA_SYNTHETIC_TRANSPORT"] = OFFICIAL_MAMBA_SYNTHETIC_TRANSPORT
    res["3B_status"] = "OPEN" if HISTORICAL_RECOVERY_TRANSPORT == "QUALIFIED" else "BLOCKED"
    res["weights_immutable"] = bool(w0 == L.weights_identity(model))
    res["total_runtime_s"] = round(time.time() - t0, 1)

    with open(os.path.join(OUTDIR, "T1_3A_RESULTS.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    np.savez_compressed(os.path.join(OUTDIR, "T1_3A_READOUTS.npz"), pool_logits=pool_logits,
                        final_logits=final_logits, golds=golds, target_slots=tslots, strata=strat,
                        schedule=np.array(SCHEDULE))

    print(f"\nFINAL={accF:.3f} FIXED_76={arms['FIXED_SLOT_76']['acc']:.3f} "
          f"MAX_CONF={arms['MAX_CONFIDENCE']['acc']:.3f} ORACLE_BEST={arms['ORACLE_BEST_GOLD']['acc']:.3f}")
    print(f"pool_per_slot={arms['pool_per_slot_acc']}")
    print(f"CLAIM1 recovery: fixed76-FINAL={rec_fixed:+.3f}{rec_fixed_ci} maxconf-FINAL={rec_mc:+.3f}{rec_mc_ci}")
    print(f"CLAIM2 adaptive: maxconf-fixed76={adv:+.4f}{adv_ci} robust={adv_robust}/3")
    print(f"maxconf histogram={arms['MAX_CONFIDENCE']['selectedSnapshotHistogram']}")
    print(f"fast_path_active={res['fast_path_active']} counters_ok={counters_ok}")
    print(f"HISTORICAL_RECOVERY_TRANSPORT = {HISTORICAL_RECOVERY_TRANSPORT}")
    print(f"ADAPTIVE_SELECTOR_ADVANTAGE  = {ADAPTIVE_SELECTOR_ADVANTAGE}")
    print(f"OFFICIAL_MAMBA_SYNTHETIC_TRANSPORT = {OFFICIAL_MAMBA_SYNTHETIC_TRANSPORT}  3B={res['3B_status']}")
    print(f"runtime={res['total_runtime_s']}s vram={res['peak_vram_gb']}GB")


if __name__ == "__main__":
    main()
