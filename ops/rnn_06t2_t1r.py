#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06T2-T1R — fresh recovery confirmation. modes: narrow | wide_calib | wide_qual.

Single-pass in-run capture on the official Mamba-2 fast path (fixed batch). MAX_CONFIDENCE is FROZEN.
Wide fixed-control TIE policy: carry all fixed snapshots within TAU_TIE of the best calibration acc;
adaptive gate must beat the STRONGEST carried control on qualification. Full recovery/harm tables for
every fixed control and MAX_CONFIDENCE; mechanism-activation counters; boundary-check sample.
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
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06T2")
TOK_ID = "EleutherAI/gpt-neox-20b"
POOL_SEED = 20260817
BATCH = 16
M = 192
SCHEDULE = D6.schedule_slots(M, 4)                 # [38,76,115,153]
CTX_LEN = 4 * M
CAP_BOUNDS = [4 * (s + 1) for s in SCHEDULE] + [CTX_LEN]
# frozen thresholds (see T1R_PRE_REGISTRATION.md)
SESOI_RECOVERY = 0.15
SESOI_ADAPTIVE = 0.05
CI_LB_RECOVERY = 0.05
ROBUST_MIN = 3
TAU_TIE = 0.02
NARROW_FIXED_SLOT = 76


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
    rb = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 0x71])))
    idxs = {s: np.where(strat == s)[0] for s in range(s_strata)}
    b = []
    for _ in range(2000):
        idx = np.concatenate([idxs[s][rb.integers(0, len(idxs[s]), size=len(idxs[s]))] for s in idxs])
        b.append(float(delta[idx].mean()))
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def capture_readouts(model, examples, pools, vt, V):
    """Single-pass capture + K+1 restore/readouts per example. Returns pool_logits(N,K,V),
    final_logits(N,V), golds, target_slots, counters, boundary_sample."""
    N = len(examples); K = len(SCHEDULE)
    ctx_all, query_all, golds, tslots = [], [], [], []
    for e in examples:
        toks, gold = D6.materialize_d0(e, M, pools)
        ctx_all.append(toks[:CTX_LEN]); query_all.append(toks[-2:]); golds.append(gold); tslots.append(e["target_slot"])
    pool_logits = np.zeros((N, K, V), np.float32); final_logits = np.zeros((N, V), np.float32)
    counters = {"singlePassRuns": 0, "snapshotsCapturedInRun": 0, "snapshotsRestored": 0,
                "candidateSnapshotsScored": 0, "queriesEvaluated": 0,
                "snapshotBoundaryChecks": 0, "snapshotBoundaryFailures": 0}
    boundary_sample = []
    L.reset_counters()
    for b in range(0, N, BATCH):
        rows = list(range(b, min(b + BATCH, N)))
        ctx = torch.tensor([ctx_all[r] for r in rows], device=L.DEVICE, dtype=torch.long)
        q = torch.tensor([query_all[r] for r in rows], device=L.DEVICE, dtype=torch.long)
        snaps = L.run_trajectory(model, ctx, CAP_BOUNDS)
        counters["singlePassRuns"] += 1
        counters["snapshotsCapturedInRun"] += len(rows) * (K + 1)
        # boundary-check on the FIRST batch only: independent same-path replay must match at every boundary
        if b == 0:
            replay = L.run_trajectory(model, ctx, CAP_BOUNDS)
            run_id = hashlib.sha256(("T1Rqual|" + ",".join(map(str, ctx_all[0][:8]))).encode()).hexdigest()[:16]
            for oi, pos in enumerate(CAP_BOUNDS):
                h = L.state_hash(snaps[pos]); hr = L.state_hash(replay[pos])
                match = (h == hr)
                counters["snapshotBoundaryChecks"] += 1
                if not match:
                    counters["snapshotBoundaryFailures"] += 1
                boundary_sample.append({"runId": run_id, "exampleId": rows[0], "boundary": pos,
                                        "captured_state_hash": h[:24], "replay_state_hash": hr[:24],
                                        "match": bool(match), "restore_result": "ok"})
            del replay
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
    return pool_logits, final_logits, np.array(golds), np.array(tslots), counters, boundary_sample


def recovery_harm(correct, final_correct, strat, s_strata, seed):
    N = len(correct)
    n_final_wrong = int((~final_correct).sum())
    n_final_correct = int(final_correct.sum())
    n_recovered = int((~final_correct & correct).sum())
    n_harmed = int((final_correct & ~correct).sum())
    d = correct.astype(float) - final_correct.astype(float)
    per = {int(s): round(float(correct[strat == s].mean() - final_correct[strat == s].mean()), 4)
           for s in range(s_strata)}
    return {"N": N, "acc": round(float(correct.mean()), 4),
            "n_final_wrong": n_final_wrong, "n_recovered": n_recovered,
            "recovery_rate": round(n_recovered / n_final_wrong, 4) if n_final_wrong else None,
            "n_final_correct": n_final_correct, "n_harmed": n_harmed,
            "harm_rate": round(n_harmed / n_final_correct, 4) if n_final_correct else None,
            "net_recovery_count": n_recovered - n_harmed,
            "net_recovery_rate": round((n_recovered - n_harmed) / N, 4),
            "accuracy_delta_vs_final": round(float(d.mean()), 4),
            "ci95_paired_stratified": boot_ci(d, strat, s_strata, seed),
            "per_stratum_delta": per, "robust_strata": sum(1 for s in per if per[s] >= 0)}


def paired_vs(correct_a, correct_b, strat, s_strata, seed):
    d = correct_a.astype(float) - correct_b.astype(float)
    per = {int(s): round(float(correct_a[strat == s].mean() - correct_b[strat == s].mean()), 4)
           for s in range(s_strata)}
    return {"delta": round(float(d.mean()), 4), "ci95": boot_ci(d, strat, s_strata, seed),
            "per_stratum": per, "robust_strata": sum(1 for s in per if per[s] >= 0)}


def load_model_pools():
    L.install_counters()
    tok = AutoTokenizer.from_pretrained(TOK_ID)
    pools, _ = L.build_official_pools(tok, POOL_SEED)
    vset = sorted(set(pools["scored_vals"]))
    vt = torch.tensor(vset, device=L.DEVICE, dtype=torch.long)
    model = L.load_model()
    return model, pools, vset, vt


def base_identity(runner, prereg="T1R_PRE_REGISTRATION.md"):
    import mamba_ssm
    return {"runner_git_blob": git("hash-object", runner),
            "runner_dirty": git("status", "--porcelain", "--", runner),
            "git_head": git("rev-parse", "HEAD"), "repo_id": L.REPO_ID, "revision": L.REVISION,
            "mamba_ssm": mamba_ssm.__version__,
            "protocol_sha256": sha256_file(os.path.join(OUTDIR, prereg))}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "wide_qual"
    t0 = time.time(); runner = os.path.abspath(__file__)
    model, pools, vset, vt = load_model_pools()
    V = len(vset); w0 = L.weights_identity(model); K = len(SCHEDULE)

    if mode == "narrow":
        spec = json.load(open(os.path.join(OUTDIR, "T1R_NARROW_QUAL_SPEC.json")))
        ex = spec["examples"]; N = len(ex); strat = np.array([e["stratum"] for e in ex]); s_strata = spec["s_strata"]
        seed = spec["master_seed"]
        pl, fl, golds, tslots, counters, bsample = capture_readouts(model, ex, pools, vt, V)
        gcol = np.array([vset.index(int(g)) for g in golds])
        pool_pred = pl.argmax(-1); pool_correct = (pool_pred == gcol[:, None]); pool_top1 = softmax(pl).max(-1)
        final_correct = (fl.argmax(-1) == gcol)
        fixed_k = SCHEDULE.index(NARROW_FIXED_SLOT)
        fixed76 = pool_correct[:, fixed_k]
        maxconf_sel = pool_top1.argmax(1); maxconf = (pool_pred[np.arange(N), maxconf_sel] == gcol)
        oracle_best = pool_correct.any(1)
        prox = [D6.proximal_snapshot_index(SCHEDULE, int(t)) for t in tslots]
        proximal = pool_correct[np.arange(N), np.array([p[0] for p in prox])]
        rh_fixed = recovery_harm(fixed76, final_correct, strat, s_strata, seed)
        rh_maxconf = recovery_harm(maxconf, final_correct, strat, s_strata, seed)
        adaptive_vs_fixed = paired_vs(maxconf, fixed76, strat, s_strata, seed)
        hist_recovery = ("QUALIFIED" if (rh_fixed["accuracy_delta_vs_final"] >= SESOI_RECOVERY
                         and rh_fixed["ci95_paired_stratified"][0] > CI_LB_RECOVERY
                         and rh_fixed["robust_strata"] >= ROBUST_MIN) else
                         ("PARTIAL" if rh_fixed["accuracy_delta_vs_final"] >= SESOI_RECOVERY else "NOT_REPLICATED"))
        adaptive = ("QUALIFIED" if (adaptive_vs_fixed["delta"] >= SESOI_ADAPTIVE
                    and adaptive_vs_fixed["ci95"][0] > 0 and adaptive_vs_fixed["robust_strata"] >= ROBUST_MIN)
                    else ("DIRECTIONAL" if adaptive_vs_fixed["delta"] > 0 else "NOT_QUALIFIED"))
        res = {"packet": "RNN-06T2-T1R-NARROW", "band": spec["band"], "schedule": SCHEDULE,
               "qualificationSetSha256": spec["setSha256"], "disjointness_proof": spec["disjointness_proof"],
               "executed_source_identity": base_identity(runner), "model_weights_sentinel": w0,
               "arms": {"FINAL": {"acc": round(float(final_correct.mean()), 4),
                                  "ci95": wilson(int(final_correct.sum()), N)},
                        "FIXED_SLOT_76": {"acc": rh_fixed["acc"], "recovery_harm_vs_final": rh_fixed},
                        "MAX_CONFIDENCE": {"acc": rh_maxconf["acc"], "recovery_harm_vs_final": rh_maxconf,
                                           "vs_fixed_slot_76": adaptive_vs_fixed,
                                           "selectedSnapshotHistogram": [int((maxconf_sel == k).sum()) for k in range(K)]},
                        "ORACLE_BEST_GOLD": {"acc": round(float(oracle_best.mean()), 4), "note": "diagnostic"},
                        "ORACLE_TARGET_PROXIMAL": {"acc": round(float(proximal.mean()), 4), "note": "diagnostic"}},
               "all_fixed_slot_acc": {int(SCHEDULE[k]): round(float(pool_correct[:, k].mean()), 4) for k in range(K)},
               "mechanism_activation": counters, "boundary_check_sample": bsample,
               "HISTORICAL_RECOVERY_NARROW": hist_recovery, "ADAPTIVE_SELECTION_NARROW": adaptive,
               "fast_path_active": bool(not any(L.fallback_reachable().values()) and L.KCOUNT["selective_state_update"] > 0),
               "weights_immutable": bool(w0 == L.weights_identity(model)),
               "total_runtime_s": round(time.time() - t0, 1)}
        json.dump(res, open(os.path.join(OUTDIR, "T1R_NARROW_RESULTS.json"), "w"), indent=2, default=str)
        np.savez_compressed(os.path.join(OUTDIR, "T1R_NARROW_READOUTS.npz"), pool_logits=pl, final_logits=fl,
                            golds=golds, target_slots=tslots, strata=strat, schedule=np.array(SCHEDULE),
                            vset=np.array(vset), final_correct=final_correct, maxconf_correct=maxconf,
                            fixed76_correct=fixed76)
        print(f"[narrow] FINAL={res['arms']['FINAL']['acc']} FIXED76={rh_fixed['acc']} "
              f"MAXCONF={rh_maxconf['acc']} ORACLE={res['arms']['ORACLE_BEST_GOLD']['acc']}")
        print(f"[narrow] all_fixed={res['all_fixed_slot_acc']}")
        print(f"[narrow] FIXED76 vs FINAL: {rh_fixed['accuracy_delta_vs_final']:+.3f}{rh_fixed['ci95_paired_stratified']} "
              f"net={rh_fixed['net_recovery_count']} robust={rh_fixed['robust_strata']}/{s_strata}")
        print(f"[narrow] MAXCONF vs FIXED76: {adaptive_vs_fixed['delta']:+.3f}{adaptive_vs_fixed['ci95']}")
        print(f"HISTORICAL_RECOVERY_NARROW={hist_recovery} ADAPTIVE_SELECTION_NARROW={adaptive}")
        return

    if mode == "wide_calib":
        spec = json.load(open(os.path.join(OUTDIR, "T1R_WIDE_CALIB_SPEC.json")))
        ex = spec["examples"]; N = len(ex)
        pl, fl, golds, tslots, counters, _ = capture_readouts(model, ex, pools, vt, V)
        gcol = np.array([vset.index(int(g)) for g in golds])
        pool_correct = (pl.argmax(-1) == gcol[:, None])
        fixed_acc = {int(SCHEDULE[k]): float(pool_correct[:, k].mean()) for k in range(K)}
        best_acc = max(fixed_acc.values())
        # carry ALL within TAU_TIE; deterministic best = highest acc, tie -> smallest slot
        carried = sorted([s for s, a in fixed_acc.items() if a >= best_acc - TAU_TIE])
        best_slot = sorted(fixed_acc.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        maxconf_sel = softmax(pl).max(-1).argmax(1)
        maxconf_acc = float((pl.argmax(-1)[np.arange(N), maxconf_sel] == gcol).mean())
        out = {"packet": "RNN-06T2-T1R-WIDE-CALIB", "calibrationSetSha256": spec["setSha256"],
               "fixed_slot_acc": {k: round(v, 4) for k, v in fixed_acc.items()},
               "TAU_TIE": TAU_TIE, "best_acc": round(best_acc, 4),
               "CARRIED_FIXED_CONTROLS": carried, "deterministic_best_slot": best_slot,
               "maxconf_acc_calib": round(maxconf_acc, 4),
               "final_acc_calib": round(float((fl.argmax(-1) == gcol).mean()), 4),
               "tie_break_rule": "highest calib acc; ties within TAU_TIE all carried; single-best tie -> smallest slot",
               "counters": counters}
        json.dump(out, open(os.path.join(OUTDIR, "T1R_WIDE_CALIBRATION.json"), "w"), indent=2, default=str)
        print(f"[wide_calib] fixed_slot_acc={out['fixed_slot_acc']}")
        print(f"[wide_calib] CARRIED_FIXED_CONTROLS={carried} best_slot={best_slot} (TAU_TIE={TAU_TIE})")
        return

    # ---------- wide_qual ----------
    calib = json.load(open(os.path.join(OUTDIR, "T1R_WIDE_CALIBRATION.json")))
    carried = [int(s) for s in calib["CARRIED_FIXED_CONTROLS"]]
    spec = json.load(open(os.path.join(OUTDIR, "T1R_WIDE_QUAL_SPEC.json")))
    ex = spec["examples"]; N = len(ex); strat = np.array([e["stratum"] for e in ex]); s_strata = spec["s_strata"]
    seed = spec["master_seed"]
    pl, fl, golds, tslots, counters, bsample = capture_readouts(model, ex, pools, vt, V)
    gcol = np.array([vset.index(int(g)) for g in golds])
    pool_pred = pl.argmax(-1); pool_correct = (pool_pred == gcol[:, None]); pool_top1 = softmax(pl).max(-1)
    final_correct = (fl.argmax(-1) == gcol)
    maxconf_sel = pool_top1.argmax(1); maxconf = (pool_pred[np.arange(N), maxconf_sel] == gcol)
    oracle_best = pool_correct.any(1)
    prox = [D6.proximal_snapshot_index(SCHEDULE, int(t)) for t in tslots]
    proximal = pool_correct[np.arange(N), np.array([p[0] for p in prox])]

    fixed_correct = {int(SCHEDULE[k]): pool_correct[:, k] for k in range(K)}
    fixed_acc_qual = {s: round(float(fixed_correct[s].mean()), 4) for s in fixed_correct}
    # strongest carried control on QUALIFICATION
    strongest_carried = sorted(carried, key=lambda s: (-fixed_acc_qual[s], s))[0]

    rh_all_fixed = {int(s): recovery_harm(fixed_correct[s], final_correct, strat, s_strata, seed) for s in fixed_correct}
    rh_maxconf = recovery_harm(maxconf, final_correct, strat, s_strata, seed)
    adaptive_vs_strongest = paired_vs(maxconf, fixed_correct[strongest_carried], strat, s_strata, seed)
    # post-hoc descriptive: best fixed observed on qualification
    best_fixed_qual = sorted(fixed_acc_qual.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    posthoc_vs_bestqual = paired_vs(maxconf, fixed_correct[best_fixed_qual], strat, s_strata, seed)

    wide_recovery = ("QUALIFIED" if (rh_maxconf["accuracy_delta_vs_final"] >= SESOI_RECOVERY
                     and rh_maxconf["ci95_paired_stratified"][0] > CI_LB_RECOVERY
                     and rh_maxconf["robust_strata"] >= ROBUST_MIN) else
                     ("PARTIAL" if rh_maxconf["accuracy_delta_vs_final"] >= SESOI_RECOVERY else "NOT_REPLICATED"))
    adaptive = ("QUALIFIED" if (adaptive_vs_strongest["delta"] >= SESOI_ADAPTIVE
                and adaptive_vs_strongest["ci95"][0] > 0 and adaptive_vs_strongest["robust_strata"] >= ROBUST_MIN)
                else ("DIRECTIONAL" if adaptive_vs_strongest["delta"] > 0 else "NOT_QUALIFIED"))

    res = {"packet": "RNN-06T2-T1R-WIDE", "band": spec["band"], "regions": spec["regions"], "schedule": SCHEDULE,
           "qualificationSetSha256": spec["setSha256"], "disjointness_proof": spec["disjointness_proof"],
           "calibrationSetSha256": calib["calibrationSetSha256"],
           "CARRIED_FIXED_CONTROLS": carried, "strongest_carried_on_qual": strongest_carried,
           "TAU_TIE": TAU_TIE, "executed_source_identity": base_identity(runner), "model_weights_sentinel": w0,
           "arms": {"FINAL": {"acc": round(float(final_correct.mean()), 4), "ci95": wilson(int(final_correct.sum()), N)},
                    "ORACLE_BEST_GOLD": {"acc": round(float(oracle_best.mean()), 4), "note": "diagnostic"},
                    "ORACLE_TARGET_PROXIMAL": {"acc": round(float(proximal.mean()), 4), "note": "diagnostic"},
                    "MAX_CONFIDENCE": {"acc": rh_maxconf["acc"], "recovery_harm_vs_final": rh_maxconf,
                                       "vs_strongest_carried_fixed": {"strongest_carried_slot": strongest_carried,
                                                                      **adaptive_vs_strongest},
                                       "POST_HOC_DESCRIPTIVE_vs_best_fixed_on_qual": {"best_fixed_qual_slot": best_fixed_qual,
                                                                                     **posthoc_vs_bestqual},
                                       "selectedSnapshotHistogram": [int((maxconf_sel == k).sum()) for k in range(K)]}},
           "all_fixed_slot_acc_qual": fixed_acc_qual,
           "recovery_harm_per_fixed_control": rh_all_fixed,
           "per_region": {int(s): {"final": round(float(final_correct[strat == s].mean()), 4),
                                   "maxconf": round(float(maxconf[strat == s].mean()), 4),
                                   **{f"slot{SCHEDULE[k]}": round(float(pool_correct[strat == s, k].mean()), 4) for k in range(K)}}
                          for s in range(s_strata)},
           "mechanism_activation": counters, "boundary_check_sample": bsample,
           "WIDE_TARGET_RECOVERY_T1R": wide_recovery, "ADAPTIVE_SELECTION_T1R": adaptive,
           "fast_path_active": bool(not any(L.fallback_reachable().values()) and L.KCOUNT["selective_state_update"] > 0),
           "weights_immutable": bool(w0 == L.weights_identity(model)),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
           "total_runtime_s": round(time.time() - t0, 1)}
    json.dump(res, open(os.path.join(OUTDIR, "T1R_WIDE_RESULTS.json"), "w"), indent=2, default=str)
    np.savez_compressed(os.path.join(OUTDIR, "T1R_WIDE_READOUTS.npz"), pool_logits=pl, final_logits=fl,
                        golds=golds, target_slots=tslots, strata=strat, schedule=np.array(SCHEDULE),
                        vset=np.array(vset), final_correct=final_correct, maxconf_correct=maxconf,
                        **{f"fixed{SCHEDULE[k]}_correct": pool_correct[:, k] for k in range(K)})
    # scored-value token mapping for the bundle (gold-column index <-> token id)
    json.dump({"vset_token_ids": vset, "note": "column index i in *_logits/*_correct maps to token id vset[i]; "
               "gold column = vset.index(gold_token_id)"},
              open(os.path.join(OUTDIR, "T1R_SCORED_VALUE_TOKEN_MAP.json"), "w"), indent=2)

    print(f"[wide] CARRIED={carried} strongest_on_qual=slot{strongest_carried}")
    print(f"[wide] FINAL={res['arms']['FINAL']['acc']} MAXCONF={rh_maxconf['acc']} "
          f"ORACLE={res['arms']['ORACLE_BEST_GOLD']['acc']} all_fixed={fixed_acc_qual}")
    print(f"[wide] MAXCONF vs FINAL: {rh_maxconf['accuracy_delta_vs_final']:+.3f}"
          f"{rh_maxconf['ci95_paired_stratified']} net={rh_maxconf['net_recovery_count']} "
          f"robust={rh_maxconf['robust_strata']}/{s_strata}")
    print(f"[wide] MAXCONF vs strongest_carried(slot{strongest_carried}): {adaptive_vs_strongest['delta']:+.3f}"
          f"{adaptive_vs_strongest['ci95']} robust={adaptive_vs_strongest['robust_strata']}/{s_strata} "
          f"hist={res['arms']['MAX_CONFIDENCE']['selectedSnapshotHistogram']}")
    print(f"WIDE_TARGET_RECOVERY_T1R={wide_recovery} ADAPTIVE_SELECTION_T1R={adaptive}")


if __name__ == "__main__":
    main()
