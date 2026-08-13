#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-07A-BRIDGE-R1 — TRUE in-run historical recovery requalification.

Load-bearing correction: recovery states are captured from ONE canonical qualified single-pass
trajectory (ops/rnn_06t_lib.run_trajectory) at boundaries 25/50/75/90/FINAL, NOT independent prefix
re-prefills. Fresh, disjoint, declared-stratified recovery set (>=64). Frozen 512-tok SHORT eligibility.
Frozen MAX_CONFIDENCE. ORACLE_HISTORICAL_ONLY vs ORACLE_ALL kept distinct. Temporal identity + same-path
replay hash check. See R1_PRE_REGISTRATION.md. Nothing pushed.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_07a_bridge_lib as B   # noqa: E402  (make_context, load_book_tokens, _subst, choice/stem builders)
import rnn_07a_lib as A          # noqa: E402  (prefill_state, readout_from_state, pad_choices, wilson, bootstrap, enc)
import rnn_06t_lib as L          # noqa: E402  (run_trajectory, slice_state, state_hash_row, load_model, counters)

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-07A")
NEEDLES = B.NEEDLES

R1_POOL_SEED = 20261500
R1_SHORT_FILLER_BASE = 20261501000
R1_RECOVERY_FILLER_BASE = 20261510000
R1_SAMPLE_SEED = 20261501
R1_BOOT_SEED = 20261502
N_CHAR = 6
MAX_POOL = 168
N_RECOVERY = 64
SHORT_TOKENS = 512
LEN32 = 32000
DEPTH = 0.15
B_CAP = 32
REPLAY_SUBSET_N = 8
PROGRESS = [0.25, 0.50, 0.75, 0.90, 1.00]
SNAP_NAMES = ["SNAP_25", "SNAP_50", "SNAP_75", "SNAP_90", "FINAL"]
REC_EFFECT_MIN = 0.05
ADAPT_MIN = 0.05
PRESENCE_MIN = 0.05
PRESENCE_FRAC_MIN = 0.20
MIN_N = 48
RUNTIME_GUARD_S = 90 * 60

# historical recovery set identity (for disjointness proof)
HIST_POOL_SEED = 20261400
HIST_N_CHAR = 4
HIST_MAX_POOL = 112
HIST_RECOVERY_FILLER_BASE = 20261400 + 2000 + 2 * 100000   # CELL_SEED["32K"]=2 -> 20463400
HIST_MAX_LONG_EVAL = 90
HIST_MAX_RECOVERY = 48


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


def sha_str(s):
    return hashlib.sha256(s.encode()).hexdigest()


def tok_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def build_pool_meta(tok, seed, n_char, max_pool):
    """Deterministic pool of (needle,test,char-assignment) examples. No context yet."""
    needles = json.load(open(NEEDLES))
    rng = np.random.default_rng(seed)
    pool = []
    for e in needles:
        cset = e["character_set"]; qtmpl = e["questions"]["direct"]
        for tk, tv in e["tests"].items():
            args = tv["input_args"]
            for _ in range(n_char):
                perm = rng.permutation(len(cset))
                char = cset[perm[0]]; distractors = [cset[perm[1]], cset[perm[2]], cset[perm[3]]]
                opts = [char] + distractors
                order = rng.permutation(4)
                options = [opts[i] for i in order]; gold = int(np.where(order == 0)[0][0])
                needle_text = B._subst(e["needle"], char, args); question = B._subst(qtmpl, char, args)
                pool.append({"needle_id": e["id"], "reasoning_type": e["reasoning_type"], "test": tk,
                             "char": char, "options": options, "gold": gold,
                             "needle_text": needle_text, "question": question,
                             "option_order": "|".join(options),
                             "needle_ids": A.enc(tok, " " + needle_text.strip() + " "),
                             "stem_ids": A.build_query_stem(tok, question),
                             "choice_ids": A.build_choice_tokens(tok, options)})
                if len(pool) >= max_pool:
                    return pool
    return pool


def uid_of(ex, filler_seed, ctx_ids):
    parts = [ex["needle_id"], ex["test"], ex["char"], ex["option_order"],
             sha_str(ex["needle_text"]), sha_str(ex["question"]), str(filler_seed), tok_sha(ctx_ids)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def historical_recovery_uids(tok, book):
    """Regenerate the historical first-48 recovery example identities for disjointness proof."""
    pool = build_pool_meta(tok, HIST_POOL_SEED, HIST_N_CHAR, HIST_MAX_POOL)
    short = json.load(open(os.path.join(OUTDIR, "BRIDGE_SHORT_RESULTS.json")))["rows"]
    n = min(len(pool), len(short))
    eligible = [k for k in range(n) if short[k]["correct"] == 1][:HIST_MAX_LONG_EVAL][:HIST_MAX_RECOVERY]
    uids = set()
    for k in eligible:
        fs = HIST_RECOVERY_FILLER_BASE + k
        ctx, _ = B.make_context(book, pool[k]["needle_ids"], LEN32, DEPTH, filler_seed=fs)
        uids.add(uid_of(pool[k], fs, ctx))
    return uids


@torch.no_grad()
def short_correct(model, tok, ex, k):
    ctx, _ = B.make_context(book_g, ex["needle_ids"], SHORT_TOKENS, DEPTH,
                            filler_seed=R1_SHORT_FILLER_BASE + k)
    pred, _ = B.eval_state_from_ctx(model, ctx, ex, SHORT_TOKENS + 512)
    return int(pred == ex["gold"])


def stratified_sample(eligible_idx, pool, n_target, seed):
    """Deterministic stratified draw across needle_id x reasoning_type."""
    rng = np.random.default_rng(seed)
    strata = {}
    for k in eligible_idx:
        key = (pool[k]["needle_id"], pool[k]["reasoning_type"])
        strata.setdefault(key, []).append(k)
    keys = sorted(strata.keys())
    total = len(eligible_idx)
    # proportional allocation with largest-remainder
    raw = {key: n_target * len(strata[key]) / total for key in keys}
    alloc = {key: int(np.floor(raw[key])) for key in keys}
    rem = n_target - sum(alloc.values())
    order = sorted(keys, key=lambda key: (-(raw[key] - alloc[key]), key))
    for key in order[:rem]:
        alloc[key] += 1
    selected = []
    for key in keys:
        pool_k = sorted(strata[key])
        take = min(alloc[key], len(pool_k))
        idx = rng.permutation(len(pool_k))[:take]
        selected.extend(pool_k[i] for i in idx)
    return sorted(selected)


def main():
    t0 = time.time(); runner = os.path.abspath(__file__)
    L.install_counters()
    import mamba_ssm  # noqa
    global book_g
    tok = A.load_tokenizer()
    model = L.load_model()
    book_g = B.load_book_tokens(tok)
    pool = build_pool_meta(tok, R1_POOL_SEED, N_CHAR, MAX_POOL)

    # ---- SHORT eligibility (frozen rule) ----
    short = [short_correct(model, tok, ex, k) for k, ex in enumerate(pool)]
    eligible = [k for k in range(len(pool)) if short[k] == 1]
    from collections import Counter
    rt_dist = Counter(pool[k]["reasoning_type"] for k in eligible)
    nd_dist = Counter(pool[k]["needle_id"] for k in eligible)
    print(f"[r1] pool={len(pool)} short_correct={len(eligible)}")

    cap = min(N_RECOVERY, len(eligible))
    selected = stratified_sample(eligible, pool, cap, R1_SAMPLE_SEED) if len(eligible) > cap else sorted(eligible)

    # ---- fresh-set identity + disjointness ----
    sel_records, sel_uids = [], []
    ctx_cache = {}
    for k in selected:
        fs = R1_RECOVERY_FILLER_BASE + k
        ctx, npos = B.make_context(book_g, pool[k]["needle_ids"], LEN32, DEPTH, filler_seed=fs)
        ctx_cache[k] = ctx
        u = uid_of(pool[k], fs, ctx)
        sel_uids.append(u)
        sel_records.append({"pool_idx": k, "uid": u, "needle_id": pool[k]["needle_id"],
                            "reasoning_type": pool[k]["reasoning_type"], "test": pool[k]["test"],
                            "char": pool[k]["char"], "gold": pool[k]["gold"], "filler_seed": fs,
                            "needle_token_pos": npos, "n_ctx": len(ctx)})
    hist_uids = historical_recovery_uids(tok, book_g)
    overlap = set(sel_uids) & hist_uids
    r1_set_sha = hashlib.sha256("|".join(sorted(sel_uids)).encode()).hexdigest()
    sel_set_sha = hashlib.sha256("|".join(str(k) for k in selected).encode()).hexdigest()
    print(f"[r1] selected={len(selected)} disjoint_from_historical={len(overlap)==0} (overlap={len(overlap)})")

    boundaries = [int(round(p * LEN32)) for p in PROGRESS]
    max_seqlen_read = LEN32 + 600

    # ---- TRUE in-run capture (single trajectory) + readout ----
    rows = []
    captured_hashes = {}   # uid -> {boundary: state_hash_row}
    blocked = False
    for bstart in range(0, len(selected), B_CAP):
        if time.time() - t0 > RUNTIME_GUARD_S:
            blocked = True; break
        batch_k = selected[bstart:bstart + B_CAP]
        ids = torch.tensor([ctx_cache[k] for k in batch_k], device=L.DEVICE, dtype=torch.long)
        snaps = L.run_trajectory(model, ids, boundaries)     # ONE canonical trajectory; in-run captures
        for row_i, k in enumerate(batch_k):
            ex = pool[k]; g = ex["gold"]
            stem_t = torch.tensor([ex["stem_ids"]], device=L.DEVICE, dtype=torch.long)
            choices_pad = A.pad_choices([ex["choice_ids"]])
            preds, confs, hashes = [], [], {}
            for bi, b in enumerate(boundaries):
                st = L.slice_state(snaps[b], [row_i])
                r = A.readout_from_state(model, st, b, stem_t, choices_pad, max_seqlen_read)
                preds.append(int(r["content_pred"][0])); confs.append(float(r["confidence"][0]))
                hashes[SNAP_NAMES[bi]] = L.state_hash_row(snaps[b], row_i)
            final_pred = preds[-1]; mci = int(np.argmax(confs))
            hist_correct = [int(preds[i] == g) for i in range(4)]
            rows.append({"uid": sel_records[[r_["pool_idx"] for r_ in sel_records].index(k)]["uid"],
                         "pool_idx": k, "needle_id": ex["needle_id"], "reasoning_type": ex["reasoning_type"],
                         "gold": g, "snap_preds": preds, "snap_confs": confs,
                         "snap_correct": [int(p == g) for p in preds], "final_correct": int(final_pred == g),
                         "max_conf_idx": mci, "max_conf_correct": int(preds[mci] == g),
                         "oracle_hist_only_correct": int(any(hist_correct)),
                         "oracle_all_correct": int(any(p == g for p in preds)),
                         "boundary_hashes": hashes})
            captured_hashes[rows[-1]["uid"]] = hashes
        del snaps; torch.cuda.empty_cache()
        print(f"[r1] captured {min(bstart+B_CAP,len(selected))}/{len(selected)} elapsed={time.time()-t0:.1f}s")

    if blocked:
        out = {"packet": "RNN-07A-BRIDGE-R1", "status": "BLOCKED_BY_RUNTIME_BUDGET",
               "elapsed_s": round(time.time() - t0, 1), "n_selected": len(selected), "n_captured": len(rows),
               "TRUE_IN_RUN_RECOVERY_R1": "BLOCKED_BY_RUNTIME_BUDGET"}
        json.dump(out, open(os.path.join(OUTDIR, "R1_RESULTS.json"), "w"), indent=2, default=str)
        print("[r1] BLOCKED_BY_RUNTIME_BUDGET"); return

    # ---- same-path replay hash check on preregistered subset ----
    replay_k = selected[:REPLAY_SUBSET_N]
    ids = torch.tensor([ctx_cache[k] for k in replay_k], device=L.DEVICE, dtype=torch.long)
    snaps2 = L.run_trajectory(model, ids, boundaries)
    replay_records, replay_ok = [], True
    for row_i, k in enumerate(replay_k):
        u = sel_records[[r_["pool_idx"] for r_ in sel_records].index(k)]["uid"]
        for bi, b in enumerate(boundaries):
            h2 = L.state_hash_row(snaps2[b], row_i); h1 = captured_hashes[u][SNAP_NAMES[bi]]
            match = (h1 == h2); replay_ok = replay_ok and match
            replay_records.append({"uid": u, "snapshot": SNAP_NAMES[bi], "match": match})
    del snaps2; torch.cuda.empty_cache()
    print(f"[r1] replay boundary-hash match: {replay_ok} ({sum(r['match'] for r in replay_records)}/{len(replay_records)})")

    # ---- metrics ----
    n = len(rows)
    final_c = [r["final_correct"] for r in rows]
    maxc_c = [r["max_conf_correct"] for r in rows]
    snap_acc = {SNAP_NAMES[i]: float(np.mean([r["snap_correct"][i] for r in rows])) for i in range(5)}
    oracle_hist = [r["oracle_hist_only_correct"] for r in rows]
    oracle_all = [r["oracle_all_correct"] for r in rows]

    def decomp(arm_correct):
        nfw = sum(1 for r in rows if r["final_correct"] == 0)
        nfc = n - nfw
        recovered = sum(1 for i, r in enumerate(rows) if r["final_correct"] == 0 and arm_correct[i] == 1)
        harmed = sum(1 for i, r in enumerate(rows) if r["final_correct"] == 1 and arm_correct[i] == 0)
        pt, lo, hi = A.paired_bootstrap_delta(arm_correct, final_c, n_boot=2000, seed=R1_BOOT_SEED)
        return {"n_final_wrong": nfw, "n_final_correct": nfc, "n_recovered": recovered, "n_harmed": harmed,
                "net_recovery": recovered - harmed, "accuracy_delta_vs_final": pt, "ci_lo": lo, "ci_hi": hi}

    fixed_decomp = {SNAP_NAMES[i]: decomp([r["snap_correct"][i] for r in rows]) for i in range(4)}
    maxc_decomp = decomp(maxc_c)
    fw = [r for r in rows if r["final_correct"] == 0]
    hist_recov_fw = sum(1 for r in fw if any(r["snap_correct"][i] == 1 for i in range(4)))
    hrr, hrr_lo, hrr_hi = A.wilson(hist_recov_fw, len(fw)) if fw else (0.0, 0.0, 0.0)
    oh_pt, oh_lo, oh_hi = A.paired_bootstrap_delta(oracle_hist, final_c, n_boot=2000, seed=R1_BOOT_SEED)

    # ---- gates ----
    if n < MIN_N:
        RECOV = "INCONCLUSIVE"; MAXC = "INCONCLUSIVE"; PRES = "INCONCLUSIVE"
    else:
        pos_fixed = [SNAP_NAMES[i] for i in range(4)
                     if fixed_decomp[SNAP_NAMES[i]]["ci_lo"] > 0 and fixed_decomp[SNAP_NAMES[i]]["net_recovery"] > 0
                     and fixed_decomp[SNAP_NAMES[i]]["accuracy_delta_vs_final"] >= REC_EFFECT_MIN]
        RECOV = "POSITIVE_SIGNAL" if pos_fixed else "NO_NET_SIGNAL"
        d = maxc_decomp
        if d["ci_lo"] > 0 and d["accuracy_delta_vs_final"] >= ADAPT_MIN:
            MAXC = "POSITIVE_SIGNAL"
        elif d["accuracy_delta_vs_final"] <= -ADAPT_MIN and d["ci_hi"] < 0:
            MAXC = "HARMFUL"
        else:
            MAXC = "NO_SIGNAL"
        cond_a = (oh_pt >= PRESENCE_MIN and oh_lo > 0)
        cond_b = (len(fw) > 0 and hrr >= PRESENCE_FRAC_MIN and hrr_lo > 0)
        PRES = "PRESENT" if (cond_a or cond_b) else "NOT_DETECTED"

    # ---- non-gating confidence diagnostics ----
    sel_conf = [r["snap_confs"][r["max_conf_idx"]] for r in rows]
    sel_corr = [r["max_conf_correct"] for r in rows]
    mc_corr = float(np.mean([c for c, ok in zip(sel_conf, sel_corr) if ok])) if any(sel_corr) else None
    mc_inc = float(np.mean([c for c, ok in zip(sel_conf, sel_corr) if not ok])) if any(1 - x for x in sel_corr) else None
    corr = float(np.corrcoef(sel_conf, sel_corr)[0, 1]) if len(set(sel_corr)) > 1 else None
    selector_hist = {SNAP_NAMES[i]: int(sum(1 for r in rows if r["max_conf_idx"] == i)) for i in range(5)}

    def strat(field):
        out = {}
        for v in sorted(set(r[field] for r in rows)):
            rr = [r for r in rows if r[field] == v]
            out[str(v)] = {"n": len(rr), "final_acc": float(np.mean([r["final_correct"] for r in rr])),
                           "oracle_hist_only_acc": float(np.mean([r["oracle_hist_only_correct"] for r in rr])),
                           "maxconf_acc": float(np.mean([r["max_conf_correct"] for r in rr]))}
        return out

    out = {"packet": "RNN-07A-BRIDGE-R1", "status": "RUN",
           "workload": "NoLiMa ONLYDirect SEMI_SYNTHETIC_CONTROLLED_BRIDGE (32K recovery cell, in-run capture)",
           "capture_semantics": "single_trajectory_in_run_run_trajectory (NOT prefix re-prefill)",
           "n_recovery": n, "n_pool": len(pool), "n_short_correct": len(eligible),
           "declared_cap": N_RECOVERY, "b_cap": B_CAP, "context_len": LEN32, "needle_depth": DEPTH,
           "boundaries": boundaries, "snapshot_schedule": PROGRESS,
           "short_eligibility": {"n_generated": len(pool), "n_short_correct": len(eligible),
                                 "reasoning_type_dist": dict(rt_dist), "needle_id_dist": dict(nd_dist)},
           "fresh_set_identity": {"R1QualificationSetSha256": r1_set_sha, "selectedSetSha256": sel_set_sha,
                                  "disjoint_from_historical_recovery": bool(len(overlap) == 0),
                                  "overlap_count": len(overlap), "n_historical_uids": len(hist_uids),
                                  "selected_records": sel_records},
           "temporal_identity_replay": {"replay_subset_n": REPLAY_SUBSET_N, "all_boundary_hashes_match": bool(replay_ok),
                                        "records": replay_records},
           "arm_accuracy": {"FINAL": float(np.mean(final_c)), **snap_acc,
                            "MAX_CONFIDENCE": float(np.mean(maxc_c)),
                            "ORACLE_HISTORICAL_ONLY_diagnostic": float(np.mean(oracle_hist)),
                            "ORACLE_ALL_diagnostic": float(np.mean(oracle_all))},
           "fixed_arm_vs_final": fixed_decomp, "max_confidence_vs_final": maxc_decomp,
           "historical_information": {"oracle_hist_only_acc": float(np.mean(oracle_hist)),
                                      "oracle_hist_only_minus_final": oh_pt, "ci_lo": oh_lo, "ci_hi": oh_hi,
                                      "n_final_wrong": len(fw), "historically_recoverable_final_wrong": hist_recov_fw,
                                      "recoverability_rate_over_final_wrong": hrr,
                                      "recoverability_rate_wilson_lb": hrr_lo},
           "selector_histogram": selector_hist,
           "confidence_diagnostics_nongating": {"mean_selected_conf_correct": mc_corr,
                                                "mean_selected_conf_incorrect": mc_inc,
                                                "corr_selected_conf_correctness": corr},
           "strata": {"by_needle_id": strat("needle_id"), "by_reasoning_type": strat("reasoning_type")},
           "thresholds": {"REC_EFFECT_MIN": REC_EFFECT_MIN, "ADAPT_MIN": ADAPT_MIN,
                          "PRESENCE_MIN": PRESENCE_MIN, "PRESENCE_FRAC_MIN": PRESENCE_FRAC_MIN, "MIN_N": MIN_N},
           "HISTORICAL_INFORMATION_PRESENCE_R1": PRES,
           "TRUE_IN_RUN_COARSE_HISTORICAL_RECOVERY_R1": RECOV,
           "TRUE_IN_RUN_MAX_CONFIDENCE_R1": MAXC,
           "provenance": {"runner_git_blob": git("hash-object", runner),
                          "runner_dirty": git("status", "--porcelain", "--", runner),
                          "git_head": git("rev-parse", "HEAD"),
                          "r1_protocol_sha256": sha256_file(os.path.join(OUTDIR, "R1_PRE_REGISTRATION.md")),
                          "external_provenance_sha256": sha256_file(os.path.join(OUTDIR, "EXTERNAL_WORKLOAD_PROVENANCE.json")),
                          "repo_id": L.REPO_ID, "revision": L.REVISION, "mamba_ssm": mamba_ssm.__version__},
           "fast_path_active": bool(not any(L.fallback_reachable().values()) and L.KCOUNT["selective_state_update"] > 0),
           "kernel_counts": dict(L.KCOUNT), "seconds": round(time.time() - t0, 1),
           "rows": rows}
    json.dump(out, open(os.path.join(OUTDIR, "R1_RESULTS.json"), "w"), indent=2, default=str)
    print("== RNN-07A-BRIDGE-R1 ==")
    print(f"n={n} disjoint={len(overlap)==0} replay_ok={replay_ok} fast_path={out['fast_path_active']}")
    print(f"arm acc: FINAL={np.mean(final_c):.3f} MAXCONF={np.mean(maxc_c):.3f} "
          f"ORACLE_HIST_ONLY={np.mean(oracle_hist):.3f} ORACLE_ALL={np.mean(oracle_all):.3f}")
    print(f"HISTORICAL_INFORMATION_PRESENCE_R1        = {PRES}")
    print(f"TRUE_IN_RUN_COARSE_HISTORICAL_RECOVERY_R1 = {RECOV}")
    print(f"TRUE_IN_RUN_MAX_CONFIDENCE_R1             = {MAXC}")


if __name__ == "__main__":
    main()
