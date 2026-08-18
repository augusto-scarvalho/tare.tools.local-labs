#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06T Section 5 — non-synthetic (natural-language) recovery SCOUT (exploratory, bounded).

RULER-style single-needle-in-haystack in NATURAL-LANGUAGE filler (real English sentences, derived
from the repo's rnn_ruler_smoke.py fixture), single-token answer so the T0/3A/3B constrained readout
scores it deterministically. Needle depth VARIES per example; the snapshot schedule and MAX_CONFIDENCE
selector are target-agnostic. Compares FINAL, FIXED slot-115 (3B BEST_FIXED), frozen MAX_CONFIDENCE on
the official fast path via single-pass capture. Exploratory only — no population generalization minted.
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
import rnn_06t_lib as L  # noqa: E402

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06T")
TOK_ID = "EleutherAI/gpt-neox-20b"
POOL_SEED = 20260817
SCOUT_SEED = 20260990
BATCH = 16
N = 64
CTX_LEN = 768
SCHEDULE = [38, 76, 115, 153]
CAP_BOUNDS = [4 * (s + 1) for s in SCHEDULE] + [CTX_LEN]
FIXED_SLOT = 115
DEPTH_MIN, DEPTH_MAX = 40, 560
FILLER = ("The grass is green and the sky is blue. Nothing important happens here. "
          "The city was quiet that morning and people walked slowly along the river. "
          "A gentle wind moved through the trees while the market opened for the day. "
          "Children played in the square and the old clock tower rang on the hour. ")
SIGNAL = 0.15


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


def main():
    t0 = time.time(); runner = os.path.abspath(__file__)
    L.install_counters(); import mamba_ssm  # noqa
    tok = AutoTokenizer.from_pretrained(TOK_ID)
    pools, _ = L.build_official_pools(tok, POOL_SEED)
    vset = sorted(set(pools["scored_vals"])); vt = torch.tensor(vset, device=L.DEVICE, dtype=torch.long); V = len(vset)
    model = L.load_model(); w0 = L.weights_identity(model)

    filler_ids = tok(FILLER * 40, add_special_tokens=False)["input_ids"]
    pre_ids = tok(" The magic number for", add_special_tokens=False)["input_ids"]
    is_ids = tok(" is", add_special_tokens=False)["input_ids"]
    dot_ids = tok(".", add_special_tokens=False)["input_ids"]

    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SCOUT_SEED])))
    ctx_all, query_all, golds, depths = [], [], [], []
    for i in range(N):
        key_tok = int(pools["scored_keys"][int(rng.integers(0, len(pools["scored_keys"])))])
        val_tok = int(pools["scored_vals"][int(rng.integers(0, len(pools["scored_vals"])))])
        needle = pre_ids + [key_tok] + is_ids + [val_tok] + dot_ids
        depth = int(rng.integers(DEPTH_MIN, DEPTH_MAX))
        body = filler_ids[:depth] + needle + filler_ids[depth:]
        body = body[:CTX_LEN]
        # guarantee the needle survives the fixed 768-token window
        if depth + len(needle) > CTX_LEN:
            depth = CTX_LEN - len(needle) - 1
            body = (filler_ids[:depth] + needle + filler_ids[depth:])[:CTX_LEN]
        query = pre_ids + [key_tok] + is_ids
        ctx_all.append(body); query_all.append(query); golds.append(val_tok); depths.append(depth)
    golds = np.array(golds); depths = np.array(depths)
    gcol = np.array([vset.index(int(g)) for g in golds])
    Lq = len(query_all[0])
    assert all(len(q) == Lq for q in query_all)

    K = len(SCHEDULE)
    pool_logits = np.zeros((N, K, V), np.float32); final_logits = np.zeros((N, V), np.float32)
    counters = {"singlePassRuns": 0, "snapshotsRestored": 0, "queriesEvaluated": 0, "snapshotBoundaryFailures": 0}
    L.reset_counters()
    for b in range(0, N, BATCH):
        rows = list(range(b, min(b + BATCH, N)))
        ctx = torch.tensor([ctx_all[r] for r in rows], device=L.DEVICE, dtype=torch.long)
        qy = torch.tensor([query_all[r] for r in rows], device=L.DEVICE, dtype=torch.long)
        snaps = L.run_trajectory(model, ctx, CAP_BOUNDS); counters["singlePassRuns"] += 1
        for ki, s in enumerate(SCHEDULE):
            pos = 4 * (s + 1)
            _, sub = L.readout_multi(model, snaps[pos], qy, pos, vt)
            counters["snapshotsRestored"] += len(rows); counters["queriesEvaluated"] += len(rows)
            for li, r in enumerate(rows):
                pool_logits[r, ki] = sub[li].cpu().numpy()
        _, sub = L.readout_multi(model, snaps[CTX_LEN], qy, CTX_LEN, vt)
        counters["snapshotsRestored"] += len(rows); counters["queriesEvaluated"] += len(rows)
        for li, r in enumerate(rows):
            final_logits[r] = sub[li].cpu().numpy()
        del snaps; torch.cuda.empty_cache()

    def softmax(x):
        e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
    pool_pred = pool_logits.argmax(-1); pool_correct = (pool_pred == gcol[:, None]); pool_top1 = softmax(pool_logits).max(-1)
    final_correct = (final_logits.argmax(-1) == gcol); accF = float(final_correct.mean())
    fixed = pool_correct[:, SCHEDULE.index(FIXED_SLOT)]
    maxconf_sel = pool_top1.argmax(1); maxconf = (pool_pred[np.arange(N), maxconf_sel] == gcol)
    oracle_best = pool_correct.any(1)

    res = {"packet": "RNN-06T-SCOUT", "kind": "natural_language_needle_scout_exploratory",
           "predeclared": {"source": "self-contained RULER-style NL needle-in-haystack (repo "
                           "rnn_ruler_smoke.py fixture family); NO download", "seed": SCOUT_SEED,
                           "n": N, "ctx_len": CTX_LEN, "depth_range": [DEPTH_MIN, DEPTH_MAX],
                           "metric": "constrained single-token needle-value retrieval accuracy",
                           "selector": "frozen MAX_CONFIDENCE", "fixed_control_slot": FIXED_SLOT,
                           "schedule": SCHEDULE, "arms": ["FINAL", f"FIXED_SLOT_{FIXED_SLOT}", "MAX_CONFIDENCE"]},
           "filler_sha256": hashlib.sha256(FILLER.encode()).hexdigest()[:16],
           "arms": {"FINAL": accF, f"FIXED_SLOT_{FIXED_SLOT}": float(fixed.mean()),
                    "MAX_CONFIDENCE": float(maxconf.mean()), "ORACLE_BEST_GOLD": float(oracle_best.mean()),
                    "pool_per_slot": {int(SCHEDULE[k]): float(pool_correct[:, k].mean()) for k in range(K)}},
           "deltas": {"maxconf_minus_final": round(float(maxconf.mean() - accF), 4),
                      "maxconf_minus_fixed": round(float(maxconf.mean() - fixed.mean()), 4),
                      "fixed_minus_final": round(float(fixed.mean() - accF), 4)},
           "selectedSnapshotHistogram": [int((maxconf_sel == k).sum()) for k in range(K)],
           "mechanism_activation": counters,
           "fast_path_active": bool(not any(L.fallback_reachable().values()) and L.KCOUNT["selective_state_update"] > 0),
           "weights_immutable": bool(w0 == L.weights_identity(model)),
           "executed_source_identity": {"runner_git_blob": git("hash-object", runner),
                                        "runner_dirty": git("status", "--porcelain", "--", runner),
                                        "git_head": git("rev-parse", "HEAD"), "repo_id": L.REPO_ID,
                                        "revision": L.REVISION, "mamba_ssm": mamba_ssm.__version__}}
    dmf = res["deltas"]["maxconf_minus_final"]
    if not res["fast_path_active"] or counters["queriesEvaluated"] == 0:
        verdict = "BLOCKED"
    elif dmf >= SIGNAL:
        verdict = "POSITIVE_SIGNAL"
    else:
        verdict = "NO_SIGNAL"
    res["NON_SYNTHETIC_RECOVERY_SCOUT"] = verdict
    res["note"] = ("Exploratory scout: real-English filler + inserted single-token needle at varying "
                   "depth. Semi-synthetic (natural-language context, controlled needle). No "
                   "population-level generalization claimed.")
    res["total_runtime_s"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(OUTDIR, "T1_NONSYNTH_SCOUT.json"), "w"), indent=2, default=str)

    print(f"FINAL={accF:.3f} FIXED_{FIXED_SLOT}={float(fixed.mean()):.3f} MAX_CONF={float(maxconf.mean()):.3f} "
          f"ORACLE={float(oracle_best.mean()):.3f}")
    print(f"pool_per_slot={res['arms']['pool_per_slot']}")
    print(f"deltas={res['deltas']} histogram={res['selectedSnapshotHistogram']}")
    print(f"fast_path_active={res['fast_path_active']}")
    print(f"NON_SYNTHETIC_RECOVERY_SCOUT = {verdict}  runtime={res['total_runtime_s']}s")


if __name__ == "__main__":
    main()
