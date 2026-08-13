#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-07A-BRIDGE — SHORT-context NoLiMa competence gate.

Builds the controlled pool, embeds each needle in ~SHORT_TOKENS of real book filler, scores 4-way MC
option-likelihood. Mints BRIDGE_SHORT_CONTEXT_COMPETENCE. If SUFFICIENT the long/recovery runner may run.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_07a_bridge_lib as B   # noqa: E402
import rnn_07a_lib as A          # noqa: E402
import rnn_06t_lib as L          # noqa: E402

OUTDIR = os.path.join("/mnt/c/projects/local-model-lifecycle", "runs", "rnn", "RNN-07A")
COMP_WILSON_LB = 0.50
MIN_ELIGIBLE = 20


def main():
    t0 = time.time()
    L.install_counters()
    tok = A.load_tokenizer()
    model = L.load_model()
    book = B.load_book_tokens(tok)
    pool = B.build_pool(tok)
    max_seqlen = B.SHORT_TOKENS + 512
    rows = []
    for k, ex in enumerate(pool):
        ctx, npos = B.make_context(book, ex["needle_ids"], B.SHORT_TOKENS, B.NEEDLE_DEPTH,
                                   filler_seed=B.POOL_SEED + 1000 + k)
        pred, conf = B.eval_state_from_ctx(model, ctx, ex, max_seqlen)
        rows.append({"needle_id": ex["needle_id"], "reasoning_type": ex["reasoning_type"], "test": ex["test"],
                     "gold": ex["gold"], "pred": pred, "conf": conf, "correct": int(pred == ex["gold"]),
                     "n_ctx": len(ctx)})
        if (k + 1) % 20 == 0:
            print(f"[bridge-short] {k+1}/{len(pool)} elapsed={time.time()-t0:.1f}s")
    n = len(rows)
    k_correct = sum(r["correct"] for r in rows)
    acc, lb, ub = A.wilson(k_correct, n)
    n_elig = k_correct
    competence = "SUFFICIENT" if (lb > COMP_WILSON_LB and n_elig >= MIN_ELIGIBLE) else "INSUFFICIENT"
    by_rt = {}
    for rt in sorted(set(r["reasoning_type"] for r in rows)):
        rr = [r for r in rows if r["reasoning_type"] == rt]
        by_rt[rt] = {"n": len(rr), "acc": float(np.mean([r["correct"] for r in rr]))}
    out = {"packet": "RNN-07A-BRIDGE-SHORT", "workload": "NoLiMa SEMI_SYNTHETIC_CONTROLLED_BRIDGE",
           "n": n, "short_tokens": B.SHORT_TOKENS, "needle_depth": B.NEEDLE_DEPTH,
           "accuracy": {"acc": acc, "wilson_lb": lb, "wilson_ub": ub, "k": k_correct},
           "n_eligible": n_elig, "chance": 0.25,
           "thresholds": {"COMP_WILSON_LB": COMP_WILSON_LB, "MIN_ELIGIBLE": MIN_ELIGIBLE},
           "by_reasoning_type": by_rt,
           "BRIDGE_SHORT_CONTEXT_COMPETENCE": competence,
           "fast_path_active": bool(not any(L.fallback_reachable().values())
                                    and L.KCOUNT["selective_state_update"] > 0),
           "kernel_counts": dict(L.KCOUNT), "seconds": round(time.time() - t0, 1),
           "rows": rows}
    json.dump(out, open(os.path.join(OUTDIR, "BRIDGE_SHORT_RESULTS.json"), "w"), indent=2, default=str)
    print("== RNN-07A-BRIDGE SHORT ==")
    print(f"n={n} acc={acc:.3f} (Wilson LB {lb:.3f}) eligible={n_elig} chance=0.25")
    print(f"BRIDGE_SHORT_CONTEXT_COMPETENCE = {competence}")


if __name__ == "__main__":
    main()
