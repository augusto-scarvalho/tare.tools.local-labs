#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-07A-BRIDGE — long-context degradation + (gated) recovery on the NoLiMa controlled bridge.

Runs ONLY if BRIDGE_SHORT_CONTEXT_COMPETENCE == SUFFICIENT. On the short-competence-eligible population
(short-correct; eligibility independent of the long outcome), measures bounded long-context (8/16/32K)
FINAL accuracy vs short. Mints BRIDGE_LONG_CONTEXT_DEGRADATION. Only if degradation QUALIFIED, runs the
frozen historical-snapshot + MAX_CONFIDENCE recovery machinery and mints BRIDGE_HISTORICAL_RECOVERY_SIGNAL
and BRIDGE_ADAPTIVE_SELECTION_SIGNAL. B=1.
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
CELLS = {"8K": 8000, "16K": 16000, "32K": 32000}
CELL_SEED = {"8K": 0, "16K": 1, "32K": 2}   # stable per-cell seed offsets (NOT hash(); process-stable)
DEGRADE_MARGIN = 0.15
MIN_FORGOTTEN = 15
REC_MIN = 0.15
ADAPT_MIN = 0.05
MAX_LONG_EVAL = 90
MAX_RECOVERY = 48
SNAP_NAMES = ["SNAP_25", "SNAP_50", "SNAP_75", "SNAP_90", "FINAL"]


def stub(reason):
    out = {"packet": "RNN-07A-BRIDGE-LONG", "status": "NOT_RUN", "reason": reason,
           "BRIDGE_LONG_CONTEXT_DEGRADATION": "NOT_RUN",
           "BRIDGE_HISTORICAL_RECOVERY_SIGNAL": "N/A_GATE_NOT_MET",
           "BRIDGE_ADAPTIVE_SELECTION_SIGNAL": "N/A_GATE_NOT_MET"}
    json.dump(out, open(os.path.join(OUTDIR, "BRIDGE_LONG_RESULTS.json"), "w"), indent=2)
    print(f"[bridge-long] NOT RUN: {reason}")


def main():
    t0 = time.time()
    short = json.load(open(os.path.join(OUTDIR, "BRIDGE_SHORT_RESULTS.json")))
    if short["BRIDGE_SHORT_CONTEXT_COMPETENCE"] != "SUFFICIENT":
        return stub(f"short competence = {short['BRIDGE_SHORT_CONTEXT_COMPETENCE']}")

    L.install_counters()
    tok = A.load_tokenizer()
    model = L.load_model()
    book = B.load_book_tokens(tok)
    pool = B.build_pool(tok)
    short_rows = short["rows"]
    assert len(pool) == len(short_rows), "pool/short row mismatch"
    eligible = [(k, ex) for k, (ex, r) in enumerate(zip(pool, short_rows)) if r["correct"] == 1][:MAX_LONG_EVAL]
    print(f"[bridge-long] eligible={len(eligible)}")

    # ---- long-context degradation per cell (FINAL) ----
    cell_res = {}
    for cell, budget in CELLS.items():
        max_seqlen = budget + B.SHORT_TOKENS + 512
        rows = []
        for j, (k, ex) in enumerate(eligible):
            ctx, npos = B.make_context(book, ex["needle_ids"], budget, B.NEEDLE_DEPTH,
                                       filler_seed=B.POOL_SEED + 2000 + CELL_SEED[cell] * 100000 + k)
            pred, conf = B.eval_state_from_ctx(model, ctx, ex, max_seqlen)
            rows.append({"idx": k, "reasoning_type": ex["reasoning_type"], "gold": ex["gold"],
                         "final_pred": pred, "final_correct": int(pred == ex["gold"]), "n_ctx": len(ctx)})
            if (j + 1) % 20 == 0:
                print(f"[bridge-long {cell}] {j+1}/{len(eligible)} elapsed={time.time()-t0:.1f}s")
        n = len(rows)
        long_correct = [r["final_correct"] for r in rows]
        short_correct = [1] * n   # eligible are short-correct by construction
        long_acc, lb, ub = A.wilson(sum(long_correct), n)
        d_pt, d_lo, d_hi = A.paired_bootstrap_delta(short_correct, long_correct)  # short - long
        forgotten = [r for r in rows if r["final_correct"] == 0]
        cell_res[cell] = {"budget": budget, "n": n, "long_final_acc": long_acc, "long_wilson_lb": lb,
                          "degradation_short_minus_long": {"point": d_pt, "ci_lo": d_lo, "ci_hi": d_hi},
                          "n_forgotten": len(forgotten),
                          "qualifies": bool(d_pt >= DEGRADE_MARGIN and d_lo > 0 and len(forgotten) >= MIN_FORGOTTEN),
                          "rows": rows}
        print(f"[bridge-long {cell}] long_acc={long_acc:.3f} degr={d_pt:.3f} (CI_lo {d_lo:.3f}) "
              f"forgotten={len(forgotten)} qualifies={cell_res[cell]['qualifies']}")

    qual_cells = [c for c in CELLS if cell_res[c]["qualifies"]]
    if qual_cells:
        rec_cell = max(qual_cells, key=lambda c: cell_res[c]["degradation_short_minus_long"]["point"])
        DEGRADATION = "QUALIFIED"
    else:
        rec_cell = None
        DEGRADATION = "NOT_QUALIFIED"

    out = {"packet": "RNN-07A-BRIDGE-LONG", "status": "RUN",
           "workload": "NoLiMa SEMI_SYNTHETIC_CONTROLLED_BRIDGE",
           "eligible_population": len(eligible),
           "thresholds": {"DEGRADE_MARGIN": DEGRADE_MARGIN, "MIN_FORGOTTEN": MIN_FORGOTTEN,
                          "REC_MIN": REC_MIN, "ADAPT_MIN": ADAPT_MIN},
           "per_cell_degradation": {c: {k: v for k, v in cell_res[c].items() if k != "rows"} for c in CELLS},
           "BRIDGE_LONG_CONTEXT_DEGRADATION": DEGRADATION, "recovery_cell": rec_cell}

    # ---- recovery (only if degradation QUALIFIED) ----
    if DEGRADATION != "QUALIFIED":
        out["BRIDGE_HISTORICAL_RECOVERY_SIGNAL"] = "N/A_NO_DEGRADATION"
        out["BRIDGE_ADAPTIVE_SELECTION_SIGNAL"] = "N/A_NO_DEGRADATION"
    else:
        budget = CELLS[rec_cell]
        rec_elig = eligible[:MAX_RECOVERY]
        rrows = []
        for j, (k, ex) in enumerate(rec_elig):
            ctx, npos = B.make_context(book, ex["needle_ids"], budget, B.NEEDLE_DEPTH,
                                       filler_seed=B.POOL_SEED + 2000 + CELL_SEED[rec_cell] * 100000 + k)
            preds, confs = B.snapshot_eval(model, ctx, ex, budget)
            g = ex["gold"]; final_pred = preds[-1]
            mci = int(np.argmax(confs));
            rrows.append({"idx": k, "reasoning_type": ex["reasoning_type"], "gold": g,
                          "snap_preds": preds, "snap_confs": confs,
                          "snap_correct": [int(p == g) for p in preds], "final_correct": int(final_pred == g),
                          "max_conf_idx": mci, "max_conf_correct": int(preds[mci] == g),
                          "oracle_correct": int(any(p == g for p in preds))})
            if (j + 1) % 10 == 0:
                print(f"[bridge-recovery {rec_cell}] {j+1}/{len(rec_elig)} elapsed={time.time()-t0:.1f}s")
        n = len(rrows)
        final_c = [r["final_correct"] for r in rrows]
        maxc_c = [r["max_conf_correct"] for r in rrows]
        oracle_c = [r["oracle_correct"] for r in rrows]
        snap_acc = {SNAP_NAMES[i]: float(np.mean([r["snap_correct"][i] for r in rrows])) for i in range(5)}
        forgotten = [r for r in rrows if r["final_correct"] == 0]
        nf = len(forgotten)
        rec_by_snap = {}
        for i in range(4):
            rate = float(np.mean([r["snap_correct"][i] for r in forgotten])) if forgotten else 0.0
            pt, lo, hi = A.paired_bootstrap_delta([r["snap_correct"][i] for r in rrows], final_c)
            rec_by_snap[SNAP_NAMES[i]] = {"recovery_rate_over_forgotten": rate,
                                          "acc_delta_vs_final": pt, "ci_lo": lo, "ci_hi": hi}
        best_snap = max(rec_by_snap.items(), key=lambda kv: kv[1]["recovery_rate_over_forgotten"])
        if nf < MIN_FORGOTTEN:
            HIST = "INCONCLUSIVE"
        else:
            bs = best_snap[1]
            HIST = "POSITIVE_SIGNAL" if (bs["ci_lo"] > 0 and bs["recovery_rate_over_forgotten"] >= REC_MIN) else "NO_SIGNAL"
        adapt_pt, adapt_lo, adapt_hi = A.paired_bootstrap_delta(maxc_c, final_c)
        if n < MIN_FORGOTTEN:
            ADAPT = "INCONCLUSIVE"
        else:
            ADAPT = "POSITIVE_SIGNAL" if (adapt_lo > 0 and adapt_pt >= ADAPT_MIN) else "NO_SIGNAL"
        selector_hist = {SNAP_NAMES[i]: int(sum(1 for r in rrows if r["max_conf_idx"] == i)) for i in range(5)}
        out["recovery"] = {"cell": rec_cell, "n_eligible": n, "n_forgotten": nf,
                           "arm_accuracy": {"FINAL": float(np.mean(final_c)), **snap_acc,
                                            "MAX_CONFIDENCE": float(np.mean(maxc_c)),
                                            "ORACLE_BEST_GOLD_diagnostic": float(np.mean(oracle_c))},
                           "historical_recovery_by_snapshot": rec_by_snap, "best_earlier_snapshot": best_snap[0],
                           "adaptive_max_conf_vs_final": {"delta": adapt_pt, "ci_lo": adapt_lo, "ci_hi": adapt_hi},
                           "selector_histogram": selector_hist, "rows": rrows}
        out["BRIDGE_HISTORICAL_RECOVERY_SIGNAL"] = HIST
        out["BRIDGE_ADAPTIVE_SELECTION_SIGNAL"] = ADAPT

    out["fast_path_active"] = bool(not any(L.fallback_reachable().values()) and L.KCOUNT["selective_state_update"] > 0)
    out["kernel_counts"] = dict(L.KCOUNT)
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(OUTDIR, "BRIDGE_LONG_RESULTS.json"), "w"), indent=2, default=str)
    print("== RNN-07A-BRIDGE LONG ==")
    print(f"BRIDGE_LONG_CONTEXT_DEGRADATION = {DEGRADATION} (recovery cell: {rec_cell})")
    print(f"BRIDGE_HISTORICAL_RECOVERY_SIGNAL = {out.get('BRIDGE_HISTORICAL_RECOVERY_SIGNAL')}")
    print(f"BRIDGE_ADAPTIVE_SELECTION_SIGNAL  = {out.get('BRIDGE_ADAPTIVE_SELECTION_SIGNAL')}")


if __name__ == "__main__":
    main()
