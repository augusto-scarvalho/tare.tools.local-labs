#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-07A RECOVERY — runs ONLY if the scout minted REALISTIC_FORGETTING_OPERATING_POINT = FOUND.

On the operating-point cell's competence-eligible population (control-correct examples), capture
historical snapshots at normalized progress 25/50/75/90% + FINAL (positional, never gold-aligned),
score each with the frozen content readout, and evaluate arms:
  FINAL, SNAP_25/50/75/90, MAX_CONFIDENCE (frozen, non-oracle), ORACLE_BEST_GOLD (diagnostic).
Mints REALISTIC_HISTORICAL_RECOVERY_SIGNAL and REALISTIC_ADAPTIVE_SELECTION_SIGNAL. B=1.
DISCOVERY only — the discovered cell must not be reused as confirmatory evidence (that is RNN-07B).
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_07a_lib as A      # noqa: E402
import rnn_06t_lib as L      # noqa: E402

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-07A")
MAX_RECOVERY = 40
REC_MIN = 0.15
ADAPT_MIN = 0.05
MIN_FORGOTTEN = 15
SNAP_NAMES = ["SNAP_25", "SNAP_50", "SNAP_75", "SNAP_90", "FINAL"]


@torch.no_grad()
def snapshot_preds(model, tok, ctx_ids, ex, letter_ids, budget):
    L_ctx = len(ctx_ids)
    max_seqlen = budget + A.CHUNK + 512
    stem = torch.tensor([A.build_query_stem(tok, ex["question"])], device=A.DEVICE, dtype=torch.long)
    choices = A.pad_choices([A.build_choice_tokens(tok, [ex["choice_" + c] for c in A.LETTERS])])
    preds, confs = [], []
    for p in A.PROGRESS:
        cut = max(1, int(round(p * L_ctx)))
        ctx = torch.tensor([ctx_ids[:cut]], device=A.DEVICE, dtype=torch.long)
        state, off, _ = A.prefill_state(model, ctx, max_seqlen)
        r = A.readout_from_state(model, state, off, stem, choices, max_seqlen)
        preds.append(int(r["content_pred"][0])); confs.append(float(r["confidence"][0]))
    return preds, confs


def main():
    t_start = time.time()
    L.install_counters()
    summ = json.load(open(os.path.join(OUTDIR, "SCOUT_SUMMARY.json")))
    if summ["REALISTIC_FORGETTING_OPERATING_POINT"] != "FOUND":
        out = {"packet": "RNN-07A-RECOVERY", "status": "NOT_RUN",
               "reason": f"operating point = {summ['REALISTIC_FORGETTING_OPERATING_POINT']} "
                         f"(competence={summ['REALISTIC_TASK_COMPETENCE']}); recovery mints are N/A",
               "REALISTIC_HISTORICAL_RECOVERY_SIGNAL": "N/A_NO_OPERATING_POINT",
               "REALISTIC_ADAPTIVE_SELECTION_SIGNAL": "N/A_NO_OPERATING_POINT"}
        json.dump(out, open(os.path.join(OUTDIR, "RECOVERY_RESULTS.json"), "w"), indent=2)
        print("[recovery] operating point NOT FOUND -> recovery not run; mints N/A")
        return

    cell = summ["operating_point_cell"]
    budget = summ["per_cell"][cell]["budget_tokens"]
    scout = json.load(open(os.path.join(OUTDIR, f"SCOUT_{cell}.json")))
    by_id = {ex["_id"]: ex for ex in A.load_data()}
    tok = A.load_tokenizer(); letter_ids = A.letter_token_ids(tok)
    model = L.load_model()

    eligible = [r for r in scout["rows"] if r["control_correct"] == 1][:MAX_RECOVERY]
    rows = []
    for k, r in enumerate(eligible):
        ex = by_id[r["id"]]; g = A.gold_index(ex)
        ctx_ids = A.enc(tok, ex["context"])
        preds, confs = snapshot_preds(model, tok, ctx_ids, ex, letter_ids, budget)
        final_pred = preds[-1]
        max_conf_idx = int(np.argmax(confs))            # frozen MAX_CONFIDENCE (non-oracle)
        max_conf_pred = preds[max_conf_idx]
        oracle_correct = int(any(pp == g for pp in preds))
        rows.append({"id": r["id"], "domain": ex["domain"], "length": ex["length"], "gold": g,
                     "n_ctx_tokens": len(ctx_ids), "snap_preds": preds, "snap_confs": confs,
                     "snap_correct": [int(pp == g) for pp in preds],
                     "final_correct": int(final_pred == g),
                     "max_conf_idx": max_conf_idx, "max_conf_correct": int(max_conf_pred == g),
                     "oracle_best_gold_correct": oracle_correct})
        if (k + 1) % 5 == 0:
            print(f"[recovery {cell}] {k+1}/{len(eligible)} elapsed={time.time()-t_start:.1f}s")

    n = len(rows)
    final_c = [r["final_correct"] for r in rows]
    maxc_c = [r["max_conf_correct"] for r in rows]
    oracle_c = [r["oracle_best_gold_correct"] for r in rows]
    snap_acc = {SNAP_NAMES[i]: float(np.mean([r["snap_correct"][i] for r in rows])) for i in range(5)}
    # forgotten population = eligible (control-correct by construction) AND final wrong
    forgotten = [r for r in rows if r["final_correct"] == 0]
    nf = len(forgotten)

    # historical recovery: best fixed EARLIER snapshot (25/50/75/90) recovery over forgotten pop
    rec_by_snap = {}
    for i in range(4):  # exclude FINAL
        rec = [r["snap_correct"][i] for r in forgotten]
        rate = float(np.mean(rec)) if forgotten else 0.0
        pt, lo, hi = A.paired_bootstrap_delta([r["snap_correct"][i] for r in rows], final_c)
        rec_by_snap[SNAP_NAMES[i]] = {"recovery_rate_over_forgotten": rate,
                                      "acc_delta_vs_final_full": pt, "ci_lo": lo, "ci_hi": hi,
                                      "n_forgotten": nf}
    best_snap = max(rec_by_snap.items(), key=lambda kv: kv[1]["recovery_rate_over_forgotten"]) if rec_by_snap else (None, {})
    # any-historical recovery over forgotten (best of the 4 earlier snapshots per-example)
    any_hist_rec = [int(any(r["snap_correct"][i] == 1 for i in range(4))) for r in forgotten]
    any_hist_rate = float(np.mean(any_hist_rec)) if forgotten else 0.0

    if nf < MIN_FORGOTTEN:
        HIST_SIGNAL = "INCONCLUSIVE"
    else:
        bs = best_snap[1]
        HIST_SIGNAL = "POSITIVE_SIGNAL" if (bs.get("ci_lo", -1) > 0 and
                                            bs.get("recovery_rate_over_forgotten", 0) >= REC_MIN) else "NO_SIGNAL"

    # adaptive selection: frozen MAX_CONFIDENCE vs FINAL on eligible population
    adapt_pt, adapt_lo, adapt_hi = A.paired_bootstrap_delta(maxc_c, final_c)
    best_fixed_acc = max(snap_acc[SNAP_NAMES[i]] for i in range(4)) if n else 0.0
    maxc_acc = float(np.mean(maxc_c)) if n else 0.0
    if n < MIN_FORGOTTEN:
        ADAPT_SIGNAL = "INCONCLUSIVE"
    else:
        ADAPT_SIGNAL = "POSITIVE_SIGNAL" if (adapt_lo > 0 and adapt_pt >= ADAPT_MIN) else "NO_SIGNAL"

    hist = [r["max_conf_idx"] for r in rows]
    selector_hist = {SNAP_NAMES[i]: int(sum(1 for h in hist if h == i)) for i in range(5)}

    out = {"packet": "RNN-07A-RECOVERY", "status": "RUN", "cell": cell, "budget_tokens": budget,
           "n_eligible_evaluated": n, "n_forgotten": nf,
           "arm_accuracy": {"FINAL": float(np.mean(final_c)), **snap_acc,
                            "MAX_CONFIDENCE": maxc_acc, "ORACLE_BEST_GOLD_diagnostic": float(np.mean(oracle_c))},
           "denominators": {"n_eligible": n, "n_forgotten": nf},
           "historical_recovery_by_snapshot": rec_by_snap,
           "best_earlier_snapshot": best_snap[0],
           "any_historical_recovery_rate_over_forgotten": any_hist_rate,
           "adaptive_max_conf_vs_final": {"delta": adapt_pt, "ci_lo": adapt_lo, "ci_hi": adapt_hi,
                                          "maxconf_acc": maxc_acc, "final_acc": float(np.mean(final_c)),
                                          "best_fixed_earlier_acc": best_fixed_acc},
           "selector_histogram": selector_hist,
           "by_domain": {},
           "REALISTIC_HISTORICAL_RECOVERY_SIGNAL": HIST_SIGNAL,
           "REALISTIC_ADAPTIVE_SELECTION_SIGNAL": ADAPT_SIGNAL,
           "thresholds": {"REC_MIN": REC_MIN, "ADAPT_MIN": ADAPT_MIN, "MIN_FORGOTTEN": MIN_FORGOTTEN},
           "fast_path_active": bool(not any(L.fallback_reachable().values())
                                    and L.KCOUNT["selective_state_update"] > 0),
           "seconds_total": round(time.time() - t_start, 1),
           "rows": rows}
    for dom in A.PRIORITY_DOMAINS:
        dr = [r for r in rows if r["domain"] == dom]
        if dr:
            out["by_domain"][dom] = {"n": len(dr), "final_acc": float(np.mean([r["final_correct"] for r in dr])),
                                     "maxconf_acc": float(np.mean([r["max_conf_correct"] for r in dr]))}
    json.dump(out, open(os.path.join(OUTDIR, "RECOVERY_RESULTS.json"), "w"), indent=2, default=str)
    print("== RNN-07A RECOVERY ==")
    print(f"cell={cell} n_elig={n} n_forgotten={nf}")
    print(f"arm acc: FINAL={out['arm_accuracy']['FINAL']:.3f} MAXCONF={maxc_acc:.3f} "
          f"ORACLE={out['arm_accuracy']['ORACLE_BEST_GOLD_diagnostic']:.3f}")
    print(f"REALISTIC_HISTORICAL_RECOVERY_SIGNAL  = {HIST_SIGNAL}")
    print(f"REALISTIC_ADAPTIVE_SELECTION_SIGNAL   = {ADAPT_SIGNAL}")


if __name__ == "__main__":
    main()
