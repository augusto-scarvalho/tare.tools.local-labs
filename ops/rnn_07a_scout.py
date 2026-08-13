#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-07A SCOUT — competence + degradation screen (<90 GPU min target).

For a bounded candidate pool (priority domains, difficulty=easy, native token length <= cell budget):
  control  = question-conditioned BM25 RAG (2048 tok, target-agnostic)  -> content + letter readout
  full     = native context (<= cell budget)                            -> content + letter readout (FINAL)
Mints REALISTIC_TASK_COMPETENCE and REALISTIC_FORGETTING_OPERATING_POINT. B=1 (no context padding).
Writes runs/rnn/RNN-07A/SCOUT_<cell>.json and SCOUT_SUMMARY.json.
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
SEED = 20261300
MAX_PER_CELL = 60
DEGRADE_MARGIN = 0.10
FORMAT_TOL = 0.10
MIN_ELIGIBLE = 20
MIN_FORGOTTEN = 15
COMP_WILSON_LB = 0.35
CELLS = {"16K": 16000, "32K": 32000}   # ~8K infeasible (0 native); 64K deferred to recovery if needed


def build_pool(data, tok, budget, cap):
    pool = []
    for ex in data:
        if ex["domain"] not in A.PRIORITY_DOMAINS:
            continue
        if ex["difficulty"] != "easy":
            continue
        ids = A.enc(tok, ex["context"])
        if len(ids) > budget:
            continue
        pool.append((ex, ids))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pool))
    return [pool[i] for i in order[:cap]]


@torch.no_grad()
def eval_context(model, tok, ctx_ids, ex, letter_ids, max_seqlen):
    choice_texts = [ex["choice_" + c] for c in A.LETTERS]
    stem = A.build_query_stem(tok, ex["question"])
    stem_fmt = A.build_query_stem_format(tok, ex["question"], choice_texts)
    choices = A.build_choice_tokens(tok, choice_texts)
    ctx = torch.tensor([ctx_ids], device=A.DEVICE, dtype=torch.long)
    state, off, _ = A.prefill_state(model, ctx, max_seqlen)
    stem_t = torch.tensor([stem], device=A.DEVICE, dtype=torch.long)
    stem_fmt_t = torch.tensor([stem_fmt], device=A.DEVICE, dtype=torch.long)
    choices_pad = A.pad_choices([choices])
    r = A.readout_from_state(model, state, off, stem_t, choices_pad, max_seqlen)
    r["letter_pred"] = A.letter_readout(model, state, off, stem_fmt_t, letter_ids, max_seqlen)
    return r


def run_cell(model, tok, data, cell, budget, letter_ids):
    pool = build_pool(data, tok, budget, MAX_PER_CELL)
    max_seqlen = budget + 512
    rows = []
    t0 = time.time()
    for k, (ex, ids) in enumerate(pool):
        g = A.gold_index(ex)
        rag = A.bm25_rag_control(ids, A.enc(tok, ex["question"]))
        rc = eval_context(model, tok, rag, ex, letter_ids, max_seqlen)
        fc = eval_context(model, tok, ids, ex, letter_ids, A.CHUNK + budget + 512)
        rows.append({
            "id": ex["_id"], "domain": ex["domain"], "length": ex["length"], "gold": g,
            "n_ctx_tokens": len(ids), "n_rag_tokens": len(rag),
            "control_content_pred": int(rc["content_pred"][0]),
            "control_letter_pred": int(rc["letter_pred"][0]),
            "control_conf": float(rc["confidence"][0]),
            "full_content_pred": int(fc["content_pred"][0]),
            "full_letter_pred": int(fc["letter_pred"][0]),
            "full_conf": float(fc["confidence"][0]),
            "control_correct": int(rc["content_pred"][0] == g),
            "full_correct": int(fc["content_pred"][0] == g),
            "control_letter_matches_content": int(rc["letter_pred"][0] == rc["content_pred"][0]),
            "full_letter_matches_content": int(fc["letter_pred"][0] == fc["content_pred"][0]),
        })
        if (k + 1) % 10 == 0:
            print(f"[scout {cell}] {k+1}/{len(pool)} elapsed={time.time()-t0:.1f}s")
    n = len(rows)
    ctrl_correct = [r["control_correct"] for r in rows]
    full_correct = [r["full_correct"] for r in rows]
    ctrl_acc, ctrl_lb, ctrl_ub = A.wilson(sum(ctrl_correct), n)
    full_acc, full_lb, full_ub = A.wilson(sum(full_correct), n)
    d_point, d_lo, d_hi = A.paired_bootstrap_delta(ctrl_correct, full_correct)  # control - full
    # format adherence on eligible (control-correct) population
    elig = [r for r in rows if r["control_correct"] == 1]
    forgotten = [r for r in elig if r["full_correct"] == 0]
    fmt_ctrl = float(np.mean([r["control_letter_matches_content"] for r in elig])) if elig else 0.0
    fmt_full = float(np.mean([r["full_letter_matches_content"] for r in elig])) if elig else 0.0
    out = {"cell": cell, "budget_tokens": budget, "n": n,
           "seconds": round(time.time() - t0, 1),
           "control_accuracy": {"acc": ctrl_acc, "wilson_lb": ctrl_lb, "wilson_ub": ctrl_ub,
                                "k": int(sum(ctrl_correct))},
           "full_accuracy": {"acc": full_acc, "wilson_lb": full_lb, "wilson_ub": full_ub,
                             "k": int(sum(full_correct))},
           "degradation_control_minus_full": {"point": d_point, "ci_lo": d_lo, "ci_hi": d_hi},
           "n_eligible": len(elig), "n_forgotten": len(forgotten),
           "format_adherence_eligible": {"control": fmt_ctrl, "full": fmt_full,
                                         "drop": fmt_ctrl - fmt_full},
           "by_domain": {},
           "rows": rows}
    for dom in A.PRIORITY_DOMAINS:
        dr = [r for r in rows if r["domain"] == dom]
        if dr:
            out["by_domain"][dom] = {"n": len(dr),
                                     "control_acc": float(np.mean([r["control_correct"] for r in dr])),
                                     "full_acc": float(np.mean([r["full_correct"] for r in dr]))}
    json.dump(out, open(os.path.join(OUTDIR, f"SCOUT_{cell}.json"), "w"), indent=2, default=str)
    print(f"[scout {cell}] n={n} control_acc={ctrl_acc:.3f} (LB {ctrl_lb:.3f}) full_acc={full_acc:.3f} "
          f"elig={len(elig)} forgotten={len(forgotten)} fmt_drop={fmt_ctrl-fmt_full:.3f} secs={out['seconds']}")
    return out


def main():
    t_start = time.time()
    L.install_counters()
    tok = A.load_tokenizer()
    data = A.load_data()
    letter_ids = A.letter_token_ids(tok)
    model = L.load_model()
    cell_outs = {}
    for cell, budget in CELLS.items():
        cell_outs[cell] = run_cell(model, tok, data, cell, budget, letter_ids)
        if time.time() - t_start > 80 * 60:
            print("[scout] approaching 90-min scout budget; stopping cell sweep")
            break

    # ---- mints: pick the primary cell (largest n with control competence) ----
    # REALISTIC_TASK_COMPETENCE across cells: SUFFICIENT iff any run cell has control Wilson-LB>0.35 and n_elig>=MIN
    comp_cell = None
    for cell, o in cell_outs.items():
        if o["control_accuracy"]["wilson_lb"] > COMP_WILSON_LB and o["n_eligible"] >= MIN_ELIGIBLE:
            comp_cell = cell if comp_cell is None else comp_cell
    competence = "SUFFICIENT" if comp_cell else "INSUFFICIENT"

    # forgetting operating point on the primary competent cell (or the largest-budget run cell)
    op_cell = comp_cell or (list(cell_outs.keys())[-1] if cell_outs else None)
    op = "BLOCKED"
    op_detail = {}
    if competence == "SUFFICIENT" and op_cell:
        o = cell_outs[op_cell]
        deg = o["degradation_control_minus_full"]
        fmt_drop = o["format_adherence_eligible"]["drop"]
        cond_degrade = (deg["point"] >= DEGRADE_MARGIN and deg["ci_lo"] > 0)
        cond_format = (fmt_drop <= FORMAT_TOL)
        cond_forgotten = (o["n_forgotten"] >= MIN_FORGOTTEN)
        op_detail = {"cell": op_cell, "degrade_point": deg["point"], "degrade_ci_lo": deg["ci_lo"],
                     "cond_degrade": bool(cond_degrade), "format_drop": fmt_drop,
                     "cond_format": bool(cond_format), "n_forgotten": o["n_forgotten"],
                     "cond_forgotten": bool(cond_forgotten)}
        op = "FOUND" if (cond_degrade and cond_format and cond_forgotten) else "NOT_FOUND_WITHIN_BUDGET"

    summary = {"packet": "RNN-07A-SCOUT", "seconds_total": round(time.time() - t_start, 1),
               "cells_run": list(cell_outs.keys()),
               "competence_cell": comp_cell, "operating_point_cell": op_cell,
               "REALISTIC_TASK_COMPETENCE": competence,
               "REALISTIC_FORGETTING_OPERATING_POINT": op,
               "operating_point_detail": op_detail,
               "per_cell": {c: {k: v for k, v in o.items() if k != "rows"} for c, o in cell_outs.items()},
               "thresholds": {"COMP_WILSON_LB": COMP_WILSON_LB, "MIN_ELIGIBLE": MIN_ELIGIBLE,
                              "DEGRADE_MARGIN": DEGRADE_MARGIN, "FORMAT_TOL": FORMAT_TOL,
                              "MIN_FORGOTTEN": MIN_FORGOTTEN},
               "fast_path_active": bool(not any(L.fallback_reachable().values())
                                        and L.KCOUNT["selective_state_update"] > 0),
               "kernel_counts": dict(L.KCOUNT)}
    json.dump(summary, open(os.path.join(OUTDIR, "SCOUT_SUMMARY.json"), "w"), indent=2, default=str)
    print("== RNN-07A SCOUT ==")
    print(f"REALISTIC_TASK_COMPETENCE = {competence}  (competent cell: {comp_cell})")
    print(f"REALISTIC_FORGETTING_OPERATING_POINT = {op}  (cell: {op_cell})")
    print(f"detail: {op_detail}")


if __name__ == "__main__":
    main()
