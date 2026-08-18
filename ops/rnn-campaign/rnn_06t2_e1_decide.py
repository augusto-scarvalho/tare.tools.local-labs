#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06T2-E1 economics aggregation + separate mints. Pools RAW warm samples across process starts,
verifies the output-domain assertions passed in every process, and mints the four E1 verdicts SEPARATELY:
  ECONOMICS_OUTPUT_COMPARABILITY_E1
  MARGINAL_RECOVERY_UTILITY_ON_STEP_PATH_E1
  RECOVERY_PATH_VS_FUSED_BASELINE_E1
  GENERAL_END_TO_END_DEPLOYMENT_UTILITY = OPEN   (asserted OPEN unconditionally)
Cites the frozen historical wide-band quality gate; does NOT rerun recovery qualification. No GPU.
"""
import glob
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06T2-E1")
HIST_DIR = os.path.join(REPO, "runs", "rnn", "RNN-06T2")
ENVELOPE_MS = 250.0
Q_SESOI = 0.05


def stats(a):
    a = np.array(a, float)
    return {"n": len(a), "median": round(float(np.median(a)), 3), "p25": round(float(np.percentile(a, 25)), 3),
            "p75": round(float(np.percentile(a, 75)), 3), "p95": round(float(np.percentile(a, 95)), 3),
            "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)}


def main():
    runs = sorted(glob.glob(os.path.join(OUTDIR, "E1_ECON_run*.json")))
    assert len(runs) >= 2, f"need >=2 process starts, found {len(runs)}"
    ff, fs, rec, add_step, add_fused = [], [], [], [], []
    per_run, comps, domain_all = [], [], []
    for rp in runs:
        d = json.load(open(rp))
        raw = d["raw_warm_per_query_ms"]
        ff += raw["final_fused"]; fs += raw["final_step"]; rec += raw["recovery"]; add_step += raw["added_vs_step"]
        add_fused += [r - f for r, f in zip(raw["recovery"], raw["final_fused"])]
        comps.append(d["recovery_components_per_query_ms"])
        dev = d["output_domain_evidence"]
        domain_all.append({"process_index": d["process_index"], **{k: dev[k] for k in
                           ("ASSERTIONS_PASSED", "final_fused_all_in_vt", "final_step_all_in_vt",
                            "recovery_all_in_vt", "recovery_old_colidx_all_in_vt",
                            "recovery_fix_changed_output", "vt_size", "vt_min", "vt_max")}})
        per_run.append({"process_index": d["process_index"], "timing_method": d["timing_method"],
                        "shuffle_seed": d["shuffle_seed"], "compile_model_load_s": d["compile_model_load_s"],
                        "rec_median_ms": d["warm_per_query_ms"]["RECOVERY_ENABLED_EQUIVALENT_WORK"]["median"],
                        "memory": d["memory"]})

    # ---- MINT 1: ECONOMICS_OUTPUT_COMPARABILITY_E1 ----
    comparability_ok = all(
        r["ASSERTIONS_PASSED"] and r["final_fused_all_in_vt"] and r["final_step_all_in_vt"]
        and r["recovery_all_in_vt"] and (not r["recovery_old_colidx_all_in_vt"]) for r in domain_all)
    ECON_OUTPUT_COMPARABILITY = "QUALIFIED" if comparability_ok else "NOT_QUALIFIED"

    # ---- MINT 2: MARGINAL_RECOVERY_UTILITY_ON_STEP_PATH_E1 ----
    add_step_stats = stats(add_step)
    p95_add_step = add_step_stats["p95"]
    cost_ok = (p95_add_step <= ENVELOPE_MS)
    wide = json.load(open(os.path.join(HIST_DIR, "T1R_WIDE_RESULTS.json")))
    mc = wide["arms"]["MAX_CONFIDENCE"]["recovery_harm_vs_final"]
    q_delta = mc["accuracy_delta_vs_final"]; net = mc["net_recovery_count"]
    quality_ok = (net > 0 and q_delta >= Q_SESOI)
    if not comparability_ok:
        MARGINAL_UTILITY = "NOT_QUALIFIED"   # cannot claim utility on non-comparable outputs
    elif quality_ok and cost_ok:
        MARGINAL_UTILITY = "QUALIFIED"
    elif not cost_ok:
        MARGINAL_UTILITY = "COST_FAIL"
    else:
        MARGINAL_UTILITY = "NOT_QUALIFIED"

    # ---- MINT 3: RECOVERY_PATH_VS_FUSED_BASELINE_E1 (descriptive) ----
    add_fused_stats = stats(add_fused)
    p95_add_fused = add_fused_stats["p95"]
    FUSED_BASELINE = "COMPETITIVE_WITH_FUSED" if p95_add_fused <= ENVELOPE_MS else "NOT_COMPETITIVE_WITH_FUSED"

    # ---- MINT 4: GENERAL_END_TO_END_DEPLOYMENT_UTILITY = OPEN (unconditional) ----
    GENERAL_UTILITY = "OPEN"

    out = {"packet": "RNN-06T2-E1-ECON-AGGREGATE", "n_process_starts": len(runs),
           "timing_method": "randomized_interleaved_cycles_one_iter_per_arm",
           "output_domain_evidence_per_run": domain_all,
           "warm_per_query_ms": {"FINAL_FUSED_EQUIVALENT_WORK": stats(ff),
                                 "FINAL_STEP_EQUIVALENT_WORK": stats(fs),
                                 "RECOVERY_ENABLED_EQUIVALENT_WORK": stats(rec)},
           "added_latency_per_query_ms": {"vs_final_step_PRIMARY": add_step_stats,
                                          "vs_final_fused_descriptive": add_fused_stats},
           "recovery_components_per_query_ms_by_run": comps, "per_run": per_run,
           "envelope_ms_per_query": ENVELOPE_MS,
           "cited_frozen_quality_gate": {"source": "runs/rnn/RNN-06T2/T1R_WIDE_RESULTS.json",
                                         "q_delta_maxconf_vs_final_wide": q_delta,
                                         "net_recovery_count_wide": net, "Q_SESOI": Q_SESOI,
                                         "quality_ok": bool(quality_ok),
                                         "note": "cited from frozen T1R; recovery qualification NOT rerun in E1"},
           "cost_gate": {"p95_added_vs_step_ms": p95_add_step, "envelope_ms": ENVELOPE_MS, "cost_ok": bool(cost_ok)},
           "MINTS": {
               "ECONOMICS_OUTPUT_COMPARABILITY_E1": ECON_OUTPUT_COMPARABILITY,
               "MARGINAL_RECOVERY_UTILITY_ON_STEP_PATH_E1": MARGINAL_UTILITY,
               "RECOVERY_PATH_VS_FUSED_BASELINE_E1": {
                   "verdict": FUSED_BASELINE,
                   "p95_recovery_minus_fused_ms": p95_add_fused,
                   "median_recovery_minus_fused_ms": add_fused_stats["median"],
                   "note": "orthogonal step-vs-fused path cost; NOT the recovery premium"},
               "GENERAL_END_TO_END_DEPLOYMENT_UTILITY": GENERAL_UTILITY}}
    json.dump(out, open(os.path.join(OUTDIR, "E1_ECONOMICS.json"), "w"), indent=2, default=str)
    print("== RNN-06T2-E1 ECONOMICS SEMANTIC CLOSURE ==")
    print(f"FUSED median={out['warm_per_query_ms']['FINAL_FUSED_EQUIVALENT_WORK']['median']}ms "
          f"STEP median={out['warm_per_query_ms']['FINAL_STEP_EQUIVALENT_WORK']['median']}ms "
          f"REC median={out['warm_per_query_ms']['RECOVERY_ENABLED_EQUIVALENT_WORK']['median']}ms")
    print(f"added vs STEP (primary): median={add_step_stats['median']}ms p95={p95_add_step}ms envelope={ENVELOPE_MS}ms")
    print(f"added vs FUSED (descriptive): median={add_fused_stats['median']}ms p95={p95_add_fused}ms")
    print(f"MINT ECONOMICS_OUTPUT_COMPARABILITY_E1        = {ECON_OUTPUT_COMPARABILITY}")
    print(f"MINT MARGINAL_RECOVERY_UTILITY_ON_STEP_PATH_E1= {MARGINAL_UTILITY}")
    print(f"MINT RECOVERY_PATH_VS_FUSED_BASELINE_E1       = {FUSED_BASELINE} (p95 {p95_add_fused}ms)")
    print(f"MINT GENERAL_END_TO_END_DEPLOYMENT_UTILITY    = {GENERAL_UTILITY}")


if __name__ == "__main__":
    main()
