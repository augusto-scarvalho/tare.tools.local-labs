#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06T2-T1R economics aggregation + verdict. Combines RAW warm samples across process starts,
recomputes warm stats, and mints END_TO_END_RECOVERY_UTILITY_T1R against the frozen envelope
(p95 of RECOVERY-FINAL_STEP <= 250 ms) AND the wide quality/net-recovery gates. No GPU.
"""
import glob
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06T2")
ENVELOPE_MS = 250.0
Q_SESOI = 0.05


def stats(a):
    a = np.array(a, float)
    return {"n": len(a), "median": round(float(np.median(a)), 3), "p25": round(float(np.percentile(a, 25)), 3),
            "p75": round(float(np.percentile(a, 75)), 3), "p95": round(float(np.percentile(a, 95)), 3),
            "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)}


def main():
    runs = sorted(glob.glob(os.path.join(OUTDIR, "T1R_ECON_run*.json")))
    assert runs, "no econ run files"
    ff, fs, rec, add_step, add_fused = [], [], [], [], []
    per_run = []
    comps = []
    for rp in runs:
        d = json.load(open(rp))
        raw = d["raw_warm_per_query_ms"]
        ff += raw["final_fused"]; fs += raw["final_step"]; rec += raw["recovery"]; add_step += raw["added_vs_step"]
        add_fused += [r - f for r, f in zip(raw["recovery"], raw["final_fused"])]
        comps.append(d["recovery_components_per_query_ms"])
        per_run.append({"process_index": d["process_index"], "compile_model_load_s": d["compile_model_load_s"],
                        "cold_batch_s": d["cold_batch_s"],
                        "rec_median_ms": d["warm_per_query_ms"]["RECOVERY_ENABLED_EQUIVALENT_WORK"]["median"],
                        "memory": d["memory"]})

    wide = json.load(open(os.path.join(OUTDIR, "T1R_WIDE_RESULTS.json")))
    mc = wide["arms"]["MAX_CONFIDENCE"]["recovery_harm_vs_final"]
    q_delta = mc["accuracy_delta_vs_final"]; net = mc["net_recovery_count"]

    add_step_stats = stats(add_step)
    p95_add_step = add_step_stats["p95"]
    quality_ok = (net > 0 and q_delta >= Q_SESOI)
    cost_ok = (p95_add_step <= ENVELOPE_MS)
    if not quality_ok:
        verdict = "NOT_QUALIFIED"
    elif cost_ok:
        verdict = "QUALIFIED"
    else:
        verdict = "COST_FAIL"

    out = {"packet": "RNN-06T2-ECON-AGGREGATE", "n_process_starts": len(runs),
           "warm_per_query_ms": {"FINAL_FUSED_EQUIVALENT_WORK": stats(ff),
                                 "FINAL_STEP_EQUIVALENT_WORK": stats(fs),
                                 "RECOVERY_ENABLED_EQUIVALENT_WORK": stats(rec)},
           "added_latency_per_query_ms": {"vs_final_step_PRIMARY": add_step_stats,
                                          "vs_final_fused_descriptive": stats(add_fused)},
           "recovery_components_per_query_ms_by_run": comps,
           "per_run": per_run,
           "envelope_ms_per_query": ENVELOPE_MS, "envelope_gate_statistic": "p95(added_vs_final_step)",
           "p95_added_vs_step_ms": p95_add_step,
           "quality_gate": {"q_delta_maxconf_vs_final_wide": q_delta, "net_recovery_count_wide": net,
                            "Q_SESOI": Q_SESOI, "quality_ok": bool(quality_ok)},
           "cost_gate": {"p95_added_vs_step_ms": p95_add_step, "envelope_ms": ENVELOPE_MS, "cost_ok": bool(cost_ok)},
           "primary_comparator": "RECOVERY_ENABLED - FINAL_STEP_EQUIVALENT_WORK",
           "END_TO_END_RECOVERY_UTILITY_T1R": verdict}
    json.dump(out, open(os.path.join(OUTDIR, "T1R_ECONOMICS.json"), "w"), indent=2, default=str)
    print(f"FUSED median={out['warm_per_query_ms']['FINAL_FUSED_EQUIVALENT_WORK']['median']}ms "
          f"STEP median={out['warm_per_query_ms']['FINAL_STEP_EQUIVALENT_WORK']['median']}ms "
          f"REC median={out['warm_per_query_ms']['RECOVERY_ENABLED_EQUIVALENT_WORK']['median']}ms")
    print(f"added vs STEP (primary): median={add_step_stats['median']}ms p95={p95_add_step}ms envelope={ENVELOPE_MS}ms")
    print(f"added vs FUSED (descriptive): median={out['added_latency_per_query_ms']['vs_final_fused_descriptive']['median']}ms")
    print(f"quality: q_delta={q_delta} net={net} quality_ok={quality_ok}  cost_ok={cost_ok}")
    print(f"END_TO_END_RECOVERY_UTILITY_T1R = {verdict}")


if __name__ == "__main__":
    main()
