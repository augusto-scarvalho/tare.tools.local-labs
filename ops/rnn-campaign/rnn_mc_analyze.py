#!/usr/bin/env python
"""
RNN-04 analyzer: Pareto outputs (section 15), outcome classification (section 21), QWEN_GDN_TRANSPLANT_GATE
(section 22), heuristic labeling (section 23). Reads the experiment + benchmark + unittest artifacts.

Usage: python rnn_mc_analyze.py --dir <RNN-04 dir> [--out analysis.json --pareto pareto.csv]
"""
import argparse, csv, json, os

MARGIN = 0.02   # OPERATOR_HEURISTIC (section 23): NOT an empirically established noise floor.


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def run(args):
    d = args.dir
    R = load(d, "rnn04_results.json")
    ST = load(d, "RNN-04-benchmark-selftest.json")
    UT = load(d, "substrate_unittest.json")
    arms = R.get("arms", {})

    def acc(k):
        return arms.get(k, {}).get("acc")

    A, B = acc("A_base_single"), acc("B_grm")
    C = acc("C_equal_mem_single")
    B0, Post = acc("B0_residual_free"), acc("Post_moving_avg_free")
    SSC, Drand = acc("SSC_learned_k2"), acc("D_ssc_random_k2")
    repl = R.get("replication", {})
    gain_main = round(B - A, 4) if (A is not None and B is not None) else None
    gain_repl = repl.get("grm_minus_base")

    # ---- Pareto curves (section 15) ----
    curves = {}
    sw = R.get("sweep", {})
    curves["acc_vs_n_cached"] = [dict(n=v["N"], acc=v["acc"], cache_bytes=v["total_cache_bytes"],
                                      infer_ms=v["total_infer_ms"], agg_read_ms=v["aggregation_read_ms"])
                                 for v in sorted(sw.values(), key=lambda z: z["N"])]
    curves["acc_vs_seqlen"] = [dict(seq_len=int(L), base=v["base"], grm=v["grm"], N=v["N"])
                               for L, v in sorted(R.get("sweep_seqlen", {}).items(), key=lambda z: int(z[0]))]
    curves["acc_vs_distance"] = dict(base=arms.get("A_base_single", {}).get("dist_curve", {}),
                                     grm=arms.get("B_grm", {}).get("dist_curve", {}))

    pareto_rows = []
    for r in curves["acc_vs_n_cached"]:
        pareto_rows += [("acc_vs_n_cached", r["n"], "acc", r["acc"]),
                        ("acc_vs_cache_bytes", r["cache_bytes"], "acc", r["acc"]),
                        ("latency_vs_n_cached", r["n"], "total_infer_ms", r["infer_ms"]),
                        ("aggregation_read_ms_vs_n_cached", r["n"], "agg_read_ms", r["agg_read_ms"])]
    for r in curves["acc_vs_seqlen"]:
        pareto_rows += [("acc_vs_seqlen_base", r["seq_len"], "acc", r["base"]),
                        ("acc_vs_seqlen_grm", r["seq_len"], "acc", r["grm"])]
    for lbl, cv in (("base", curves["acc_vs_distance"]["base"]), ("grm", curves["acc_vs_distance"]["grm"])):
        for x, y in cv.items():
            pareto_rows.append((f"acc_vs_distance_{lbl}", int(x), "acc", y))

    # capability-per-byte / per-compute (section 15 goal)
    per = {}
    if gain_main is not None:
        b_cost = R.get("cost", {}).get("B_grm", {})
        cache_bytes = b_cost.get("total_cache_bytes")
        agg_ms = b_cost.get("aggregation_read_ms")
        per = dict(gain_main=gain_main, gain_replication=gain_repl,
                   grm_cache_bytes=cache_bytes, grm_agg_read_ms=agg_ms,
                   acc_gain_per_kb=(round(gain_main / (cache_bytes / 1024), 6) if cache_bytes else None),
                   equal_mem_single_acc=C, equal_mem_beats_grm=bool(C is not None and B is not None
                                                                    and C >= B - MARGIN))

    # ---- outcome classification (section 21) ----
    reasons = []
    repeatable = (gain_main is not None and gain_main >= MARGIN and gain_repl is not None and gain_repl > 0)
    if gain_main is None:
        outcome = "MC_IMPLEMENTATION_AMBIGUOUS"; reasons.append("missing arm accuracies")
    elif not repeatable and gain_main < MARGIN:
        outcome = "MC_REPRODUCED_NO_GAIN"; reasons.append(f"GRM-BASE={gain_main} < margin {MARGIN}")
    elif C is not None and C >= B - MARGIN:
        outcome = "MC_ONLY_HELPS_WITH_MORE_MEMORY"
        reasons.append(f"equal-byte single state C={C} matches/beats GRM={B}; gain attributable to bytes")
    else:
        outcome = "MC_REPRODUCED_POSITIVE"
        reasons.append(f"GRM>BASE by {gain_main} (main) and {gain_repl} (replication); "
                       f"exceeds equal-byte single control C={C}")

    # selection-policy signal (SSC learned vs random, section 11D)
    selection = None
    if SSC is not None and Drand is not None:
        selection = dict(ssc_learned=SSC, ssc_random=Drand, delta=round(SSC - Drand, 4),
                         policy_helps=bool(SSC - Drand >= MARGIN),
                         note="learned Top-k vs random selection at equal k (separates policy from 'more state exists')")
    # residual collapse observation (B0 ~ A expected for linear memory)
    collapse = None
    if B0 is not None and A is not None:
        collapse = dict(residual_free=B0, base=A, delta=round(B0 - A, 4),
                        matches_prediction=bool(abs(B0 - A) < 0.05),
                        note="Eq.7 residual collapses to single full state for LINEAR memory (predicted)")
    post_training = None
    if Post is not None and A is not None:
        post_training = dict(moving_avg_free=Post, base=A, delta=round(Post - A, 4),
                             note="POST_TRAINING_MC: param-free moving-average on FROZEN base weights (section 18)")

    # ---- QWEN_GDN_TRANSPLANT_GATE (section 22) ----
    cond = dict(
        benchmark_qualified=bool(ST.get("SYNTHETIC_DATASET_REPRODUCIBILITY") == "QUALIFIED"
                                 and ST.get("BENCHMARK_SELFTEST") == "PASS"),
        aggregation_qualified=bool(UT.get("AGGREGATION_UNIT_TEST") == "PASS"),
        basic_mc_repeatable_benefit=bool(repeatable),
        state_memory_cost_known=bool(R.get("cost", {}).get("B_grm", {}).get("state_bytes_per_req") is not None),
        compute_cost_known=bool(R.get("cost", {}).get("B_grm", {}).get("total_infer_ms") is not None),
        maps_to_qwen_gdn_state=True,  # substrate state is the [d_k,d_v] matrix = GDN recurrent-state shape
    )
    gate_pass = all(cond.values())
    gate = dict(conditions=cond, QWEN_GDN_TRANSPLANT_GATE=("PASS" if gate_pass else "DEFER"),
                caveat=("Substrate is plain linear-attention memory (paper Eq.2), a simplification of Qwen's "
                        "GATED delta rule; effect is small and toy-scale. PASS means 'mechanism reproduced + "
                        "plausibly mappable', NOT 'will improve Qwen'. Packet forbids touching Qwen GDN here."))

    out = dict(
        packet="RNN-04", MEMORY_AXIS=R.get("MEMORY_AXIS"), MC_TASK=R.get("MC_TASK"),
        headline=dict(base=A, grm=B, gain_main=gain_main, gain_replication=gain_repl,
                      equal_mem_single=C, ssc_learned=SSC, ssc_random=Drand,
                      residual_free=B0, post_training_moving_avg=Post),
        MC_OUTCOME=outcome, outcome_reasons=reasons, capability_per_resource=per,
        selection_policy=selection, residual_collapse=collapse, post_training_mc=post_training,
        gate=R.get("gate", {}), replication=repl, curves=curves,
        QWEN_GDN_TRANSPLANT_GATE=gate,
        thresholds=dict(MARGIN=MARGIN, label="OPERATOR_HEURISTIC (section 23): not a measured noise floor; "
                        "raw effects + replication direction are primary"),
    )
    json.dump(out, open(args.out, "w"), indent=2)
    with open(args.pareto, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["curve", "x", "metric", "y"])
        for row in pareto_rows:
            w.writerow(row)
    print(json.dumps(dict(MEMORY_AXIS=out["MEMORY_AXIS"], MC_OUTCOME=outcome, gain_main=gain_main,
                          gain_replication=gain_repl, equal_mem_single=C,
                          selection_policy=(selection or {}).get("policy_helps"),
                          QWEN_GDN_TRANSPLANT_GATE=gate["QWEN_GDN_TRANSPLANT_GATE"]), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pareto", default=None)
    a = ap.parse_args()
    a.out = a.out or os.path.join(a.dir, "rnn04_analysis.json")
    a.pareto = a.pareto or os.path.join(a.dir, "pareto.csv")
    run(a)
