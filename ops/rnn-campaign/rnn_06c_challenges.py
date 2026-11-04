#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06C — fresh held-out historical-info spec generator (order-stable, M=192).

Emits HISTORICAL_INFO_SPEC.json (historicalInfoSetSha256). Disjoint (distinct seed + generator +
example-level) from P0 / 06B / B2-calib / B2-qual / B3-calib / B3-qual. No GPU, no model. No seed
screening.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_06b3_lib as lib  # noqa: E402

GENERATOR_VERSION = "rnn06c_historical_info_v1"
MASTER_SEED = 20260818
M = 192
S_STRATA = 3
N_PER_STRATUM = 64

PRIOR_SEEDS = {"p0_calib": 20260811, "rnn06b_qual": 20260813, "b2_calib": 20260814,
               "b2_qual": 20260815, "b3_calib": 20260816, "b3_qual": 20260817}
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06C-MAMBA-HISTORICAL-INFO")


def specs(seed):
    out, eid = [], 0
    for s in range(S_STRATA):
        for i in range(N_PER_STRATUM):
            out.append(lib.build_example_spec(seed, eid, s, M))
            eid += 1
    return out


def sig(e):
    return (e["target_key_slot"], e["target_val_slot"], tuple(e["load_positions"]),
            tuple(e["ds_key_slots"]))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    examples = specs(MASTER_SEED)
    spec = {"kind": "historical_info_v1", "generator_version": GENERATOR_VERSION,
            "master_seed": MASTER_SEED, "M": M, "s_strata": S_STRATA,
            "n_per_stratum": N_PER_STRATUM, "n_total": S_STRATA * N_PER_STRATUM,
            "examples": examples}
    hsha = lib.sha256_of_obj(spec)

    my = {sig(e) for e in examples}
    overlaps = {}
    for name, sd in PRIOR_SEEDS.items():
        overlaps[name] = len(my & {sig(e) for e in specs(sd)})
    proof = {"historicalInfoSetSha256": hsha, "master_seed": MASTER_SEED,
             "example_level_overlaps": overlaps,
             "example_level_disjoint_all": all(v == 0 for v in overlaps.values())}

    with open(os.path.join(OUTDIR, "HISTORICAL_INFO_SPEC.json"), "w") as f:
        json.dump({"historicalInfoSetSha256": hsha, "disjointness_proof": proof, **spec}, f)
    print(f"historicalInfoSetSha256 = {hsha}")
    print(f"n_total={spec['n_total']} M={M}")
    print(f"example_level_disjoint_all = {proof['example_level_disjoint_all']}  overlaps={overlaps}")


if __name__ == "__main__":
    main()
