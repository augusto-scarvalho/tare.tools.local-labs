#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06B3 — frozen qualification spec + stress grid generator (order-stable, M=192).

Emits B3_QUALIFICATION_SPEC.json (b3QualificationSetSha256) + B3_STRESS_GRID.json
(b3StressGridSha256). Disjoint (distinct seed + generator + example-level) from P0/06B/B2-qual/
B2-calib/B3-calib. No GPU, no model. No seed screening.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_06b3_lib as lib  # noqa: E402

GENERATOR_VERSION = "rnn06b3_order_stable_state_load_v1"
MASTER_SEED = 20260817
M = 192
RESERVE = 16
DOSES = [1, 24, 48, 72, 96, 128, 152, 176]
ARMS = ["DS", "SS"]
S_STRATA = 3
N_PER_STRATUM = 64
TAU_HI = 0.75
SESOI = 0.20
SENTINEL = "REPEAT1"
U_LOW, U_HIGH = 1, 176

PRIOR = {"p0_calib": "779fb37af14eea0e36b25b0407f5aa32fc23dc84b8add51a2df4ee0ad88c45f3",
         "rnn06b_qual": "e351a4449796cdf71fa04b3f77fd9038f6950cb2672d23968bb22a5e031cf0ee",
         "b2_qual": "a92870a99babdb93b3e232549e2db26f8dec68c520b50cbc13dff56bf583ea10"}
B3_CALIB_SEED = 20260816

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06B3-MAMBA-CONTROLLED-STATE-LOAD-PERTURBATION")


def specs(seed):
    out, eid = [], 0
    for s in range(S_STRATA):
        for i in range(N_PER_STRATUM):
            out.append(lib.build_example_spec(seed, eid, s, M))
            eid += 1
    return out


def sig(e):
    return (e["target_key_slot"], e["target_val_slot"], tuple(e["load_positions"]),
            tuple(e["ds_key_slots"]), tuple(e["ss_key_slots"]))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    examples = specs(MASTER_SEED)
    spec = {"kind": "order_stable_state_load_v1", "generator_version": GENERATOR_VERSION,
            "master_seed": MASTER_SEED, "M": M, "reserve": RESERVE, "sentinel_scheme": SENTINEL,
            "scored_pool": lib.SCORED_POOL, "filler_pool": lib.FILLER_POOL,
            "s_strata": S_STRATA, "n_per_stratum": N_PER_STRATUM,
            "n_total": S_STRATA * N_PER_STRATUM, "examples": examples}
    qsha = lib.sha256_of_obj(spec)
    grid = {"generator_version": GENERATOR_VERSION, "M": M, "reserve": RESERVE, "doses": DOSES,
            "arms": ARMS, "s_strata": S_STRATA, "n_per_stratum": N_PER_STRATUM,
            "n_per_cell": S_STRATA * N_PER_STRATUM, "tau_hi": TAU_HI, "sesoi": SESOI,
            "trivial_region": [-0.05, 0.05], "u_low": U_LOW, "u_high": U_HIGH,
            "sentinel_scheme": SENTINEL, "min_sentinel_reserve": RESERVE,
            "robust_strata_required": 2, "ci_lower_bound_min": 0.05,
            "primary_endpoint": "paired_DS_constrained_loss_U1_to_U176",
            "shape_interior_margin": 0.10, "shape_cliff_step_fraction": 0.60}
    gsha = lib.sha256_of_obj(grid)

    calib_sigs = {sig(e) for e in specs(B3_CALIB_SEED)}
    my_sigs = {sig(e) for e in examples}
    overlap = len(my_sigs & calib_sigs)
    proof = {"b3QualificationSetSha256": qsha,
             "distinct_master_seed": {"b3_calib": B3_CALIB_SEED, "b3_qual": MASTER_SEED,
                                      "p0_calib": 20260811, "rnn06b_qual": 20260813,
                                      "b2_qual": 20260815},
             "sha_distinct_from_prior": {k: qsha != v for k, v in PRIOR.items()},
             "example_level_overlap_with_b3_calibration": overlap,
             "example_level_disjoint_from_b3_calibration": overlap == 0,
             "note": "P0/06B/B2 use different construction schemas; disjointness there via "
                     "distinct seeds+generator+SHAs. B3-calib shares schema -> example-level checked."}

    with open(os.path.join(OUTDIR, "B3_QUALIFICATION_SPEC.json"), "w") as f:
        json.dump({"b3QualificationSetSha256": qsha, "disjointness_proof": proof, **spec}, f)
    with open(os.path.join(OUTDIR, "B3_STRESS_GRID.json"), "w") as f:
        json.dump({"b3StressGridSha256": gsha, **grid}, f, indent=2)

    print(f"b3QualificationSetSha256 = {qsha}")
    print(f"b3StressGridSha256       = {gsha}")
    print(f"n_total={spec['n_total']} M={M} reserve={RESERVE} doses={DOSES}")
    print(f"disjoint_from_b3_calib={proof['example_level_disjoint_from_b3_calibration']} "
          f"(overlap={overlap})  sha_distinct={proof['sha_distinct_from_prior']}")


if __name__ == "__main__":
    main()
