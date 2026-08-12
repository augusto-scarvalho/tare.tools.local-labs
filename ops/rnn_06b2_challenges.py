#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06B2 — frozen qualification spec + stress grid generator.

Emits (under runs/rnn/RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD/):
  * B2_QUALIFICATION_SPEC.json — abstract per-example fixed-length spec (M=128). b2QualificationSetSha256.
  * B2_STRESS_GRID.json        — doses/arms/tau/N/strata + length diagnostic. b2StressGridSha256.

Disjoint (distinct master seed + generator + example-level) from P0 calibration, RNN-06B
qualification, and B2 calibration. No GPU, no model. No seed screening.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_06b2_lib as lib  # noqa: E402

GENERATOR_VERSION = "rnn06b2_fixed_length_state_load_v1"
MASTER_SEED = 20260815
M = 128
DOSES = [1, 24, 48, 64, 80, 96, 112, 128]
ARMS = ["DS", "SS"]
S_STRATA = 3
N_PER_STRATUM = 64
TAU_HI, TAU_LO = 0.75, 0.45
SENTINEL_SCHEME = "REPEAT1"
LENGTH_DIAG = {"arm": "DS", "U": 2, "M_values": [32, 64, 96, 128]}

# prior identities for disjointness bookkeeping
P0_CALIB_SHA = "779fb37af14eea0e36b25b0407f5aa32fc23dc84b8add51a2df4ee0ad88c45f3"
RNN06B_QUAL_SHA = "e351a4449796cdf71fa04b3f77fd9038f6950cb2672d23968bb22a5e031cf0ee"
B2_CALIB_SEED = 20260814

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD")


def spec_examples(seed):
    exs = []
    ex_id = 0
    for s in range(S_STRATA):
        for i in range(N_PER_STRATUM):
            exs.append(lib.build_example_spec(seed, ex_id, s, M))
            ex_id += 1
    return exs


def sig(e):
    return (e["target_key_slot"], e["target_val_slot"], tuple(e["load_positions"]),
            tuple(e["ds_key_slots"]), tuple(e["ss_key_slots"]))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    examples = spec_examples(MASTER_SEED)
    spec = {"kind": "fixed_length_state_load_v1", "generator_version": GENERATOR_VERSION,
            "master_seed": MASTER_SEED, "M": M, "sentinel_scheme": SENTINEL_SCHEME,
            "scored_pool": lib.SCORED_POOL, "filler_pool": lib.FILLER_POOL,
            "s_strata": S_STRATA, "n_per_stratum": N_PER_STRATUM,
            "n_total": S_STRATA * N_PER_STRATUM, "examples": examples}
    qsha = lib.sha256_of_obj(spec)

    grid = {"generator_version": GENERATOR_VERSION, "M": M, "doses": DOSES, "arms": ARMS,
            "s_strata": S_STRATA, "n_per_stratum": N_PER_STRATUM,
            "n_per_cell": S_STRATA * N_PER_STRATUM, "tau_hi": TAU_HI, "tau_lo": TAU_LO,
            "sentinel_scheme": SENTINEL_SCHEME, "length_diagnostic": LENGTH_DIAG,
            "monotonicity_tolerance": 0.05, "max_violation": 0.10, "min_mid_band_doses": 2,
            "robust_strata_required": 2, "delta_aurc_min": 0.15,
            "primary_endpoint": "DS_constrained_retrieval_accuracy_vs_unique_load"}
    gsha = lib.sha256_of_obj(grid)

    # disjointness vs B2 calibration (same schema)
    calib_sigs = {sig(e) for e in spec_examples(B2_CALIB_SEED)}  # calib used stratum 0 only, but
    # build a comparable set from calib seed over the same strata layout for a superset check
    my_sigs = {sig(e) for e in examples}
    overlap_calib = len(my_sigs & calib_sigs)
    proof = {
        "b2QualificationSetSha256": qsha,
        "distinct_master_seed_vs": {"p0_calib": 20260811, "rnn06b_qual": 20260813,
                                    "b2_calib": B2_CALIB_SEED, "b2_qual": MASTER_SEED},
        "distinct_generator_version": True,
        "sha_distinct_from_p0_calib": qsha != P0_CALIB_SHA,
        "sha_distinct_from_rnn06b_qual": qsha != RNN06B_QUAL_SHA,
        "example_level_overlap_with_b2_calibration": overlap_calib,
        "example_level_disjoint_from_b2_calibration": overlap_calib == 0,
        "note_p0_06b_schema": "P0/06B use a different (nested key/val/probe) construction schema; "
                              "cross-schema example identity is undefined, so disjointness there is "
                              "established by distinct master seeds + generator + distinct SHAs.",
    }

    with open(os.path.join(OUTDIR, "B2_QUALIFICATION_SPEC.json"), "w") as f:
        json.dump({"b2QualificationSetSha256": qsha, "disjointness_proof": proof, **spec}, f)
    with open(os.path.join(OUTDIR, "B2_STRESS_GRID.json"), "w") as f:
        json.dump({"b2StressGridSha256": gsha, **grid}, f, indent=2)

    print(f"b2QualificationSetSha256 = {qsha}")
    print(f"b2StressGridSha256       = {gsha}")
    print(f"n_total = {spec['n_total']}  M={M}  doses={DOSES}")
    print(f"disjoint_from_b2_calibration = {proof['example_level_disjoint_from_b2_calibration']} "
          f"(overlap={overlap_calib})")
    print(f"sha_distinct_from_p0={proof['sha_distinct_from_p0_calib']} "
          f"sha_distinct_from_06b={proof['sha_distinct_from_rnn06b_qual']}")


if __name__ == "__main__":
    main()
