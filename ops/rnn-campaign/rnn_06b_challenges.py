#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06B — deterministic qualification spec + stress grid generator.

Emits (both under runs/rnn/RNN-06B-MAMBA-BASE/):
  * QUALIFICATION_SPEC.json  — abstract, model-independent per-example slot spec
                               (scored key/val slot perms, probe index, stratum id,
                               filler key/val slot perms). qualificationSetSha256.
  * STRESS_GRID.json         — dose ladder, conditions, tau's, N, strata. stressGridSha256.

Proves disjointness vs the quarantined P0 calibration set (calibrationSetSha256 = 779fb37a...)
at BOTH the sha level and the example (slot-tuple) level. Process-stable numpy PCG64 RNG;
NO seed screening. No GPU, no model.
"""
import hashlib
import json
import os

import numpy as np

GENERATOR_VERSION = "rnn06b_mqar_matched_control_v1"
MASTER_SEED = 20260813
POOL_SIZE = 256           # scored key pool = scored value pool size
FILLER_POOL_SIZE = 256    # disjoint filler key pool = filler value pool size
DOSES = [8, 16, 24, 32, 48, 64, 96, 128]
P_MIN, P_MAX = min(DOSES), max(DOSES)
CONDITIONS = ["MP", "LC"]
S_STRATA = 3
N_PER_STRATUM = 64
TAU_HI, TAU_LO = 0.75, 0.45
LOW_DOSE = P_MIN

# P0 calibration identity (quarantined)
P0_CALIBRATION_SHA = "779fb37af14eea0e36b25b0407f5aa32fc23dc84b8add51a2df4ee0ad88c45f3"

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06B-MAMBA-BASE")
P0_CALIB = os.path.join(REPO, "runs", "rnn", "RNN-06-P0", "calibration_examples.json")


def rng_for(*ints):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(ints))))


def sha256_of_obj(obj):
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_spec():
    examples = []
    ex_id = 0
    for s in range(S_STRATA):
        for i in range(N_PER_STRATUM):
            r = rng_for(MASTER_SEED, 0x06B, s, i)
            key_slots = r.permutation(POOL_SIZE)[:P_MAX].tolist()
            val_slots = r.permutation(POOL_SIZE)[:P_MAX].tolist()
            probe_index = int(r.integers(0, P_MIN))
            fk_slots = r.permutation(FILLER_POOL_SIZE)[:P_MAX].tolist()
            fv_slots = r.permutation(FILLER_POOL_SIZE)[:P_MAX].tolist()
            examples.append({"ex_id": ex_id, "stratum": s,
                             "key_slots": [int(x) for x in key_slots],
                             "val_slots": [int(x) for x in val_slots],
                             "probe_index": probe_index,
                             "filler_key_slots": [int(x) for x in fk_slots],
                             "filler_val_slots": [int(x) for x in fv_slots]})
            ex_id += 1
    spec = {
        "kind": "mqar_assoc_recall_matched_control_v1",
        "generator_version": GENERATOR_VERSION,
        "master_seed": MASTER_SEED,
        "pool_size": POOL_SIZE, "filler_pool_size": FILLER_POOL_SIZE,
        "p_min": P_MIN, "p_max": P_MAX,
        "s_strata": S_STRATA, "n_per_stratum": N_PER_STRATUM,
        "n_total": S_STRATA * N_PER_STRATUM,
        "examples": examples,
    }
    return spec


def build_grid():
    return {
        "generator_version": GENERATOR_VERSION,
        "doses": DOSES, "conditions": CONDITIONS,
        "s_strata": S_STRATA, "n_per_stratum": N_PER_STRATUM,
        "n_per_cell": S_STRATA * N_PER_STRATUM,
        "tau_hi": TAU_HI, "tau_lo": TAU_LO, "low_dose": LOW_DOSE,
        "monotonicity_tolerance": 0.05, "max_violation": 0.10,
        "min_mid_band_doses": 2, "robust_strata_required": 2,
        "confound_high_doses": [96, 128], "confound_min_separation": 0.15,
        "primary_endpoint": "MP_constrained_retrieval_accuracy",
    }


def disjointness_vs_p0(spec):
    proof = {"calibrationSetSha256": P0_CALIBRATION_SHA,
             "p0_calibration_present": os.path.isfile(P0_CALIB)}
    my = {(tuple(e["key_slots"]), tuple(e["val_slots"]), e["probe_index"]) for e in spec["examples"]}
    if os.path.isfile(P0_CALIB):
        p0 = json.load(open(P0_CALIB))
        p0ex = p0.get("examples", [])
        p0set = {(tuple(e["key_slots"]), tuple(e["val_slots"]), e["probe_index"]) for e in p0ex}
        overlap = my & p0set
        proof["p0_n_examples"] = len(p0ex)
        proof["example_level_overlap_count"] = len(overlap)
        proof["example_level_disjoint"] = (len(overlap) == 0)
    else:
        proof["example_level_overlap_count"] = None
        proof["example_level_disjoint"] = None
    return proof


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    spec = build_spec()
    qsha = sha256_of_obj(spec)
    grid = build_grid()
    gsha = sha256_of_obj(grid)
    proof = disjointness_vs_p0(spec)
    proof["qualificationSetSha256"] = qsha
    proof["sha_disjoint_from_calibration"] = (qsha != P0_CALIBRATION_SHA)

    with open(os.path.join(OUTDIR, "QUALIFICATION_SPEC.json"), "w") as f:
        json.dump({"qualificationSetSha256": qsha, "disjointness_proof": proof, **spec}, f)
    with open(os.path.join(OUTDIR, "STRESS_GRID.json"), "w") as f:
        json.dump({"stressGridSha256": gsha, **grid}, f, indent=2)

    print(f"qualificationSetSha256 = {qsha}")
    print(f"stressGridSha256       = {gsha}")
    print(f"n_total_examples       = {spec['n_total']}")
    print(f"sha_disjoint_from_calib= {proof['sha_disjoint_from_calibration']}")
    print(f"example_level_disjoint = {proof['example_level_disjoint']} "
          f"(overlap={proof['example_level_overlap_count']}, p0_n={proof.get('p0_n_examples')})")


if __name__ == "__main__":
    main()
