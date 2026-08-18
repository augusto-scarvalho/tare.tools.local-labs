#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06T 3A — fresh disjoint exact-contract qualification set (06D semantics; official subject).

Same construction as RNN-06D (v2 anti-oracle: random target slot in [8,64], sentinel-pre, unique DS
load-post, M=192, K=4 schedule [38,76,115,153]) but a NEW master seed, cryptographically disjoint
from D0 calibration/qualification and all B3/C sets. No GPU. No seed screening.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_06d_lib as D6  # noqa: E402

MASTER_SEED = 20260970
M = 192
T_MIN, T_MAX = 8, 64
K = 4
SCHEDULE = D6.schedule_slots(M, K)          # [38,76,115,153]
S_STRATA = 3
N_PER_STRATUM = 64
PRIOR_SEEDS = {"p0_calib": 20260811, "rnn06b_qual": 20260813, "b2_calib": 20260814,
               "b2_qual": 20260815, "b3_calib": 20260816, "b3_qual": 20260817, "c06": 20260818,
               "d0_calib": 20260901, "d0_qual": 20260902}
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06T")


def specs(seed):
    out, eid = [], 0
    for s in range(S_STRATA):
        for _ in range(N_PER_STRATUM):
            out.append(D6.build_d0_example_spec(seed, eid, s, M, T_MIN, T_MAX)); eid += 1
    return out


def sig(e):
    return (e["target_slot"], e["target_key_slot"], e["target_val_slot"],
            tuple(e["load_key_slots"]), tuple(e["load_val_slots"]))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    examples = specs(MASTER_SEED)
    spec = {"kind": "official_transport_3a_v1", "generator_version": D6.GENERATOR_VERSION,
            "master_seed": MASTER_SEED, "M": M, "t_band": [T_MIN, T_MAX], "K": K, "schedule": SCHEDULE,
            "s_strata": S_STRATA, "n_per_stratum": N_PER_STRATUM, "n_total": S_STRATA * N_PER_STRATUM,
            "examples": examples}
    qsha = D6.sha256_of_obj(spec)
    my = {sig(e) for e in examples}
    overlaps = {name: len(my & {sig(e) for e in specs(sd)}) for name, sd in PRIOR_SEEDS.items()}
    proof = {"qualificationSetSha256_3A": qsha, "master_seed": MASTER_SEED,
             "example_level_overlaps": overlaps, "example_level_disjoint_all": all(v == 0 for v in overlaps.values())}
    with open(os.path.join(OUTDIR, "T1_3A_QUALIFICATION_SPEC.json"), "w") as f:
        json.dump({"qualificationSetSha256_3A": qsha, "disjointness_proof": proof, **spec}, f)
    print(f"qualificationSetSha256_3A = {qsha}")
    print(f"schedule={SCHEDULE} M={M} band=[{T_MIN},{T_MAX}] n_total={spec['n_total']}")
    print(f"example_level_disjoint_all={proof['example_level_disjoint_all']} overlaps={overlaps}")


if __name__ == "__main__":
    main()
