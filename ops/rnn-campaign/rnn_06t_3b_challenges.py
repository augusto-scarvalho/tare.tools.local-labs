#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06T 3B — wide-target calibration + qualification sets (band [8,144], region strata).

Same v2 anti-oracle construction, but the target support is broadened to [8,144] so that no single
fixed schedule snapshot is guaranteed to have observed every target: for a target in a late region,
only a late snapshot has seen it. Region strata align to the schedule [38,76,115,153]:
  S0 [8,38]  S1 [39,76]  S2 [77,115]  S3 [116,144].
Calibration (seed 20260980) is used ONLY to freeze BEST_FIXED_SNAPSHOT; qualification (seed 20260981)
is fresh + disjoint. No GPU. No seed screening.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_06d_lib as D6  # noqa: E402

M = 192
K = 4
SCHEDULE = D6.schedule_slots(M, K)              # [38,76,115,153]
REGIONS = [(8, 38), (39, 76), (77, 115), (116, 144)]
N_PER_STRATUM = 48
CALIB_SEED = 20260980
QUAL_SEED = 20260981
PRIOR_SEEDS = {"d0_calib": 20260901, "d0_qual": 20260902, "t3a_qual": 20260970,
               "b3_qual": 20260817, "c06": 20260818}
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06T")


def specs(seed):
    out, eid = [], 0
    for s, (lo, hi) in enumerate(REGIONS):
        for _ in range(N_PER_STRATUM):
            out.append(D6.build_d0_example_spec(seed, eid, s, M, lo, hi)); eid += 1
    return out


def sig(e):
    return (e["target_slot"], e["target_key_slot"], e["target_val_slot"],
            tuple(e["load_key_slots"]), tuple(e["load_val_slots"]))


def emit(seed, name, path):
    examples = specs(seed)
    spec = {"kind": f"wide_target_3b_{name}", "generator_version": D6.GENERATOR_VERSION,
            "master_seed": seed, "M": M, "band": [8, 144], "regions": REGIONS, "K": K,
            "schedule": SCHEDULE, "s_strata": len(REGIONS), "n_per_stratum": N_PER_STRATUM,
            "n_total": len(REGIONS) * N_PER_STRATUM, "examples": examples}
    sha = D6.sha256_of_obj(spec)
    my = {sig(e) for e in examples}
    ov = {nm: len(my & {sig(e) for e in specs(sd)}) for nm, sd in PRIOR_SEEDS.items()}
    ov["other_3b_set"] = len(my & {sig(e) for e in specs(QUAL_SEED if seed == CALIB_SEED else CALIB_SEED)})
    spec["disjointness_proof"] = {"sha256": sha, "overlaps": ov, "disjoint_all": all(v == 0 for v in ov.values())}
    spec["setSha256"] = sha
    with open(path, "w") as f:
        json.dump(spec, f)
    print(f"{name}: sha={sha}  disjoint_all={spec['disjointness_proof']['disjoint_all']} overlaps={ov}")
    return sha


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    emit(CALIB_SEED, "calibration", os.path.join(OUTDIR, "T1_3B_CALIBRATION_SPEC.json"))
    emit(QUAL_SEED, "qualification", os.path.join(OUTDIR, "T1_3B_QUALIFICATION_SPEC.json"))
    print(f"schedule={SCHEDULE} regions={REGIONS} n_per_stratum={N_PER_STRATUM}")


if __name__ == "__main__":
    main()
