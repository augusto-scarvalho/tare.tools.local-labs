#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06D0 — frozen qualification spec + snapshot schedule generator (anti-oracle, M=192).

Emits D0_QUALIFICATION_SPEC.json (qualificationSetSha256) + SNAPSHOT_SCHEDULE.json
(snapshotScheduleSha256). K and the target band are the calibration-frozen configuration
(CHOSEN_K, T_MIN/T_MAX). Disjoint (distinct seed + generator + example-level) from all prior sets
including the D0 calibration set. No GPU, no model. No seed screening.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_06d_lib as lib  # noqa: E402

GENERATOR_VERSION = lib.GENERATOR_VERSION
MASTER_SEED = 20260902           # distinct from D0 calib (20260901) and all prior
CALIB_SEED = 20260901
M = 192
T_MIN, T_MAX = 8, 64             # calibration-frozen band
# FROZEN_K = 4 (see D0_CALIBRATION_DECISION.md). Raw parsimony (smallest adequate K) picked K=2, but
# at K=2 the schedule [64,128] places every snapshot at/after the target band (t<=64) -> no
# pre-target distractor snapshots -> the target-agnostic selection is trivial. K=4 is the SMALLEST
# adequate K that also yields a pre-target snapshot (slot 38) for part of the band, making the
# anti-oracle selection non-trivial. Ceiling is identical to K=2 (OB-FINAL=0.688), so this is an
# experimental-validity choice, not an effect-size choice. Frozen BEFORE any qualification outcome.
FROZEN_K = 4
S_STRATA = 3
N_PER_STRATUM = 64

PRIOR_SEEDS = {"p0_calib": 20260811, "rnn06b_qual": 20260813, "b2_calib": 20260814,
               "b2_qual": 20260815, "b3_calib": 20260816, "b3_qual": 20260817,
               "c06": 20260818, "d0_calib": CALIB_SEED}
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06D")


def specs(seed):
    out, eid = [], 0
    for s in range(S_STRATA):
        for _ in range(N_PER_STRATUM):
            out.append(lib.build_d0_example_spec(seed, eid, s, M, T_MIN, T_MAX)); eid += 1
    return out


def sig(e):
    return (e["target_slot"], e["target_key_slot"], e["target_val_slot"],
            tuple(e["load_key_slots"]), tuple(e["load_val_slots"]))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    # K is frozen by the calibration decision (see D0_CALIBRATION_DECISION.md). We deliberately do
    # NOT read the raw parsimony pick (chosen_K) from D0_CALIBRATION.json; the decision supersedes it
    # to FROZEN_K for anti-oracle validity.
    k = FROZEN_K
    schedule = lib.schedule_slots(M, k)

    examples = specs(MASTER_SEED)
    spec = {"kind": "anti_oracle_random_target_v1", "generator_version": GENERATOR_VERSION,
            "master_seed": MASTER_SEED, "M": M, "t_band": [T_MIN, T_MAX], "s_strata": S_STRATA,
            "n_per_stratum": N_PER_STRATUM, "n_total": S_STRATA * N_PER_STRATUM, "examples": examples}
    qsha = lib.sha256_of_obj(spec)
    sched = {"generator_version": GENERATOR_VERSION, "K": k, "schedule_slots": schedule, "M": M,
             "final_slot": M - 1}
    ssha = lib.sha256_of_obj({kk: sched[kk] for kk in ("K", "schedule_slots", "M", "final_slot",
                                                       "generator_version")})

    my = {sig(e) for e in examples}
    overlaps = {name: len(my & {sig(e) for e in specs(sd)}) for name, sd in PRIOR_SEEDS.items()}
    proof = {"qualificationSetSha256": qsha, "master_seed": MASTER_SEED,
             "example_level_overlaps": overlaps,
             "example_level_disjoint_all": all(v == 0 for v in overlaps.values()),
             "note": "Prior P0/06B/B2/B3/06C use different construction schemas; here disjointness "
                     "is via distinct seed+generator; overlap vs same-schema D0 calib is the "
                     "binding check."}

    with open(os.path.join(OUTDIR, "D0_QUALIFICATION_SPEC.json"), "w") as f:
        json.dump({"qualificationSetSha256": qsha, "disjointness_proof": proof, **spec}, f)
    with open(os.path.join(OUTDIR, "SNAPSHOT_SCHEDULE.json"), "w") as f:
        json.dump({"snapshotScheduleSha256": ssha, **sched}, f, indent=2)

    print(f"qualificationSetSha256 = {qsha}")
    print(f"snapshotScheduleSha256 = {ssha}")
    print(f"K={k} schedule_slots={schedule} M={M} t_band=[{T_MIN},{T_MAX}] n_total={spec['n_total']}")
    print(f"example_level_disjoint_all={proof['example_level_disjoint_all']} overlaps={overlaps}")


if __name__ == "__main__":
    main()
