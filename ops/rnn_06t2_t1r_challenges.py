#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06T2-T1R challenge sets — narrow [8,64] qualification + wide [8,144] calib/qual.

06D v2 anti-oracle construction. Fresh disjoint seeds (20261xxx). Emits per-set setSha256 +
disjointness_proof (signature-overlap counts vs all prior seeds and vs each other). No GPU.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_06d_lib as D6  # noqa: E402

M = 192
K = 4
SCHEDULE = D6.schedule_slots(M, K)                     # [38,76,115,153]
N_PER_STRATUM = 48
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06T2")

NARROW = {"name": "narrow_qualification", "seed": 20261110, "band": [8, 64],
          "regions": [(8, 22), (23, 37), (38, 52), (53, 64)]}
WIDE_CALIB = {"name": "wide_calibration", "seed": 20261120, "band": [8, 144],
              "regions": [(8, 38), (39, 76), (77, 115), (116, 144)]}
WIDE_QUAL = {"name": "wide_qualification", "seed": 20261121, "band": [8, 144],
             "regions": [(8, 38), (39, 76), (77, 115), (116, 144)]}

# all prior RNN-06/06A/06T + T0R seeds to prove disjointness against
PRIOR_SEEDS = {"b3_qual": 20260817, "c06": 20260818, "d0_calib": 20260901, "d0_qual": 20260902,
               "t3a_qual": 20260970, "t3b_calib": 20260980, "t3b_qual": 20260981,
               "t0r_sp": 20261060}


def specs(seed, regions):
    out, eid = [], 0
    for s, (lo, hi) in enumerate(regions):
        for _ in range(N_PER_STRATUM):
            out.append(D6.build_d0_example_spec(seed, eid, s, M, lo, hi)); eid += 1
    return out


def sig(e):
    return (e["target_slot"], e["target_key_slot"], e["target_val_slot"],
            tuple(e["load_key_slots"]), tuple(e["load_val_slots"]))


def emit(cfg, path, sibling_seeds):
    examples = specs(cfg["seed"], cfg["regions"])
    spec = {"kind": f"t1r_{cfg['name']}", "generator_version": D6.GENERATOR_VERSION,
            "master_seed": cfg["seed"], "M": M, "band": cfg["band"], "regions": cfg["regions"],
            "K": K, "schedule": SCHEDULE, "s_strata": len(cfg["regions"]),
            "n_per_stratum": N_PER_STRATUM, "n_total": len(cfg["regions"]) * N_PER_STRATUM,
            "examples": examples}
    sha = D6.sha256_of_obj(spec)
    my = {sig(e) for e in examples}
    ov = {nm: len(my & {sig(e) for e in specs(sd, cfg["regions"])}) for nm, sd in PRIOR_SEEDS.items()}
    for nm, sd in sibling_seeds.items():
        # compare against sibling using ITS OWN regions (disjoint construction check)
        ov[nm] = len(my & {sig(e) for e in specs(sd["seed"], sd["regions"])})
    spec["disjointness_proof"] = {"sha256": sha, "overlaps": ov,
                                  "disjoint_all": all(v == 0 for v in ov.values())}
    spec["setSha256"] = sha
    with open(path, "w") as f:
        json.dump(spec, f)
    print(f"{cfg['name']}: sha={sha[:16]} disjoint_all={spec['disjointness_proof']['disjoint_all']} "
          f"n={spec['n_total']} band={cfg['band']}")
    return sha


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    emit(NARROW, os.path.join(OUTDIR, "T1R_NARROW_QUAL_SPEC.json"),
         {"wide_calib": WIDE_CALIB, "wide_qual": WIDE_QUAL})
    emit(WIDE_CALIB, os.path.join(OUTDIR, "T1R_WIDE_CALIB_SPEC.json"),
         {"narrow": NARROW, "wide_qual": WIDE_QUAL})
    emit(WIDE_QUAL, os.path.join(OUTDIR, "T1R_WIDE_QUAL_SPEC.json"),
         {"narrow": NARROW, "wide_calib": WIDE_CALIB})
    print(f"schedule={SCHEDULE} M={M} n_per_stratum={N_PER_STRATUM}")


if __name__ == "__main__":
    main()
