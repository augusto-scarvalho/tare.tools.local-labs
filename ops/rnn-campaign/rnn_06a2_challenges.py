#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06A2 — deterministic held-out continuation-lifecycle challenge generator.

Emits runs/rnn/RNN-06A2-MAMBA-CONTINUATION/CONTINUATION_CHALLENGES.json and prints
lifecycleQualificationSetSha256. Token-id level, model-independent (ints in a safe
vocab range). Disjoint from RNN-06A's fixed lifecycle sequences by construction
(distinct seeded generator, length-16 sequences vs 06A's length-8/12 fixed lists).

Process-stable RNG via numpy PCG64/SeedSequence (NOT python hash()). No seed screening.
"""
import hashlib
import json
import os

import numpy as np

GENERATOR_VERSION = "rnn06a2_continuation_challenges_v1"
MASTER_SEED = 20260812
VOCAB_LO, VOCAB_HI = 106, 50100      # avoids special/low ids; within model vocab (50254)
PREFIX_LEN = 16
BOUNDARIES = [4, 8, 12]
CONTINUATION_LEN = 6                  # teacher-forced stream length for branch test E

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06A2-MAMBA-CONTINUATION")


def rng_for(*ints):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(ints))))


def sha256_of_obj(obj):
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def seq(salt, n):
    r = rng_for(MASTER_SEED, salt)
    return [int(x) for x in r.integers(VOCAB_LO, VOCAB_HI, size=n)]


def build():
    challenges = {
        "generator_version": GENERATOR_VERSION,
        "master_seed": MASTER_SEED,
        "vocab_lo": VOCAB_LO,
        "vocab_hi": VOCAB_HI,
        "prefix_len": PREFIX_LEN,
        "boundaries": BOUNDARIES,
        "continuation_len": CONTINUATION_LEN,
        # A: fresh determinism (two independent sequences)
        "determinism_seqs": [seq(0xA0, PREFIX_LEN), seq(0xA1, PREFIX_LEN)],
        # B/C/D: checkpoint sequences (greedy continuation from a boundary)
        "checkpoint_seqs": [seq(0xB0, PREFIX_LEN), seq(0xB1, PREFIX_LEN),
                            seq(0xB2, PREFIX_LEN)],
        # E: branch replay — one prefix, two DIFFERENT forced continuation streams
        "branch": {
            "prefix": seq(0xC0, PREFIX_LEN),
            "boundary": 8,
            "stream_1": seq(0xC1, CONTINUATION_LEN),
            "stream_2": seq(0xC2, CONTINUATION_LEN),
        },
        # G: neighbor request isolation — P fixed; Q1, Q2 differ; equal shape
        "isolation": {
            "P": seq(0xD0, PREFIX_LEN),
            "Q1": seq(0xD1, PREFIX_LEN),
            "Q2": seq(0xD2, PREFIX_LEN),
            "boundary": 8,
        },
    }
    return challenges


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ch = build()
    digest = sha256_of_obj(ch)
    ch_out = {"lifecycleQualificationSetSha256": digest, **ch}
    path = os.path.join(OUTDIR, "CONTINUATION_CHALLENGES.json")
    with open(path, "w") as f:
        json.dump(ch_out, f, indent=2)
    # disjointness witness vs RNN-06A fixed lists
    a06 = {
        "BASE12": [11, 42, 7, 128, 256, 3, 99, 64, 15, 201, 88, 333],
        "P": [11, 42, 7, 128, 256, 3, 99, 64],
        "Q": [22, 19, 300, 7, 45, 120, 8, 260],
        "Q2": [1, 2, 3, 4, 5, 6, 7, 8],
        "PREFIX20": [11, 42, 7, 128, 256, 3, 99, 64, 15, 201, 88, 333, 5, 6, 7, 8, 9, 10, 44, 77],
    }
    a06_seqs = [tuple(v) for v in a06.values()]
    my_seqs = ([tuple(s) for s in ch["determinism_seqs"]]
               + [tuple(s) for s in ch["checkpoint_seqs"]]
               + [tuple(ch["branch"]["prefix"])]
               + [tuple(ch["isolation"][k]) for k in ("P", "Q1", "Q2")])
    overlap = [list(s) for s in set(a06_seqs) & set(my_seqs)]
    print(f"lifecycleQualificationSetSha256 = {digest}")
    print(f"wrote {path}")
    print(f"disjoint_from_06A_exact_sequences = {len(overlap) == 0} (overlap={overlap})")


if __name__ == "__main__":
    main()
