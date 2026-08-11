#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06B2 / 06C shared construction library — fixed-length unique-state-load MQAR.

Single source of truth for the token construction used by B2 calibration, B2 qualification,
and 06C, so all three share an identical, deterministic, disjoint-pool build. Token-id level,
process-stable numpy PCG64 RNG. No GPU, no model here.

Construction (fixed total length regardless of unique-load dose U):
  * M association slots, each `key = value \n` (4 tokens); then query `target_key =` (2 tokens).
    Total tokens = 4*M + 2 — FIXED for a given M across all U.
  * Slot 0 = TARGET (scored-space key/value). Query at the very end. target->query gap = the
    M-1 intervening slots = FIXED across all U.
  * Dose U = number of UNIQUE active bindings INCLUDING target (1..M). U-1 "load" bindings are
    placed at the first U-1 of a per-example permutation of the non-target slots [1..M-1];
    the remaining M-U slots are SENTINEL (low-information).
  * Arm DS: load bindings from a DISJOINT filler key/value space (general state load, no
    same-scored-space competition). Arm SS: load bindings from the SCORED space (general load
    + same-space competition). Everything else identical.
  * Sentinel schemes (predeclared candidate family): REPEAT1 (one fixed sentinel pair repeated)
    and CYCLE4 (cycle 4 fixed sentinel pairs by ordinal). Sentinel space is disjoint from
    scored/filler/target.

Scoring: constrained argmax over the scored value vocabulary at the final position; gold =
target value. chance = 1/|scored value vocab|.
"""
import hashlib
import json
import re

import numpy as np

SCORED_POOL = 256
FILLER_POOL = 256
N_SENTINEL = 8
SENTINEL_SCHEMES = ["REPEAT1", "CYCLE4"]


def rng_for(*ints):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(ints))))


def sha256_of_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def build_pools(tok, seed):
    """5 mutually-disjoint single-token pools + separators."""
    vocab = int(getattr(tok, "vocab_size", 0) or len(tok))
    num_re = re.compile(r"^\s?\d{2,4}$")
    word_re = re.compile(r"^\s?[A-Za-z]{3,}$")
    num_ids, word_ids = [], []
    for tid in range(vocab):
        try:
            s = tok.decode([tid])
        except Exception:
            continue
        if num_re.match(s):
            num_ids.append(tid)
        elif word_re.match(s):
            word_ids.append(tid)

    def keep_single(ids):
        seen, out = set(), []
        for tid in ids:
            s = tok.decode([tid])
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) == 1 and enc[0] == tid and s not in seen:
                seen.add(s)
                out.append(tid)
        return out

    num_ids = keep_single(num_ids)
    word_ids = keep_single(word_ids)

    def perm(ids, salt):
        r = rng_for(seed, salt)
        return [int(x) for x in np.array(ids)[r.permutation(len(ids))]]

    words = perm(word_ids, 0xB0B2)
    nums = perm(num_ids, 0xA11CE)
    need_w = SCORED_POOL + FILLER_POOL + N_SENTINEL
    need_n = SCORED_POOL + FILLER_POOL + N_SENTINEL
    assert len(words) >= need_w and len(nums) >= need_n, \
        f"insufficient single-token pools: words={len(words)} nums={len(nums)}"
    scored_keys = words[:SCORED_POOL]
    filler_keys = words[SCORED_POOL:SCORED_POOL + FILLER_POOL]
    sentinel_keys = words[SCORED_POOL + FILLER_POOL:SCORED_POOL + FILLER_POOL + N_SENTINEL]
    scored_vals = nums[:SCORED_POOL]
    filler_vals = nums[SCORED_POOL:SCORED_POOL + FILLER_POOL]
    sentinel_vals = nums[SCORED_POOL + FILLER_POOL:SCORED_POOL + FILLER_POOL + N_SENTINEL]
    sentinel_pairs = list(zip(sentinel_keys, sentinel_vals))

    def one(t):
        enc = tok.encode(t, add_special_tokens=False)
        return enc[-1] if enc else None
    seps = {"eq": one("="), "nl": one("\n")}

    pools = {"scored_keys": scored_keys, "scored_vals": scored_vals,
             "filler_keys": filler_keys, "filler_vals": filler_vals,
             "sentinel_pairs": [list(p) for p in sentinel_pairs], "seps": seps}
    # disjointness assertion
    allsets = [set(scored_keys), set(filler_keys), set(sentinel_keys),
               set(scored_vals), set(filler_vals), set(sentinel_vals)]
    for a in range(len(allsets)):
        for b in range(a + 1, len(allsets)):
            assert not (allsets[a] & allsets[b]), "pool overlap"
    meta = {"n_word_single": len(words), "n_num_single": len(nums),
            "scored_keys_sha256": sha256_of_obj(scored_keys),
            "scored_vals_sha256": sha256_of_obj(scored_vals),
            "filler_keys_sha256": sha256_of_obj(filler_keys),
            "filler_vals_sha256": sha256_of_obj(filler_vals),
            "sentinel_pairs_sha256": sha256_of_obj(pools["sentinel_pairs"]),
            "eq_id": seps["eq"], "nl_id": seps["nl"],
            "sample_target_keys": [tok.decode([i]) for i in scored_keys[:3]],
            "sample_sentinel": [[tok.decode([p[0]]), tok.decode([p[1]])]
                                for p in pools["sentinel_pairs"][:2]]}
    return pools, meta


def build_example_spec(seed, ex_id, stratum, M):
    """Deterministic per-example assignment for fixed slot count M."""
    r = rng_for(seed, 0x06B2, stratum, ex_id)
    target_key_slot = int(r.integers(0, SCORED_POOL))
    target_val_slot = int(r.integers(0, SCORED_POOL))
    # SS load key/val slots: scored-pool indices disjoint from the target key/val slots
    ss_keys = [s for s in r.permutation(SCORED_POOL).tolist() if s != target_key_slot][:M - 1]
    ss_vals = [s for s in r.permutation(SCORED_POOL).tolist() if s != target_val_slot][:M - 1]
    ds_keys = r.permutation(FILLER_POOL).tolist()[:M - 1]
    ds_vals = r.permutation(FILLER_POOL).tolist()[:M - 1]
    load_positions = (r.permutation(M - 1) + 1).tolist()   # permutation of slots [1..M-1]
    return {"ex_id": ex_id, "stratum": stratum, "M": M,
            "target_key_slot": target_key_slot, "target_val_slot": target_val_slot,
            "ss_key_slots": [int(x) for x in ss_keys], "ss_val_slots": [int(x) for x in ss_vals],
            "ds_key_slots": [int(x) for x in ds_keys], "ds_val_slots": [int(x) for x in ds_vals],
            "load_positions": [int(x) for x in load_positions]}


def _sentinel_for(scheme, ordinal, pairs):
    if scheme == "REPEAT1":
        return pairs[0]
    if scheme.startswith("CYCLE"):
        k = int(scheme[5:]) if len(scheme) > 5 else 4
        return pairs[ordinal % min(k, len(pairs))]
    raise ValueError(f"unknown sentinel scheme {scheme}")


def materialize(spec, M, U, arm, sentinel_scheme, pools):
    """Return (token list, gold token). Fixed length 4*M+2 for all U. Target at slot 0."""
    sk, sv = pools["scored_keys"], pools["scored_vals"]
    fk, fv = pools["filler_keys"], pools["filler_vals"]
    pairs = pools["sentinel_pairs"]
    eq, nl = pools["seps"]["eq"], pools["seps"]["nl"]
    assert 1 <= U <= M
    # decide slot content
    load_pos = spec["load_positions"][:U - 1]          # slots that carry a unique load binding
    load_set = set(load_pos)
    slot_tokens = [None] * M
    # target
    slot_tokens[0] = [sk[spec["target_key_slot"]], eq, sv[spec["target_val_slot"]], nl]
    # loads + sentinels for slots 1..M-1
    load_i = 0
    sent_ord = 0
    for pos in range(1, M):
        if pos in load_set:
            if arm == "DS":
                k = fk[spec["ds_key_slots"][load_i]]
                v = fv[spec["ds_val_slots"][load_i]]
            elif arm == "SS":
                k = sk[spec["ss_key_slots"][load_i]]
                v = sv[spec["ss_val_slots"][load_i]]
            else:
                raise ValueError(arm)
            slot_tokens[pos] = [k, eq, v, nl]
            load_i += 1
        else:
            sk_, sv_ = _sentinel_for(sentinel_scheme, sent_ord, pairs)
            slot_tokens[pos] = [sk_, eq, sv_, nl]
            sent_ord += 1
    toks = [t for slot in slot_tokens for t in slot]
    toks += [sk[spec["target_key_slot"]], eq]          # query
    gold = sv[spec["target_val_slot"]]
    return toks, gold
