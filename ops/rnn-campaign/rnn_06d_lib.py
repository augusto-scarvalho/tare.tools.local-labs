#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06D shared ANTI-ORACLE construction library — historical-state recovery ceiling / utility.

Scientific frame (differs from B3/06C which fixed target at slot 0):
  * The TARGET association is written at a RANDOMIZED slot t in [T_MIN, T_MAX] (a valid subset of
    association slots chosen so that (a) enough post-target load follows to reproduce the B3
    degraded/forgetting regime, and (b) some early snapshots straddle the target write). All OTHER
    slots carry a UNIQUE DS (disjoint filler-space) load binding — general recurrent-state load,
    the B3 primary arm, no same-scored-space competition.
  * Fixed total length 4*M + 2 for every example (M slots of `key = value \n` + query `tk =`).
  * The production-candidate SNAPSHOT SCHEDULE is TARGET-AGNOSTIC: K interior slot boundaries at
    fixed fractions of M (floor(M*(k+1)/(K+1)) for k in 0..K-1). It does NOT depend on t or the
    gold answer. FINAL = state after the whole sequence (slot M-1). Historical snapshots are all
    strictly before FINAL.
  * A snapshot at slot s is DEFINED as the recurrent state produced by an INDEPENDENT prefill of
    the prefix ending at slot s (tokens [0 : 4*(s+1)]). transformers-native Mamba2 has no
    multi-token mid-sequence forward (the decode path is single-token; prefill requires
    cache_position[0]==0), so re-prefill-per-boundary is the only proven path — and it makes the
    snapshot a well-defined, reproducible object (boundary self-check re-prefills and matches
    state hashes, exactly as RNN-06C validated). No BIT_EXACT-vs-full-prefill claim is needed or
    made.

Readout: restore state@slot s, decode the query [target_key, '='], constrained argmax over the
scored value vocabulary; gold = target value; chance = 1/|scored vals|. Identical scoring to B3/06C.

Reuses the disjoint single-token pools + RNG + hashing from rnn_06b2_lib (identical pools). No GPU,
no model here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rnn_06b2_lib import (  # noqa: E402  (identical pools + RNG + hashing)
    build_pools, rng_for, sha256_of_obj, SCORED_POOL, FILLER_POOL, N_SENTINEL,
)

M_DEFAULT = 192
# v2 (calibration AMENDMENT 1, pre-outcome): sentinel padding BEFORE the target, unique DS load
# AFTER. v1 (all slots unique-load) saturated the pre-target state so the target was written into a
# near-full state and even the ORACLE_PROXIMAL snapshot could not retrieve it (calibration: proximal
# acc 0.15-0.40 << TAU_PROX 0.75) -> the recovery ceiling was not meaningfully testable. v2 isolates
# the actual quantity recovery must exploit: target cleanly encoded, then forgotten by SUBSEQUENT
# load. No threshold changed; qualification set is fresh + disjoint; this is a calibration-time
# configuration choice made BEFORE any qualification outcome.
GENERATOR_VERSION = "rnn06d_anti_oracle_random_target_v2_sentinel_pre_load_post"


def schedule_slots(M, K):
    """Target-agnostic interior snapshot schedule: K distinct slot boundaries strictly < M-1.

    slot_k = floor(M*(k+1)/(K+1)) for k in 0..K-1. Deterministic, independent of target/gold.
    Returns a sorted list of K unique slot indices in [1, M-2].
    """
    slots = sorted({max(1, min(M - 2, (M * (k + 1)) // (K + 1))) for k in range(K)})
    assert len(slots) == K, f"schedule collision at K={K}, M={M}: {slots}"
    return slots


def prefix_len_for_slot(s):
    """Token length of the prefix covering slots 0..s (each slot = key,eq,val,nl = 4 tokens)."""
    return 4 * (s + 1)


def build_d0_example_spec(seed, ex_id, stratum, M, t_min, t_max):
    """Deterministic per-example assignment. Target at a random slot in [t_min, t_max]; every other
    slot carries a unique DS filler load binding. Scored-space target -> constrained scoring."""
    r = rng_for(seed, 0x06D0, stratum, ex_id)
    target_slot = int(r.integers(t_min, t_max + 1))
    target_key_slot = int(r.integers(0, SCORED_POOL))
    target_val_slot = int(r.integers(0, SCORED_POOL))
    # M-1 unique DS (filler-space) load bindings for the non-target slots.
    load_key_slots = r.permutation(FILLER_POOL).tolist()[:M - 1]
    load_val_slots = r.permutation(FILLER_POOL).tolist()[:M - 1]
    return {"ex_id": ex_id, "stratum": stratum, "M": M,
            "target_slot": target_slot,
            "target_key_slot": target_key_slot, "target_val_slot": target_val_slot,
            "load_key_slots": [int(x) for x in load_key_slots],
            "load_val_slots": [int(x) for x in load_val_slots]}


def materialize_d0(spec, M, pools):
    """Return (token list, gold token). Fixed length 4*M+2. v2 layout:
      slots [0, t-1]  = REPEAT1 sentinel (low-info padding; target cleanly encoded)
      slot   t        = TARGET (scored key/val)
      slots [t+1, M-1]= UNIQUE DS filler load (subsequent interference that forgets the target)
    Query = target_key '=' appended. Post-target load count = M-1-t varies with the randomized t."""
    sk, sv = pools["scored_keys"], pools["scored_vals"]
    fk, fv = pools["filler_keys"], pools["filler_vals"]
    sent_k, sent_v = pools["sentinel_pairs"][0]           # REPEAT1
    eq, nl = pools["seps"]["eq"], pools["seps"]["nl"]
    t = spec["target_slot"]
    assert 0 <= t < M
    slot_tokens = [None] * M
    for pos in range(t):                                  # pre-target sentinel padding
        slot_tokens[pos] = [sent_k, eq, sent_v, nl]
    slot_tokens[t] = [sk[spec["target_key_slot"]], eq, sv[spec["target_val_slot"]], nl]
    li = 0
    for pos in range(t + 1, M):                           # post-target unique DS load
        slot_tokens[pos] = [fk[spec["load_key_slots"][li]], eq, fv[spec["load_val_slots"][li]], nl]
        li += 1
    toks = [tk for slot in slot_tokens for tk in slot]
    toks += [sk[spec["target_key_slot"]], eq]             # query
    gold = sv[spec["target_val_slot"]]
    return toks, gold


def proximal_snapshot_index(schedule, target_slot):
    """ORACLE_TARGET_PROXIMAL (diagnostic; uses target position): index into `schedule` of the
    FIRST snapshot taken at or after the target write (smallest s_k >= t). If every scheduled slot
    is before the target, returns the last (closest) index. Returns (idx, is_post_target)."""
    for i, s in enumerate(schedule):
        if s >= target_slot:
            return i, True
    return len(schedule) - 1, False


def post_target_mask(schedule, target_slot):
    """Bool list: which scheduled snapshots are at/after the target write (have 'seen' the target)."""
    return [s >= target_slot for s in schedule]


def selfcheck_construction(tok, seed=20260901):
    """No-GPU invariants: fixed length across examples, target present exactly once, all load
    bindings unique, schedule target-agnostic + interior, proximal logic sane."""
    pools, _ = build_pools(tok, seed)
    M = M_DEFAULT
    t_min, t_max = 8, 64
    specs = [build_d0_example_spec(seed, i, i % 3, M, t_min, t_max) for i in range(24)]
    lengths = set()
    sent_k = pools["sentinel_pairs"][0][0]
    for sp in specs:
        toks, gold = materialize_d0(sp, M, pools)
        lengths.add(len(toks))
        # target appears exactly once as a scored key=val block at its slot
        t = sp["target_slot"]
        blk = toks[4 * t:4 * t + 4]
        assert blk[0] == pools["scored_keys"][sp["target_key_slot"]]
        assert blk[2] == pools["scored_vals"][sp["target_val_slot"]]
        assert gold == pools["scored_vals"][sp["target_val_slot"]]
        # pre-target slots are sentinel; post-target are unique DS load
        if t > 0:
            assert toks[0] == sent_k, "pre-target slot 0 must be sentinel"
        assert toks[4 * (t - 1)] == sent_k if t > 0 else True
        if t < M - 1:
            assert toks[4 * (t + 1)] == pools["filler_keys"][sp["load_key_slots"][0]]
        # query = target_key '='
        assert toks[-2] == pools["scored_keys"][sp["target_key_slot"]] and toks[-1] == pools["seps"]["eq"]
        # load-binding pool unique (disjoint from target scored space by pool construction)
        assert len(set(sp["load_key_slots"])) == M - 1 and len(set(sp["load_val_slots"])) == M - 1
    assert lengths == {4 * M + 2}, f"non-fixed length: {lengths}"
    for K in (2, 4, 8):
        sch = schedule_slots(M, K)
        assert all(1 <= s <= M - 2 for s in sch) and sch == sorted(set(sch))
        # target-agnostic: schedule independent of any example's target
        assert sch == schedule_slots(M, K)
    # proximal: for a target between two scheduled slots, picks the first >= t
    sch = schedule_slots(M, 4)
    idx, post = proximal_snapshot_index(sch, sch[1] - 1)
    assert post and sch[idx] >= sch[1] - 1
    return {"ok": True, "fixed_length": 4 * M + 2, "n_specs": len(specs),
            "schedules": {K: schedule_slots(M, K) for K in (2, 4, 8)}, "t_band": [t_min, t_max]}


if __name__ == "__main__":
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained("AntonV/mamba2-1.3b-hf",
                                         revision="703e19a43f397c70315244a3424d79456b54fb34")
    import json
    print(json.dumps(selfcheck_construction(_tok), indent=2))
