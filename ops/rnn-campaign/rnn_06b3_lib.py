#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06B3 / 06C shared ORDER-STABLE construction library.

Fixes the RNN-06B2 temporal-order churn confound. In B2, load bindings were assigned by
scanning active physical slots ascending, so as U grew the ordinal->position->binding mapping of
already-active bindings shifted. Here the mapping is PERMANENT:

    load ordinal i  ->  fixed physical slot = spec.load_positions[i]
                    ->  fixed DS binding i = (filler_keys[ds_key_slots[i]], filler_vals[ds_val_slots[i]])
                    ->  fixed SS binding i = (scored_keys[ss_key_slots[i]], scored_vals[ss_val_slots[i]])

At dose U, ONLY ordinals 0..U-2 are active. Increasing U activates exactly one new ordinal at
its fixed slot with its fixed binding; every already-active binding keeps identity AND position.
Non-active slots (and never slot 0 = target) are REPEAT1 sentinel. Full packing (U=M) is
forbidden; a positive sentinel reserve (M-U) is always retained.

Reuses the disjoint single-token pools + per-example spec from rnn_06b2_lib (identical pools).
No GPU, no model here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rnn_06b2_lib import (  # noqa: E402  (re-export: identical pools + spec + helpers)
    build_pools, build_example_spec, rng_for, sha256_of_obj,
    SCORED_POOL, FILLER_POOL, N_SENTINEL,
)


def active_bindings(spec, U, arm, pools):
    """Return {slot: (key_id, value_id)} for the U-1 active load ordinals (order-stable)."""
    sk, sv = pools["scored_keys"], pools["scored_vals"]
    fk, fv = pools["filler_keys"], pools["filler_vals"]
    out = {}
    for i in range(U - 1):
        slot = spec["load_positions"][i]
        if arm == "DS":
            out[slot] = (fk[spec["ds_key_slots"][i]], fv[spec["ds_val_slots"][i]])
        elif arm == "SS":
            out[slot] = (sk[spec["ss_key_slots"][i]], sv[spec["ss_val_slots"][i]])
        else:
            raise ValueError(arm)
    return out


def materialize_b3(spec, M, U, arm, pools, reserve):
    """Order-stable fixed-length prompt. Requires U <= M - reserve (subpacked, no U=M).
    Returns (token list, gold token). Length = 4*M+2 for all U; target at slot 0."""
    assert 1 <= U, "U must be >=1"
    assert U <= M - reserve, f"subpacking violated: U={U} > M-reserve={M - reserve}"
    sk, sv = pools["scored_keys"], pools["scored_vals"]
    eq, nl = pools["seps"]["eq"], pools["seps"]["nl"]
    sent_k, sent_v = pools["sentinel_pairs"][0]                 # REPEAT1
    active = active_bindings(spec, U, arm, pools)
    slot_tokens = [None] * M
    slot_tokens[0] = [sk[spec["target_key_slot"]], eq, sv[spec["target_val_slot"]], nl]
    for pos in range(1, M):
        k, v = active.get(pos, (sent_k, sent_v))
        slot_tokens[pos] = [k, eq, v, nl]
    toks = [t for slot in slot_tokens for t in slot]
    toks += [sk[spec["target_key_slot"]], eq]                   # query
    return toks, sv[spec["target_val_slot"]]


def nested_identity_check(spec, M, doses, arm, pools):
    """For every adjacent dose pair assert prev_active ⊂ new_active AND every previously-active
    slot keeps identical (key,value). Returns (ok: bool, detail: dict)."""
    ds = sorted(doses)
    prev = None
    for U in ds:
        cur = active_bindings(spec, U, arm, pools)
        if prev is not None:
            if not set(prev).issubset(set(cur)):
                return False, {"fail": "position_subset", "at_dose": U,
                               "prev_slots": sorted(prev), "cur_slots": sorted(cur)}
            for slot, kv in prev.items():
                if cur.get(slot) != kv:
                    return False, {"fail": "identity_changed", "at_dose": U, "slot": slot,
                                   "before": kv, "after": cur.get(slot)}
        prev = cur
    return True, {"checked_dose_pairs": len(ds) - 1, "arm": arm}
