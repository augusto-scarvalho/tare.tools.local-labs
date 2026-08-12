#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06T shared library — official Mamba-2 single-pass trajectory, capture, restore, readout.

Canonical path: prefill the first slot (chunk_scan), then autoregressive step decode
(selective_state_update + causal_conv1d_update, the Triton/CUDA fast path) under one InferenceParams,
capturing ACTUAL in-run states at token boundaries. FINAL is the step continuation endpoint. Readout
= branch a captured state into a fresh cache and step the 2-token query. Mamba-2 is position-agnostic
(no positional encoding; step() does not use seqlen_offset in its numerics — the offset only routes
prefill vs step), so a restored state fully determines continuation.
"""
import hashlib
import os
import sys

import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams
import mamba_ssm.modules.mamba2 as m2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rnn_06b2_lib import build_pools, rng_for, sha256_of_obj, SCORED_POOL, FILLER_POOL  # noqa: E402

REPO_ID = "state-spaces/mamba2-1.3b"
REVISION = "c5b59d00ec85d313adea86a08cad2a43c962dd3b"
DEVICE, DTYPE = "cuda", torch.bfloat16
MAXLEN = 1024
WARMUP = 4                 # first slot prefilled; everything after is stepped
STATE_BYTES = 52002816

# ---- kernel counters (prove fast path) ----
KCOUNT = {"mamba_split_conv1d_scan_combined": 0, "mamba_chunk_scan_combined": 0,
          "causal_conv1d_fn": 0, "causal_conv1d_update": 0, "selective_state_update": 0}
_INSTALLED = {"done": False}


def install_counters():
    if _INSTALLED["done"]:
        return
    for nm in list(KCOUNT):
        orig = getattr(m2, nm)
        if orig is None:
            continue
        def mk(o, n):
            def w(*a, **k):
                KCOUNT[n] += 1
                return o(*a, **k)
            return w
        setattr(m2, nm, mk(orig, nm))
    _INSTALLED["done"] = True


def reset_counters():
    for k in KCOUNT:
        KCOUNT[k] = 0


def fallback_reachable():
    return {"causal_conv1d_fn_is_None": m2.causal_conv1d_fn is None,
            "causal_conv1d_update_is_None": m2.causal_conv1d_update is None,
            "selective_state_update_is_None": m2.selective_state_update is None}


def load_model():
    return MambaLMHeadModel.from_pretrained(REPO_ID, device=DEVICE, dtype=DTYPE).eval()


def weights_identity(model):
    return hashlib.sha256(
        "|".join(f"{n}:{float(p.float().sum()):.4e}" for n, p in model.named_parameters()).encode()
    ).hexdigest()


def new_cache(model, B):
    return model.allocate_inference_cache(B, MAXLEN, dtype=DTYPE)


def clone_state(cache):
    """Deep copy of the per-layer (conv,ssm) cache dict (all sequences)."""
    return {li: (cache[li][0].detach().clone(), cache[li][1].detach().clone()) for li in cache}


def slice_state(state, rows):
    return {li: (state[li][0][rows].clone(), state[li][1][rows].clone()) for li in state}


def load_state_into(cache, state):
    for li in cache:
        cache[li][0].copy_(state[li][0])
        cache[li][1].copy_(state[li][1])


def state_hash(state):
    """SHA-256 over the bf16 bytes of concatenated per-layer (conv,ssm), full batch."""
    h = hashlib.sha256()
    for li in sorted(state):
        conv, ssm = state[li]
        h.update(conv.contiguous().cpu().view(torch.uint8).numpy().tobytes())
        h.update(ssm.contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def state_hash_row(state, r):
    h = hashlib.sha256()
    for li in sorted(state):
        conv, ssm = state[li]
        h.update(conv[r].contiguous().cpu().view(torch.uint8).numpy().tobytes())
        h.update(ssm[r].contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def serialize_state(state):
    """State -> CPU byte blob (per layer conv,ssm) for save/reload roundtrip."""
    blob = {}
    for li in state:
        blob[li] = (state[li][0].detach().to("cpu").clone(), state[li][1].detach().to("cpu").clone())
    return blob


def deserialize_state(blob):
    return {li: (blob[li][0].to(DEVICE, DTYPE).clone(), blob[li][1].to(DEVICE, DTYPE).clone())
            for li in blob}


@torch.no_grad()
def run_trajectory(model, ids_2d, boundaries):
    """One canonical trajectory over ids_2d (B, L). Prefill [0:WARMUP], step the rest. Capture a deep
    copy of the full state at each token boundary in `boundaries` (prefix length p == seqlen_offset).
    Returns {p: cloned_state}. All captures share this single run."""
    B, L = ids_2d.shape
    boundaries = set(boundaries)
    cache = new_cache(model, B)
    inf = InferenceParams(max_seqlen=MAXLEN, max_batch_size=B)
    inf.key_value_memory_dict = cache
    snaps = {}
    model(ids_2d[:, :WARMUP], inference_params=inf)
    inf.seqlen_offset = WARMUP
    if WARMUP in boundaries:
        snaps[WARMUP] = clone_state(cache)
    for t in range(WARMUP, L):
        model(ids_2d[:, t:t + 1], inference_params=inf)
        inf.seqlen_offset = t + 1
        if (t + 1) in boundaries:
            snaps[t + 1] = clone_state(cache)
    return snaps


@torch.no_grad()
def continue_trajectory(model, state, ids_2d, start, boundaries):
    """Load `state` into a fresh cache and continue stepping ids_2d[:, start:L], capturing at
    boundaries. Used for save/destroy/reload/continue lifecycle tests."""
    B, L = ids_2d.shape
    boundaries = set(boundaries)
    cache = new_cache(model, B)
    load_state_into(cache, state)
    inf = InferenceParams(max_seqlen=MAXLEN, max_batch_size=B)
    inf.key_value_memory_dict = cache
    inf.seqlen_offset = start
    snaps = {}
    for t in range(start, L):
        model(ids_2d[:, t:t + 1], inference_params=inf)
        inf.seqlen_offset = t + 1
        if (t + 1) in boundaries:
            snaps[t + 1] = clone_state(cache)
    return snaps


@torch.no_grad()
def readout(model, state, query_2d, offset, vtensor):
    """Branch: load `state` into a fresh cache, step the 2-token query at `offset`, return
    (pred_id[B], constrained_logits[B,V]) via constrained argmax over vtensor. Does not mutate
    `state` (loads into a separate cache)."""
    B = state[0][0].shape[0]
    cache = new_cache(model, B)
    load_state_into(cache, state)
    inf = InferenceParams(max_seqlen=MAXLEN, max_batch_size=B)
    inf.key_value_memory_dict = cache
    inf.seqlen_offset = offset
    model(query_2d[:, 0:1], inference_params=inf)
    inf.seqlen_offset = offset + 1
    logits = model(query_2d[:, 1:2], inference_params=inf).logits[:, 0, :].float()
    sub = logits.index_select(1, vtensor)
    pred = vtensor[sub.argmax(-1)]
    return pred, sub


# ---- MQAR construction (06D anti-oracle semantics, official-tokenizer-free: token ids over vocab) ----
# The official mamba2-1.3b uses the GPT-NeoX/EleutherAI tokenizer (vocab 50277). Build disjoint
# single-token pools over that tokenizer.
def build_official_pools(tok, seed):
    return build_pools(tok, seed)
