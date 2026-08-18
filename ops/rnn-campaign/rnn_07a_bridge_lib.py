#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-07A-BRIDGE library — NoLiMa semi-synthetic controlled bridge construction.

ONLYDirect needles ("Actually, {CHAR} lives next to {1}.") + direct literal question
("Which character lives next to {1}?"), 4-way option-likelihood over character names (gold + 3 seeded
distractors). Real book text (rand_shuffle) is the haystack filler; needle planted at a fixed a-priori
depth. Reuses the parent's frozen readout (rnn_07a_lib.readout_from_state). See BRIDGE_PRE_REGISTRATION.md.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_07a_lib as A      # noqa: E402
import rnn_06t_lib as L      # noqa: E402

NOLIMA = "/home/augus/data/nolima"
NEEDLES = os.path.join(NOLIMA, "needlesets", "needle_set_ONLYDirect.json")
BOOK = os.path.join(NOLIMA, "haystack", "rand_shuffle", "rand_book_1.txt")
POOL_SEED = 20261400
N_CHAR = 4
MAX_EXAMPLES = 112
SHORT_TOKENS = 512
NEEDLE_DEPTH = 0.15
PROGRESS = A.PROGRESS   # 25/50/75/90/FINAL


def _subst(s, char, args):
    s = s.replace("{CHAR}", char)
    for i, a in enumerate(args):
        s = s.replace("{" + str(i + 1) + "}", a)
    return s


def build_pool(tok):
    needles = json.load(open(NEEDLES))
    rng = np.random.default_rng(POOL_SEED)
    pool = []
    for e in needles:
        cset = e["character_set"]
        qtmpl = e["questions"]["direct"]
        for tk, tv in e["tests"].items():
            args = tv["input_args"]
            for _ in range(N_CHAR):
                perm = rng.permutation(len(cset))
                char = cset[perm[0]]
                distractors = [cset[perm[1]], cset[perm[2]], cset[perm[3]]]
                opts = [char] + distractors
                order = rng.permutation(4)
                options = [opts[i] for i in order]
                gold = int(np.where(order == 0)[0][0])
                needle_text = _subst(e["needle"], char, args)
                question = _subst(qtmpl, char, args)
                pool.append({"needle_id": e["id"], "reasoning_type": e["reasoning_type"], "test": tk,
                             "char": char, "options": options, "gold": gold,
                             "needle_text": needle_text, "question": question,
                             "needle_ids": A.enc(tok, " " + needle_text.strip() + " "),
                             "stem_ids": A.build_query_stem(tok, question),
                             "choice_ids": A.build_choice_tokens(tok, options)})
                if len(pool) >= MAX_EXAMPLES:
                    return pool
    return pool


def load_book_tokens(tok):
    return A.enc(tok, open(BOOK, encoding="utf-8", errors="ignore").read())


def make_context(book_ids, needle_ids, total_tokens, depth, filler_seed):
    """Insert needle at `depth` into `total_tokens` of book filler (seeded offset). Returns ctx ids
    (length ~= total_tokens) and the needle token position."""
    rng = np.random.default_rng(filler_seed)
    n_needle = len(needle_ids)
    n_fill = max(1, total_tokens - n_needle)
    max_off = max(0, len(book_ids) - n_fill - 1)
    off = int(rng.integers(0, max_off + 1)) if max_off > 0 else 0
    fill = book_ids[off:off + n_fill]
    cut = int(round(depth * len(fill)))
    ctx = fill[:cut] + needle_ids + fill[cut:]
    return ctx, cut


@torch.no_grad()
def eval_state_from_ctx(model, ctx_ids, ex, max_seqlen):
    ctx = torch.tensor([ctx_ids], device=A.DEVICE, dtype=torch.long)
    state, off, _ = A.prefill_state(model, ctx, max_seqlen)
    stem_t = torch.tensor([ex["stem_ids"]], device=A.DEVICE, dtype=torch.long)
    choices_pad = A.pad_choices([ex["choice_ids"]])
    r = A.readout_from_state(model, state, off, stem_t, choices_pad, max_seqlen)
    return int(r["content_pred"][0]), float(r["confidence"][0])


@torch.no_grad()
def snapshot_eval(model, ctx_ids, ex, budget):
    """Capture states at PROGRESS prefixes of ctx_ids; return per-snapshot (pred, conf)."""
    max_seqlen = budget + A.CHUNK + 512
    stem_t = torch.tensor([ex["stem_ids"]], device=A.DEVICE, dtype=torch.long)
    choices_pad = A.pad_choices([ex["choice_ids"]])
    Ln = len(ctx_ids)
    preds, confs = [], []
    for p in PROGRESS:
        cut = max(1, int(round(p * Ln)))
        ctx = torch.tensor([ctx_ids[:cut]], device=A.DEVICE, dtype=torch.long)
        state, off, _ = A.prefill_state(model, ctx, max_seqlen)
        r = A.readout_from_state(model, state, off, stem_t, choices_pad, max_seqlen)
        preds.append(int(r["content_pred"][0])); confs.append(float(r["confidence"][0]))
    return preds, confs
