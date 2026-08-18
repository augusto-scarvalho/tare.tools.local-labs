#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-07A shared library — realistic long-context operating-point discovery on official Mamba-2.

Workload: LongBench v2 (4-way MC). Deterministic scoring:
  * PRIMARY content readout = teacher-forced length-normalized option-likelihood over the 4 choice texts.
  * SECONDARY format readout = letter-constrained next-token argmax over {A,B,C,D}.
Target-agnostic COMPRESSED_OR_RAG control = question-conditioned BM25 retrieval to RAG_BUDGET tokens.
Snapshots via chunked prefill of native-context prefixes at normalized progress (the natural prefill-
then-decode serving path). No training. MAX_CONFIDENCE frozen. See RNN-07A_PRE_REGISTRATION.md.
"""
import json
import math
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer
from mamba_ssm.utils.generation import InferenceParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_06t_lib as L      # noqa: E402  (load_model, DEVICE, DTYPE, clone_state, load_state_into, counters)

DATA = "/home/augus/data/longbench_v2/data.json"
TOK_ID = "EleutherAI/gpt-neox-20b"
PRIORITY_DOMAINS = ["Single-Document QA", "Long-dialogue History Understanding",
                    "Multi-Document QA", "Long Structured Data Understanding"]
LETTERS = ["A", "B", "C", "D"]
RAG_BUDGET = 2048
CHUNK = 512
PROGRESS = [0.25, 0.50, 0.75, 0.90, 1.00]   # normalized snapshot positions (FINAL == 1.00)
DEVICE, DTYPE = L.DEVICE, L.DTYPE


def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOK_ID)


def load_data():
    return json.load(open(DATA))


def enc(tok, s):
    return tok(s, add_special_tokens=False).input_ids


def letter_token_ids(tok):
    # id of the token emitted for a leading-space letter after "Answer:"
    return [enc(tok, " " + ltr)[-1] for ltr in LETTERS]


# ---------------- BM25 target-agnostic RAG control ----------------
def bm25_rag_control(ctx_ids, q_ids, budget=RAG_BUDGET, chunk=CHUNK, k1=1.5, b=0.75):
    """Question-conditioned BM25 over ~chunk-token windows of the context; concatenate top chunks in
    original order up to `budget` tokens. Uses ONLY the question (target-agnostic)."""
    chunks = [ctx_ids[i:i + chunk] for i in range(0, len(ctx_ids), chunk)]
    if not chunks:
        return list(ctx_ids[:budget])
    N = len(chunks)
    avgdl = sum(len(c) for c in chunks) / N
    qset = set(q_ids)
    # df
    df = {}
    for c in chunks:
        for t in set(c) & qset:
            df[t] = df.get(t, 0) + 1
    scores = []
    for ci, c in enumerate(chunks):
        tf = {}
        for t in c:
            if t in qset:
                tf[t] = tf.get(t, 0) + 1
        dl = len(c)
        s = 0.0
        for t, f in tf.items():
            idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append((s, ci))
    scores.sort(key=lambda x: (-x[0], x[1]))
    picked, total = [], 0
    for s, ci in scores:
        if total >= budget:
            break
        picked.append(ci); total += len(chunks[ci])
    picked.sort()
    out = []
    for ci in picked:
        out.extend(chunks[ci])
    return out[:budget]


# ---------------- state prefill + readouts (prefill-then-decode serving path) ----------------
def new_cache(model, B, max_seqlen):
    return model.allocate_inference_cache(B, max_seqlen, dtype=DTYPE)


@torch.no_grad()
def prefill_state(model, ids_2d, max_seqlen):
    """Chunked fast-path prefill of ids_2d (B,L) into a fresh cache; return (cloned_state, offset=L,
    last_token_logits[B,V]). num_last_tokens=1 avoids materializing all-position logits."""
    B, Ln = ids_2d.shape
    cache = new_cache(model, B, max_seqlen)
    inf = InferenceParams(max_seqlen=max_seqlen, max_batch_size=B)
    inf.key_value_memory_dict = cache
    inf.seqlen_offset = 0
    out = model(ids_2d, inference_params=inf, num_last_tokens=1)
    inf.seqlen_offset = Ln
    return L.clone_state(cache), Ln, out.logits[:, -1, :].float()


@torch.no_grad()
def _step_from(model, state, offset, ids_2d, max_seqlen, collect="last"):
    """Load `state`, step ids_2d (B,T) one token at a time from `offset` (decode fast path).
    collect='last' -> logits before each step stacked? We return (final_state, offset', logits_seq)
    where logits_seq[t] = model distribution AFTER consuming ids[:t] and BEFORE ids[t] (i.e. the
    distribution that predicts ids[t]); plus one trailing distribution predicting the token after ids.
    Actually we return per-position 'next-token' logits: logits_seq has T+1 rows [pre-tok0 .. post-last]."""
    B, T = ids_2d.shape
    cache = new_cache(model, B, max_seqlen)
    L.load_state_into(cache, state)
    inf = InferenceParams(max_seqlen=max_seqlen, max_batch_size=B)
    inf.key_value_memory_dict = cache
    inf.seqlen_offset = offset
    logits_seq = []
    for t in range(T):
        out = model(ids_2d[:, t:t + 1], inference_params=inf, num_last_tokens=1)
        logits_seq.append(out.logits[:, -1, :].float())   # distribution predicting ids[t+1]
        inf.seqlen_offset = offset + t + 1
    return L.clone_state(cache), offset + T, logits_seq


@torch.no_grad()
def readout_from_state(model, state, offset, stem_ids, choices_tok, max_seqlen):
    """PRIMARY content readout. From a context `state` at `offset`, step the non-enumerated query stem
    (question + 'Answer:'), then teacher-force each of the 4 choice texts (length-normalized logprob).
    Returns content_pred (argmax option-likelihood), per-option normalized logprobs, and confidence
    (max softmax over the 4 option scores). No gold used; no letter/format dependence."""
    B = stem_ids.shape[0]
    post_state, post_off, stem_logits = _step_from(model, state, offset, stem_ids, max_seqlen)
    first_choice_logits = stem_logits[-1]   # distribution predicting the token right after the stem
    opt_scores = torch.full((B, 4), -1e30, device=DEVICE)
    for m, (ch_ids, ch_mask) in enumerate(choices_tok):
        Kc = ch_ids.shape[1]
        cache = new_cache(model, B, max_seqlen)
        L.load_state_into(cache, post_state)
        inf = InferenceParams(max_seqlen=max_seqlen, max_batch_size=B)
        inf.key_value_memory_dict = cache
        inf.seqlen_offset = post_off
        cur = first_choice_logits
        lp = torch.zeros(B, device=DEVICE)
        for j in range(Kc):
            logp = torch.log_softmax(cur, dim=-1)
            tok_j = ch_ids[:, j]
            lp = lp + logp.gather(1, tok_j.view(-1, 1)).squeeze(1) * ch_mask[:, j]
            out = model(ch_ids[:, j:j + 1], inference_params=inf, num_last_tokens=1)
            cur = out.logits[:, -1, :].float()
            inf.seqlen_offset = post_off + j + 1
        denom = ch_mask.sum(1).clamp(min=1)
        opt_scores[:, m] = lp / denom          # length-normalized logprob
    conf = torch.softmax(opt_scores, dim=-1)
    content_pred = opt_scores.argmax(-1)
    max_conf = conf.max(-1).values
    return {"content_pred": content_pred.cpu().numpy(),
            "opt_scores": opt_scores.cpu().numpy(),
            "confidence": max_conf.cpu().numpy()}


@torch.no_grad()
def letter_readout(model, state, offset, stem_fmt_ids, letter_ids, max_seqlen):
    """SECONDARY format readout. Step the ENUMERATED query prompt (question + 'A. .. B. .. Answer:'),
    read constrained logits over the four letter tokens, return the letter argmax (0..3)."""
    _, _, logits_seq = _step_from(model, state, offset, stem_fmt_ids, max_seqlen)
    last = logits_seq[-1]                                    # distribution predicting the answer letter
    letter_logits = last.index_select(1, torch.tensor(letter_ids, device=DEVICE))
    return letter_logits.argmax(-1).cpu().numpy()


# ---------------- example materialization ----------------
def build_query_stem(tok, question):
    return enc(tok, "\n\nQuestion: " + question.strip() + "\nAnswer:")


def build_query_stem_format(tok, question, choices):
    """Enumerated MC prompt for the letter/format readout."""
    body = "\n\nQuestion: " + question.strip() + "\n"
    for ltr, c in zip(LETTERS, choices):
        body += f"{ltr}. {c.strip()}\n"
    body += "Answer:"
    return enc(tok, body)


def build_choice_tokens(tok, choices, B_index=None):
    """Return list of 4 (ids, mask) padded across a batch — but here per single example we return the
    raw token lists; batching/padding is done by the caller across the batch."""
    return [enc(tok, " " + c.strip()) for c in choices]


def pad_choices(batch_choice_lists):
    """batch_choice_lists: list over examples, each a list of 4 token-id lists.
    Return list of 4 tuples (ids LongTensor (B,Kc), mask (B,Kc))."""
    B = len(batch_choice_lists)
    out = []
    for m in range(4):
        seqs = [batch_choice_lists[i][m] for i in range(B)]
        K = max(1, max(len(s) for s in seqs))
        ids = torch.zeros(B, K, dtype=torch.long)
        mask = torch.zeros(B, K, dtype=torch.float)
        for i, s in enumerate(seqs):
            if len(s) == 0:
                s = [0]
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, :len(s)] = 1.0
        out.append((ids.to(DEVICE), mask.to(DEVICE)))
    return out


def gold_index(ex):
    return LETTERS.index(ex["answer"].strip())


# ---------------- stats ----------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def paired_bootstrap_delta(correct_a, correct_b, n_boot=2000, seed=20261301):
    """Bootstrap CI of mean(a) - mean(b) over paired per-example correctness arrays."""
    a = np.asarray(correct_a, float); b = np.asarray(correct_b, float)
    n = len(a); rng = np.random.default_rng(seed)
    if n == 0:
        return (0.0, 0.0, 0.0)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = a[idx].mean(1) - b[idx].mean(1)
    return (float(a.mean() - b.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))
