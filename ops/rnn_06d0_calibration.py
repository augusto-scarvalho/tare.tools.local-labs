#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06D0 calibration (EXPLORATORY) — choose the bounded snapshot-schedule configuration.

Fresh calibration set (distinct seed). Sweeps K in {2,4,8} on the anti-oracle construction and
reports, per K: FINAL accuracy (must be a degraded/forgetting regime), ORACLE_BEST_GOLD accuracy,
ORACLE_TARGET_PROXIMAL accuracy, ORACLE_BEST−FINAL, recovery ceiling among FINAL-wrong, n_recoverable,
and a confidence-separation diagnostic (post-target vs pre-target snapshot top1 prob). Chooses the
SMALLEST K meeting the frozen adequacy signals (parsimony); freezes K + target band. Tunes NO
threshold. No seed screening. Writes D0_CALIBRATION.json + D0_CALIBRATION_DECISION.md.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import Mamba2ForCausalLM, AutoTokenizer
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_06d_lib as lib  # noqa: E402

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06D")
REPO_ID = "AntonV/mamba2-1.3b-hf"
REVISION = "703e19a43f397c70315244a3424d79456b54fb34"
DEVICE, DTYPE, CHUNK = "cuda", torch.bfloat16, 32
POOL_SEED = 20260817           # identical token pools as B3/06C
CALIB_SEED = 20260901          # distinct from all prior sets
M = 192
T_MIN, T_MAX = 8, 64
K_CANDIDATES = [2, 4, 8]
S_STRATA = 3
N_PER_STRATUM = 16
BATCH = 2

# Frozen thresholds (from TRAIN_PROTOCOL; NOT tuned here — used only as adequacy signals)
CEILING_SESOI = 0.15
FINAL_ACC_MAX = 0.75
TAU_PROX = 0.75
RECOV_FRAC_MIN = 0.30


def state_sha(t):
    return hashlib.sha256(t.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def build_cache(model, bsz):
    return Mamba2Cache(model.config, bsz, dtype=DTYPE, device=model.device)


@torch.no_grad()
def prefill(model, cache, ids_2d):
    cp = torch.arange(0, ids_2d.shape[1], device=model.device)
    model(input_ids=ids_2d, cache_params=cache, cache_position=cp, use_cache=True)


@torch.no_grad()
def decode_token(model, cache, col_2d, offset):
    cp = torch.tensor([offset], device=model.device)
    return model(input_ids=col_2d, cache_params=cache, cache_position=cp, use_cache=True).logits


def snapshot(cache):
    return cache.conv_states.detach().clone(), cache.ssm_states.detach().clone()


def restore(model, conv, ssm, n):
    c = build_cache(model, n)
    c.conv_states.copy_(conv[:, :n]); c.ssm_states.copy_(ssm[:, :n])
    return c


@torch.no_grad()
def readout_at_slot(model, prefixes, slot, queries, golds, vtensor):
    """Prefill prefix[:4(slot+1)] for all examples (batched), restore, decode query, constrained
    argmax. Returns (correct[bool N], top1prob[float N])."""
    N = len(prefixes); plen = lib.prefix_len_for_slot(slot)
    correct = np.zeros(N, bool); top1 = np.zeros(N, float)
    for b in range(0, N, BATCH):
        rows = list(range(b, min(b + BATCH, N)))
        ids = torch.tensor([prefixes[r][:plen] for r in rows], device=DEVICE, dtype=torch.long)
        cache = build_cache(model, len(rows)); prefill(model, cache, ids)
        conv, ssm = snapshot(cache)
        rcache = restore(model, conv, ssm, len(rows))
        q0 = torch.tensor([[queries[r][0]] for r in rows], device=DEVICE, dtype=torch.long)
        decode_token(model, rcache, q0, plen)
        q1 = torch.tensor([[queries[r][1]] for r in rows], device=DEVICE, dtype=torch.long)
        logits = decode_token(model, rcache, q1, plen + 1)[:, 0, :].float()
        sub = logits.index_select(1, vtensor)
        probs = torch.softmax(sub, dim=-1)
        pred = vtensor[sub.argmax(-1)]
        for li, r in enumerate(rows):
            correct[r] = int(pred[li]) == int(golds[r])
            top1[r] = float(probs[li].max())
        del cache, rcache, conv, ssm
        torch.cuda.empty_cache()
    return correct, top1


def main():
    t0 = time.time(); os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(REPO_ID, revision=REVISION)
    model = Mamba2ForCausalLM.from_pretrained(REPO_ID, revision=REVISION, torch_dtype=DTYPE).to(DEVICE).eval()
    for blk in model.backbone.layers:
        blk.mixer.chunk_size = CHUNK
    model.config.chunk_size = CHUNK
    from transformers.models.mamba2 import modeling_mamba2
    assert modeling_mamba2.is_fast_path_available is False
    pools, pool_meta = lib.build_pools(tok, POOL_SEED)
    vset = sorted(set(pools["scored_vals"]))
    vtensor = torch.tensor(vset, device=DEVICE, dtype=torch.long)
    chance = 1.0 / len(vset)

    # fresh calibration examples
    examples, eid = [], 0
    for s in range(S_STRATA):
        for _ in range(N_PER_STRATUM):
            examples.append(lib.build_d0_example_spec(CALIB_SEED, eid, s, M, T_MIN, T_MAX)); eid += 1
    N = len(examples)
    prefixes, queries, golds, tslots = [], [], [], []
    for e in examples:
        toks, gold = lib.materialize_d0(e, M, pools)
        prefixes.append(toks); queries.append(toks[-2:]); golds.append(gold); tslots.append(e["target_slot"])
    golds = np.array(golds); tslots = np.array(tslots)

    # FINAL once (slot M-1) — shared across all K
    final_correct, final_top1 = readout_at_slot(model, prefixes, M - 1, queries, golds, vtensor)
    final_acc = float(final_correct.mean())

    per_K = {}
    max_K = max(K_CANDIDATES)
    all_slots = sorted(set().union(*[set(lib.schedule_slots(M, K)) for K in K_CANDIDATES]))
    slot_correct, slot_top1 = {}, {}
    for s in all_slots:
        c, p = readout_at_slot(model, prefixes, s, queries, golds, vtensor)
        slot_correct[s] = c; slot_top1[s] = p

    for K in K_CANDIDATES:
        sch = lib.schedule_slots(M, K)
        pool_c = np.stack([slot_correct[s] for s in sch], axis=1)   # (N, K)
        pool_p = np.stack([slot_top1[s] for s in sch], axis=1)
        oracle_best = pool_c.any(axis=1)
        prox = np.array([lib.proximal_snapshot_index(sch, int(t)) for t in tslots], dtype=object)
        prox_idx = np.array([p[0] for p in prox]); prox_post = np.array([p[1] for p in prox])
        proximal_correct = pool_c[np.arange(N), prox_idx]
        ob_acc = float(oracle_best.mean()); px_acc = float(proximal_correct.mean())
        fw = ~final_correct
        n_fw = int(fw.sum())
        n_recov = int((fw & oracle_best).sum())
        recov_frac = (n_recov / n_fw) if n_fw else 0.0
        # confidence separation: post-target vs pre-target snapshot top1 prob
        post_mask = np.stack([np.array(lib.post_target_mask(sch, int(t))) for t in tslots])  # (N,K)
        post_conf = float(pool_p[post_mask].mean()) if post_mask.any() else float("nan")
        pre_conf = float(pool_p[~post_mask].mean()) if (~post_mask).any() else float("nan")
        # per-stratum oracle_best - final
        strat = np.array([e["stratum"] for e in examples])
        per_strat = {int(s): float(oracle_best[strat == s].mean() - final_correct[strat == s].mean())
                     for s in range(S_STRATA)}
        robust = sum(1 for s in per_strat if per_strat[s] >= CEILING_SESOI)
        per_K[str(K)] = {
            "schedule_slots": sch, "final_acc": final_acc, "oracle_best_acc": ob_acc,
            "oracle_proximal_acc": px_acc, "oracle_best_minus_final": round(ob_acc - final_acc, 4),
            "n_final_wrong": n_fw, "n_recoverable": n_recov, "recovery_ceiling_frac": round(recov_frac, 4),
            "post_target_conf": round(post_conf, 4), "pre_target_conf": round(pre_conf, 4),
            "per_stratum_ob_minus_final": per_strat, "robust_strata": robust,
            "prox_post_frac": round(float(prox_post.mean()), 4),
            "adequate": bool(ob_acc - final_acc >= CEILING_SESOI and final_acc <= FINAL_ACC_MAX
                             and px_acc >= TAU_PROX and recov_frac >= RECOV_FRAC_MIN and robust >= 2)}

    chosen = next((K for K in K_CANDIDATES if per_K[str(K)]["adequate"]), None)
    calib = {"packet": "RNN-06D0-CALIBRATION", "kind": "exploratory_calibration",
             "calib_seed": CALIB_SEED, "pool_seed": POOL_SEED, "M": M, "t_band": [T_MIN, T_MAX],
             "n_total": N, "s_strata": S_STRATA, "n_per_stratum": N_PER_STRATUM, "chance": chance,
             "batch": BATCH, "chunk_size": CHUNK, "final_acc": final_acc,
             "K_candidates": K_CANDIDATES, "per_K": per_K,
             "frozen_thresholds": {"CEILING_SESOI": CEILING_SESOI, "FINAL_ACC_MAX": FINAL_ACC_MAX,
                                   "TAU_PROX": TAU_PROX, "RECOV_FRAC_MIN": RECOV_FRAC_MIN},
             "chosen_K": chosen, "pool_meta": pool_meta,
             "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
             "runtime_s": round(time.time() - t0, 1)}
    calib["calibrationSetSha256"] = lib.sha256_of_obj(
        {"seed": CALIB_SEED, "M": M, "t_band": [T_MIN, T_MAX], "examples": examples})
    with open(os.path.join(OUTDIR, "D0_CALIBRATION.json"), "w") as f:
        json.dump(calib, f, indent=2, default=str)

    print(f"\nFINAL acc = {final_acc:.3f}  (degraded band <= {FINAL_ACC_MAX})", file=sys.stderr)
    for K in K_CANDIDATES:
        d = per_K[str(K)]
        print(f"K={K}: OB={d['oracle_best_acc']:.3f} PROX={d['oracle_proximal_acc']:.3f} "
              f"OB-FINAL={d['oracle_best_minus_final']:.3f} recov={d['n_recoverable']}/{d['n_final_wrong']}"
              f"({d['recovery_ceiling_frac']:.2f}) conf(post/pre)={d['post_target_conf']:.2f}/"
              f"{d['pre_target_conf']:.2f} robust={d['robust_strata']}/3 adequate={d['adequate']}",
              file=sys.stderr)
    print(f"CHOSEN K = {chosen}   runtime={calib['runtime_s']}s peak_vram={calib['peak_vram_gb']}GB",
          file=sys.stderr)


if __name__ == "__main__":
    main()
