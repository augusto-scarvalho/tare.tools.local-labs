#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06T2-T0R — FRESH fixed-batch official-Mamba lifecycle requalification + single-pass capture.

Prospective remediation of the RNN-06T strict-preregistration defects (append-only successor; does
NOT touch historical RNN-06T). Fixes, relative to RNN-06T T0:
  * Property split: FIXED_BATCH_REQUEST_ISOLATION (A) vs BATCH_SHAPE_NUMERICAL_PORTABILITY (B, prereg
    OUT_OF_SCOPE). The batch-shape diagnostic never gates the lifecycle mint.
  * C real branch/fork: fresh per-branch reference reconstructed from the frozen parent + cross
    non-interference, with persisted hashes (NO `... or True` tautology).
  * E reset/REUSE equivalence: reset cache is REUSED for a continuation and compared BIT_EXACT to a
    genuinely fresh cache (zero-check alone is insufficient).
  * F serialization roundtrip + CONTINUATION vs a no-roundtrip continuation (immediate hash equality
    alone is insufficient).
  * H temporal snapshot identity with full per-boundary records (conv/ssm/combined hashes, positions).
  * snapshotBoundaryChecks counts ACTUAL performed boundary comparisons.
All fixed-batch comparisons are BIT_EXACT (SHA-256 of bf16 bytes); readout is argmax-identical over
the constrained scored set. Frozen per T0R_PRE_REGISTRATION.md.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer
from mamba_ssm.utils.generation import InferenceParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_06t_lib as L      # noqa: E402  (proven capture/restore/readout on official fast path)
import rnn_06d_lib as D6     # noqa: E402  (v2 anti-oracle construction)

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06T2")
TOK_ID = "EleutherAI/gpt-neox-20b"
POOL_SEED = 20260817
LC_SEED = 20261050
ALT_SEED = 20261051
REUSE_SEED = 20261052
SP_SEED = 20261060
BATCH_T0R = 8
LSEQ = 320
LC_BOUNDS = [80, 160, 240, 320]
TOL_BATCH = 3e-2               # historical batch-shape tolerance (Property B; out-of-scope)
TOL_READOUT_LOGIT = 1e-2       # descriptive alarm only


def git(*a):
    try:
        return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"<git-error:{e}>"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def conv_hash_row(state, r):
    h = hashlib.sha256()
    for li in sorted(state):
        h.update(state[li][0][r].contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def ssm_hash_row(state, r):
    h = hashlib.sha256()
    for li in sorted(state):
        h.update(state[li][1][r].contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def full_traj_with_cache(model, cache, ids_2d, boundaries):
    """Prefill[0:WARMUP]+step the rest, USING THE GIVEN cache (for reset/reuse equivalence)."""
    B, Lx = ids_2d.shape
    boundaries = set(boundaries)
    inf = InferenceParams(max_seqlen=L.MAXLEN, max_batch_size=B)
    inf.key_value_memory_dict = cache
    inf.seqlen_offset = 0
    snaps = {}
    model(ids_2d[:, :L.WARMUP], inference_params=inf)
    inf.seqlen_offset = L.WARMUP
    if L.WARMUP in boundaries:
        snaps[L.WARMUP] = L.clone_state(cache)
    for t in range(L.WARMUP, Lx):
        model(ids_2d[:, t:t + 1], inference_params=inf)
        inf.seqlen_offset = t + 1
        if (t + 1) in boundaries:
            snaps[t + 1] = L.clone_state(cache)
    return snaps


def eq_readout(model, state, query, offset, vt):
    pred, sub = L.readout(model, state, query, offset, vt)
    return pred, sub


def main():
    t_start = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    runner = os.path.abspath(__file__)
    libpath = os.path.join(os.path.dirname(runner), "rnn_06t_lib.py")
    d6path = os.path.join(os.path.dirname(runner), "rnn_06d_lib.py")
    L.install_counters()
    import mamba_ssm, causal_conv1d, triton  # noqa
    import mamba_ssm.modules.mamba2 as m2

    res = {"packet": "RNN-06T2-T0R", "kind": "fixed_batch_lifecycle_requalification"}

    model = L.load_model()
    w0 = L.weights_identity(model)              # LOADED_WEIGHT_MUTATION_SENTINEL (sum fingerprint)
    n_layer = len(model.backbone.layers)
    cfg = model.config
    chunk_size = getattr(cfg, "chunk_size", getattr(getattr(cfg, "ssm_cfg", None), "get", lambda *_: None)("chunk_size") if hasattr(cfg, "ssm_cfg") else None)
    try:
        chunk_size = model.backbone.layers[0].mixer.chunk_size
    except Exception:
        pass

    # ---- deterministic fresh qualification set (disjoint seeds) ----
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([LC_SEED])))
    vocab = int(getattr(cfg, "vocab_size", 50277))
    seqs = torch.tensor(rng.integers(0, vocab, size=(BATCH_T0R, LSEQ)), device=L.DEVICE, dtype=torch.long)
    qtok = torch.tensor(rng.integers(0, vocab, size=(BATCH_T0R, 2)), device=L.DEVICE, dtype=torch.long)
    qtok2 = torch.tensor(rng.integers(0, vocab, size=(BATCH_T0R, 2)), device=L.DEVICE, dtype=torch.long)
    sufP = torch.tensor(rng.integers(0, vocab, size=(BATCH_T0R, 40)), device=L.DEVICE, dtype=torch.long)
    sufQ = torch.tensor(rng.integers(0, vocab, size=(BATCH_T0R, 40)), device=L.DEVICE, dtype=torch.long)
    vtensor = torch.arange(0, 256, device=L.DEVICE, dtype=torch.long)

    # single-pass MQAR (06D v2), held-out subset
    tok = AutoTokenizer.from_pretrained(TOK_ID)
    pools, pool_meta = L.build_official_pools(tok, POOL_SEED)
    vset = sorted(set(pools["scored_vals"]))
    vtq = torch.tensor(vset, device=L.DEVICE, dtype=torch.long)
    M = 192
    K = 4
    SCHEDULE = D6.schedule_slots(M, K)                 # [38,76,115,153]
    sp_specs = [D6.build_d0_example_spec(SP_SEED, i, i % 4, M, 8, 144) for i in range(BATCH_T0R)]
    sp_toks, sp_gold = [], []
    for sp in sp_specs:
        tk, g = D6.materialize_d0(sp, M, pools)
        sp_toks.append(tk); sp_gold.append(g)

    # qualification-set sha over actual ids + specs
    qs_payload = {"seqs": seqs.cpu().numpy().tolist(), "qtok": qtok.cpu().numpy().tolist(),
                  "qtok2": qtok2.cpu().numpy().tolist(),
                  "sufP": sufP.cpu().numpy().tolist(), "sufQ": sufQ.cpu().numpy().tolist(),
                  "sp_toks": sp_toks, "sp_gold": [int(x) for x in sp_gold],
                  "sp_specs": sp_specs, "schedule": SCHEDULE, "M": M}
    qs_sha = D6.sha256_of_obj(qs_payload)

    # disjointness vs prior sets: prior seeds never in 20261xxx range -> structurally disjoint;
    # assert our seeds are not any historical seed.
    HIST_SEEDS = {20260811, 20260813, 20260814, 20260815, 20260816, 20260817, 20260818,
                  20260901, 20260902, 20260950, 20260960, 20260970, 20260980, 20260981}
    our_seeds = {LC_SEED, ALT_SEED, REUSE_SEED, SP_SEED}
    seeds_disjoint = len(our_seeds & HIST_SEEDS) == 0

    res["executed_source_identity"] = {
        "runner_sha256": sha256_file(runner), "runner_git_blob": git("hash-object", runner),
        "runner_dirty": git("status", "--porcelain", "--", runner),
        "lib_git_blob": git("hash-object", libpath), "d6_git_blob": git("hash-object", d6path),
        "git_head": git("rev-parse", "HEAD"),
        "repo_id": L.REPO_ID, "revision": L.REVISION, "mamba_ssm": mamba_ssm.__version__,
        "causal_conv1d": causal_conv1d.__version__, "triton": triton.__version__,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cxx11abi": torch._C._GLIBCXX_USE_CXX11_ABI, "dtype": str(L.DTYPE), "device": L.DEVICE,
        "cuda_device": torch.cuda.get_device_name(0),
        "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.strip(),
        "protocol_sha256": sha256_file(os.path.join(OUTDIR, "T0R_PRE_REGISTRATION.md")),
        "lifecycleQualificationSetSha256_T0R": qs_sha,
        "qual_set_disjoint_from_history_by_seed": bool(seeds_disjoint),
        "our_seeds": sorted(our_seeds), "n_layer": n_layer, "chunk_size": int(chunk_size) if chunk_size else None,
        "model_weights_mutation_sentinel": w0, "batch_t0r": BATCH_T0R}

    lc = {}

    # ================= A. deterministic same-path replay =================
    L.reset_counters()
    s1 = L.run_trajectory(model, seqs, LC_BOUNDS)
    s2 = L.run_trajectory(model, seqs, LC_BOUNDS)
    A_state = all(L.state_hash(s1[b]) == L.state_hash(s2[b]) for b in LC_BOUNDS)
    pA1, subA1 = eq_readout(model, s1[LSEQ], qtok, LSEQ, vtensor)
    pA2, subA2 = eq_readout(model, s2[LSEQ], qtok, LSEQ, vtensor)
    A_read = bool(torch.equal(pA1, pA2) and torch.equal(subA1, subA2))
    lc["A_same_path_replay"] = {"state_bit_exact": bool(A_state), "readout_identical": A_read,
                                "pass": bool(A_state and A_read),
                                "boundary_hashes": {b: L.state_hash(s1[b])[:16] for b in LC_BOUNDS}}

    # ================= B. destroy/reload/restore/continue vs uninterrupted =================
    full = L.run_trajectory(model, seqs, [160, LSEQ])
    finalU = full[LSEQ]                                     # uninterrupted same-path continuation
    mid = full[160]
    blob = L.serialize_state(mid)
    del mid
    torch.cuda.empty_cache()                               # destroy original runtime cache
    mid2 = L.deserialize_state(blob)
    contB = L.continue_trajectory(model, mid2, seqs, 160, [LSEQ])
    B_state = L.state_hash(contB[LSEQ]) == L.state_hash(finalU)
    pBu, subBu = eq_readout(model, finalU, qtok, LSEQ, vtensor)
    pBr, subBr = eq_readout(model, contB[LSEQ], qtok, LSEQ, vtensor)
    B_read = bool(torch.equal(pBu, pBr) and torch.equal(subBu, subBr))
    lc["B_destroy_reload_continue"] = {"final_state_bit_exact": bool(B_state), "readout_identical": B_read,
                                       "pass": bool(B_state and B_read),
                                       "final_uninterrupted_hash": L.state_hash(finalU)[:16],
                                       "final_reloaded_hash": L.state_hash(contB[LSEQ])[:16],
                                       "compared_against": "uninterrupted_same_path_step_continuation"}

    # ================= C. REAL branch/fork (no tautology) =================
    parent = L.clone_state(full[160])
    ph_before = L.state_hash(parent)
    seqsP = torch.cat([seqs[:, :160], sufP], dim=1)        # (B,200): branch P suffix after slot 160
    seqsQ = torch.cat([seqs[:, :160], sufQ], dim=1)
    LP, LQ = seqsP.shape[1], seqsQ.shape[1]
    # run P
    branchP = L.continue_trajectory(model, parent, seqsP, 160, [LP])[LP]
    ph_after_P = L.state_hash(parent)
    predP, subP = eq_readout(model, branchP, qtok, LP, vtensor)
    # fresh P-reference reconstructed independently from the SAME parent (re-derive parent too)
    parent_ref = L.run_trajectory(model, seqs, [160])[160]
    branchP_ref = L.continue_trajectory(model, parent_ref, seqsP, 160, [LP])[LP]
    predP_ref, subP_ref = eq_readout(model, branchP_ref, qtok, LP, vtensor)
    P_reproducible = bool(L.state_hash(branchP) == L.state_hash(branchP_ref)
                          and torch.equal(predP, predP_ref) and torch.equal(subP, subP_ref))
    # run Q (after P) then a fresh Q-alone; both from the frozen parent
    branchQ_afterP = L.continue_trajectory(model, parent, seqsQ, 160, [LQ])[LQ]
    ph_after_Q = L.state_hash(parent)
    branchQ_alone = L.continue_trajectory(model, L.clone_state(full[160]), seqsQ, 160, [LQ])[LQ]
    Q_reproducible = bool(L.state_hash(branchQ_afterP) == L.state_hash(branchQ_alone))
    # P not altered by Q: recompute P-branch after Q ran; compare to original P-branch
    branchP_afterQ = L.continue_trajectory(model, parent, seqsP, 160, [LP])[LP]
    P_unaffected_by_Q = bool(L.state_hash(branchP_afterQ) == L.state_hash(branchP))
    parent_immutable = bool(ph_before == ph_after_P == ph_after_Q)
    branches_distinct = bool(L.state_hash(branchP) != L.state_hash(branchQ_alone))  # different suffixes
    C_ok = bool(P_reproducible and Q_reproducible and P_unaffected_by_Q and parent_immutable and branches_distinct)
    lc["C_real_branch_fork"] = {
        "parent_immutable_after_P_and_Q": parent_immutable,
        "parent_hash_before": ph_before[:16], "parent_hash_after_P": ph_after_P[:16],
        "parent_hash_after_Q": ph_after_Q[:16],
        "P_reproducible_from_fresh_parent": P_reproducible,
        "Q_independent_of_P_execution": Q_reproducible,
        "P_unaffected_by_Q_execution": P_unaffected_by_Q,
        "branches_distinct": branches_distinct,
        "branchP_state_hash": L.state_hash(branchP)[:24], "branchP_ref_state_hash": L.state_hash(branchP_ref)[:24],
        "branchQ_afterP_hash": L.state_hash(branchQ_afterP)[:24], "branchQ_alone_hash": L.state_hash(branchQ_alone)[:24],
        "branchP_pred_sample": [int(x) for x in predP[:3].tolist()],
        "pass": C_ok}

    # ================= D. fixed-batch neighbor isolation (state + continuation + readout) =================
    s_ref = L.run_trajectory(model, seqs, [LSEQ])[LSEQ]
    row_conv = {r: conv_hash_row(s_ref, r) for r in range(BATCH_T0R)}
    row_ssm = {r: ssm_hash_row(s_ref, r) for r in range(BATCH_T0R)}
    row_full = {r: L.state_hash_row(s_ref, r) for r in range(BATCH_T0R)}
    # D.1 neighbor ORDER permutation
    perm = list(range(BATCH_T0R))[::-1]
    sPerm = L.run_trajectory(model, seqs[perm], [LSEQ])[LSEQ]
    order_state = all(row_full[perm[i]] == L.state_hash_row(sPerm, i) for i in range(BATCH_T0R))
    # D.2 neighbor CONTENT replacement (keep focal row 0, replace all others)
    rng2 = np.random.Generator(np.random.PCG64(np.random.SeedSequence([ALT_SEED])))
    seqs_alt = seqs.clone()
    seqs_alt[1:] = torch.tensor(rng2.integers(0, vocab, size=(BATCH_T0R - 1, LSEQ)), device=L.DEVICE, dtype=torch.long)
    sAlt = L.run_trajectory(model, seqs_alt, [LSEQ])[LSEQ]
    content_state = (row_full[0] == L.state_hash_row(sAlt, 0))
    # continuation-state invariance for focal row 0 under content change
    contRef = L.continue_trajectory(model, s_ref, seqsP, LSEQ, [seqsP.shape[1]])[seqsP.shape[1]]
    # D.3 readout argmax invariance under neighbor permutation
    predRef, subRef = eq_readout(model, s_ref, qtok, LSEQ, vtensor)
    predPerm, subPerm = eq_readout(model, sPerm, qtok[perm], LSEQ, vtensor)
    inv = [perm.index(r) for r in range(BATCH_T0R)]
    readout_inv = bool(torch.equal(predRef, predPerm[inv]))
    D_ok = bool(order_state and content_state and readout_inv)
    lc["D_fixed_batch_neighbor_isolation"] = {
        "neighbor_order_state_bit_exact": bool(order_state),
        "neighbor_content_state_bit_exact": bool(content_state),
        "readout_argmax_invariant": readout_inv,
        "focal_row0_conv_hash": row_conv[0][:16], "focal_row0_ssm_hash": row_ssm[0][:16],
        "pass": D_ok}

    # ================= E. reset / REUSE equivalence =================
    fresh_all_zero = all(bool((L.new_cache(model, 1)[li][0].abs().sum() == 0)) for li in [0])
    rngE = np.random.Generator(np.random.PCG64(np.random.SeedSequence([REUSE_SEED])))
    fresh_seq = torch.tensor(rngE.integers(0, vocab, size=(BATCH_T0R, 200)), device=L.DEVICE, dtype=torch.long)
    # dirty a cache with a real captured state, then reset (zero), then REUSE it
    cacheR = L.new_cache(model, BATCH_T0R)
    L.load_state_into(cacheR, s1[LC_BOUNDS[0]])
    dirty_nonzero = any(bool(cacheR[li][1].abs().sum() > 0) for li in cacheR)
    for li in cacheR:
        cacheR[li][0].zero_(); cacheR[li][1].zero_()
    E_zero = all(bool((cacheR[li][0].abs().sum() == 0) and (cacheR[li][1].abs().sum() == 0)) for li in cacheR)
    stateR = full_traj_with_cache(model, cacheR, fresh_seq, [200])[200]         # reuse the reset cache
    cacheF = L.new_cache(model, BATCH_T0R)                                       # genuinely fresh cache
    stateF = full_traj_with_cache(model, cacheF, fresh_seq, [200])[200]
    E_reuse_state = (L.state_hash(stateR) == L.state_hash(stateF))
    pER, subER = eq_readout(model, stateR, qtok, 200, vtensor)
    pEF, subEF = eq_readout(model, stateF, qtok, 200, vtensor)
    E_reuse_read = bool(torch.equal(pER, pEF) and torch.equal(subER, subEF))
    E_ok = bool(E_zero and dirty_nonzero and E_reuse_state and E_reuse_read)
    lc["E_reset_reuse_equivalence"] = {
        "dirtied_nonzero": bool(dirty_nonzero), "reset_all_zero": bool(E_zero),
        "reuse_state_bit_exact_vs_fresh": bool(E_reuse_state), "reuse_readout_identical": E_reuse_read,
        "reused_state_hash": L.state_hash(stateR)[:24], "fresh_state_hash": L.state_hash(stateF)[:24],
        "pass": E_ok}

    # ================= F. serialization roundtrip + CONTINUATION =================
    bnd = full[160]
    blobF = L.serialize_state(bnd)
    rtF = L.deserialize_state(blobF)
    del blobF
    contRT = L.continue_trajectory(model, rtF, seqs, 160, [LSEQ])[LSEQ]           # continue AFTER roundtrip
    cont0 = L.continue_trajectory(model, L.clone_state(full[160]), seqs, 160, [LSEQ])[LSEQ]  # no-roundtrip
    F_state = (L.state_hash(contRT) == L.state_hash(cont0))
    pFrt, subFrt = eq_readout(model, contRT, qtok, LSEQ, vtensor)
    pF0, subF0 = eq_readout(model, cont0, qtok, LSEQ, vtensor)
    F_read = bool(torch.equal(pFrt, pF0) and torch.equal(subFrt, subF0))
    F_ok = bool(F_state and F_read)
    lc["F_roundtrip_continuation"] = {
        "post_roundtrip_continuation_bit_exact": bool(F_state), "readout_identical": F_read,
        "roundtrip_cont_hash": L.state_hash(contRT)[:24], "no_roundtrip_cont_hash": L.state_hash(cont0)[:24],
        "pass": F_ok}

    # ================= G. fixed-batch slice ownership =================
    # focal row invariant to row permutation AND sibling substitution (state+continuation+readout)
    g_state = bool(order_state and content_state)
    # continuation ownership: focal row 0 continuation invariant when siblings replaced
    cont_ref0 = L.continue_trajectory(model, s_ref, seqsP, LSEQ, [seqsP.shape[1]])[seqsP.shape[1]]
    cont_alt0 = L.continue_trajectory(model, sAlt, seqsP, LSEQ, [seqsP.shape[1]])[seqsP.shape[1]]
    g_cont = (L.state_hash_row(cont_ref0, 0) == L.state_hash_row(cont_alt0, 0))
    G_ok = bool(g_state and g_cont)
    lc["G_fixed_batch_slice_ownership"] = {
        "row_state_invariant_under_permutation": bool(order_state),
        "row_state_invariant_under_sibling_substitution": bool(content_state),
        "focal_row_continuation_invariant_to_siblings": bool(g_cont),
        "note": "fixed-batch slice ownership; NO batch1 equivalence claimed here",
        "pass": G_ok}

    # ================= H. temporal snapshot identity (full records) =================
    ctxH = torch.tensor([sp_toks[i][:4 * M] for i in range(BATCH_T0R)], device=L.DEVICE, dtype=torch.long)
    cap_bounds = [4 * (s + 1) for s in SCHEDULE] + [4 * M]
    run_id = hashlib.sha256(("T0R|" + ",".join(map(str, sp_toks[0][:8]))).encode()).hexdigest()[:16]
    snapsH = L.run_trajectory(model, ctxH, cap_bounds)
    replayH = L.run_trajectory(model, ctxH, cap_bounds)
    boundary_checks = 0
    boundary_failures = 0
    h_records = []
    for oi, b in enumerate(cap_bounds):
        combined = L.state_hash(snapsH[b]); combined_r = L.state_hash(replayH[b])
        match = (combined == combined_r)
        boundary_checks += 1
        if not match:
            boundary_failures += 1
        h_records.append({"runId": run_id, "snapshot_ordinal": oi, "token_position": b,
                          "recurrence_boundary": b, "cache_position": b,
                          "conv_state_hash": conv_hash_row(snapsH[b], 0)[:24],
                          "ssm_state_hash": ssm_hash_row(snapsH[b], 0)[:24],
                          "combined_state_hash": combined[:24], "replay_combined_hash": combined_r[:24],
                          "matches_independent_replay": bool(match)})
    H_ok = bool(boundary_failures == 0 and cap_bounds == sorted(cap_bounds))
    lc["H_temporal_snapshot_identity"] = {"pass": H_ok, "boundary_checks": boundary_checks,
                                          "boundary_failures": boundary_failures, "run_id": run_id,
                                          "records": h_records}

    # ================= I. weight immutability (checkpoint identity vs mutation sentinel) =================
    w1 = L.weights_identity(model)
    lc["I_weight_immutability"] = {"OFFICIAL_CHECKPOINT_IDENTITY": L.REVISION,
                                   "LOADED_WEIGHT_MUTATION_SENTINEL_before": w0[:16],
                                   "LOADED_WEIGHT_MUTATION_SENTINEL_after": w1[:16],
                                   "sentinel_unchanged": bool(w0 == w1),
                                   "note": "sentinel = sum-based fingerprint (cheap mutation check), NOT a cryptographic full-weight hash",
                                   "pass": bool(w0 == w1)}

    # ================= J. backend / fast-path identity =================
    fb = L.fallback_reachable()
    kc = dict(L.KCOUNT)
    J_ok = bool((not any(fb.values())) and kc["selective_state_update"] > 0
                and kc["causal_conv1d_update"] > 0
                and (kc["mamba_chunk_scan_combined"] > 0 or kc["mamba_split_conv1d_scan_combined"] > 0))
    lc["J_backend_fastpath_identity"] = {"fallback_reachable": fb, "kernel_counts": kc,
                                         "official_prefill_fired": bool(kc["mamba_chunk_scan_combined"] > 0),
                                         "official_step_fired": bool(kc["selective_state_update"] > 0 and kc["causal_conv1d_update"] > 0),
                                         "fallback_path_count": int(sum(1 for v in fb.values() if v)),
                                         "pass": J_ok}

    # ---- batch-shape diagnostic (Property B; NOT a lifecycle gate) ----
    sa0 = L.run_trajectory(model, seqs[0:1], [LSEQ])[LSEQ]
    md = 0.0
    for li in s_ref:
        md = max(md, float((s_ref[li][0][0] - sa0[li][0][0]).abs().max()),
                 float((s_ref[li][1][0] - sa0[li][1][0]).abs().max()))
    batch_shape_within_tol = bool(md <= TOL_BATCH)
    res["batch_shape_diagnostic"] = {
        "batch1_vs_batchB_max_abs_diff": round(md, 4), "TOL_BATCH_historical": TOL_BATCH,
        "within_tol": batch_shape_within_tol,
        "scope": "OUT_OF_SCOPE_FOR_FIXED_BATCH_RECOVERY (preregistered)",
        "note": "measured for completeness; the operational recovery contract holds batch shape fixed. "
                "NOT called benign; an out-of-scope numerical divergence."}

    lifecycle_tests = [lc["A_same_path_replay"]["pass"], lc["B_destroy_reload_continue"]["pass"],
                       lc["C_real_branch_fork"]["pass"], lc["D_fixed_batch_neighbor_isolation"]["pass"],
                       lc["E_reset_reuse_equivalence"]["pass"], lc["F_roundtrip_continuation"]["pass"],
                       lc["G_fixed_batch_slice_ownership"]["pass"], lc["H_temporal_snapshot_identity"]["pass"],
                       lc["I_weight_immutability"]["pass"], lc["J_backend_fastpath_identity"]["pass"]]
    lifecycle_pass = all(lifecycle_tests)
    res["lifecycle_tests"] = lc
    res["OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE"] = "QUALIFIED" if lifecycle_pass else "NOT_QUALIFIED"
    res["BATCH_SHAPE_NUMERICAL_PORTABILITY"] = ("QUALIFIED" if batch_shape_within_tol
                                                else "OUT_OF_SCOPE_NOT_QUALIFIED")

    # ================= Single-pass historical capture T0R =================
    ctxSP = torch.tensor([sp_toks[i][:4 * M] for i in range(BATCH_T0R)], device=L.DEVICE, dtype=torch.long)
    query = torch.tensor([sp_toks[i][-2:] for i in range(BATCH_T0R)], device=L.DEVICE, dtype=torch.long)
    L.reset_counters()
    snaps = L.run_trajectory(model, ctxSP, cap_bounds)          # ONE trajectory, in-run captures
    replay_sp = L.run_trajectory(model, ctxSP, cap_bounds)      # independent same-path replay
    sp = {"run_id": run_id, "boundaries": cap_bounds, "M": M, "schedule": SCHEDULE,
          "monotonic": cap_bounds == sorted(cap_bounds), "snapshots": [],
          "snapshotBoundaryChecks": 0, "snapshotBoundaryFailures": 0}
    all_match = True
    for oi, b in enumerate(cap_bounds):
        h = L.state_hash(snaps[b]); hr = L.state_hash(replay_sp[b])
        match = (h == hr)
        sp["snapshotBoundaryChecks"] += 1
        if not match:
            sp["snapshotBoundaryFailures"] += 1
        all_match = all_match and match
        pred, sub = L.readout(model, snaps[b], query, b, vtq)
        correct = [bool(int(pred[i]) == int(sp_gold[i])) for i in range(BATCH_T0R)]
        sp["snapshots"].append({"snapshot_ordinal": oi, "boundary": b, "cache_position": b, "run_id": run_id,
                                "state_hash": h[:24], "replay_hash": hr[:24],
                                "hash_matches_uninterrupted": bool(match),
                                "n_correct_over_batch": int(sum(correct))})
    fin_pred, _ = L.readout(model, snaps[cap_bounds[-1]], query, cap_bounds[-1], vtq)
    single_pass_pass = bool(all_match and sp["monotonic"]
                            and L.KCOUNT["selective_state_update"] > 0
                            and not any(L.fallback_reachable().values())
                            and sp["snapshotBoundaryChecks"] == len(cap_bounds)
                            and sp["snapshotBoundaryFailures"] == 0)
    sp["final_readout_n_correct"] = int(sum(1 for i in range(BATCH_T0R) if int(fin_pred[i]) == int(sp_gold[i])))
    sp["kernel_counts"] = dict(L.KCOUNT)
    sp["fallback_reachable"] = L.fallback_reachable()
    res["single_pass_capture"] = sp
    res["SINGLE_PASS_HISTORICAL_CAPTURE_T0R"] = "QUALIFIED" if single_pass_pass else "NOT_QUALIFIED"

    res["both_qualified"] = bool(lifecycle_pass and single_pass_pass)
    res["item_T1R_status"] = "OPEN" if res["both_qualified"] else "BLOCKED_BY_T0R"
    res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    res["total_runtime_s"] = round(time.time() - t_start, 1)

    with open(os.path.join(OUTDIR, "T0R_RESULTS.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)

    # env provenance
    env = {"packet": "RNN-06T2", "repo_id": L.REPO_ID, "revision": L.REVISION,
           "chunk_size": res["executed_source_identity"]["chunk_size"], "n_layer": n_layer,
           "state_bytes_per_seq": L.STATE_BYTES,
           "mamba_ssm": mamba_ssm.__version__, "causal_conv1d": causal_conv1d.__version__,
           "triton": triton.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
           "cxx11abi": torch._C._GLIBCXX_USE_CXX11_ABI, "dtype": str(L.DTYPE),
           "cuda_device": torch.cuda.get_device_name(0), "python": sys.version.split()[0],
           "driver": res["executed_source_identity"]["driver"],
           "kernel_path": "mamba_ssm chunk_scan+causal_conv1d prefill / selective_state_update+causal_conv1d_update step",
           "fast_path_active": J_ok, "kernel_counts_probe": kc, "fallback_reachable": fb,
           "weight_mutation_sentinel": w0}
    with open(os.path.join(OUTDIR, "ENVIRONMENT_PROVENANCE.json"), "w") as f:
        json.dump(env, f, indent=2, default=str)

    print("=== T0R LIFECYCLE ===")
    for k in lc:
        print(f"  {k}: pass={lc[k]['pass']}")
    print(f"OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE = {res['OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE']}")
    print(f"batch_shape_diag max_abs_diff={res['batch_shape_diagnostic']['batch1_vs_batchB_max_abs_diff']} "
          f"-> BATCH_SHAPE_NUMERICAL_PORTABILITY = {res['BATCH_SHAPE_NUMERICAL_PORTABILITY']}")
    print("=== SINGLE-PASS CAPTURE T0R ===")
    for s in sp["snapshots"]:
        print(f"  ord={s['snapshot_ordinal']} b={s['boundary']} match={s['hash_matches_uninterrupted']} "
              f"n_correct={s['n_correct_over_batch']}/{BATCH_T0R}")
    print(f"boundaryChecks={sp['snapshotBoundaryChecks']} failures={sp['snapshotBoundaryFailures']}")
    print(f"SINGLE_PASS_HISTORICAL_CAPTURE_T0R = {res['SINGLE_PASS_HISTORICAL_CAPTURE_T0R']}")
    print(f"both_qualified={res['both_qualified']} item_T1R={res['item_T1R_status']} "
          f"qual_set_sha={qs_sha[:16]} runtime={res['total_runtime_s']}s vram={res['peak_vram_gb']}GB")


if __name__ == "__main__":
    main()
