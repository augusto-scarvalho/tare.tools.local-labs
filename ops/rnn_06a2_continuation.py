#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06A2-MAMBA-CONTINUATION — continuation/checkpoint lifecycle qualification runner.

Qualifies the OPERATIONAL continuation contract (CONTINUATION_CONTRACT.md /
PRE_REGISTRATION.md) for AntonV/mamba2-1.3b-hf @703e19a4 on the pinned transformers
4.48.3 naive torch_forward backend (torch 2.6.0+cu124, bf16, no mamba_ssm/causal_conv1d,
chunk_size=256). The reference and restored paths use the SAME continuation algorithm
after the checkpoint boundary (fixes historical 06A Claim-B cross-algorithm error); the
generation frontier is carried INSIDE the serialized snapshot (fixes 06A R5 gap).

Self-records executed-source identity BEFORE outcomes. Loads the committed held-out
challenge set and re-verifies lifecycleQualificationSetSha256. Emits CONTINUATION_RESULTS.json
+ CONTINUATION_MATRIX.csv and mints exactly one CONTINUATION_LIFECYCLE decision. No training,
no push, no Memory Caching, no historical-state, no FIXED_BACKBONE_GRADED_REGION mint.
"""
import csv
import gc
import hashlib
import io
import json
import os
import subprocess
import time

import torch
import transformers
from transformers import Mamba2ForCausalLM
from transformers.models.mamba2 import modeling_mamba2, configuration_mamba2
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06A2-MAMBA-CONTINUATION")
CHALLENGES_PATH = os.path.join(OUTDIR, "CONTINUATION_CHALLENGES.json")
REPO_ID = "AntonV/mamba2-1.3b-hf"
REVISION = "703e19a43f397c70315244a3424d79456b54fb34"
DEVICE = "cuda"
DTYPE = torch.bfloat16
PINNED_CHUNK_SIZE = 256

# Predeclared diagnostic tolerances (PRE_REGISTRATION section 4) — non-gating channel only.
NE_MAXABS, NE_MEANABS = 2e-2, 2e-3
BD_MAXABS = 5e-1


# ------------------------------------------------------------------ helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_obj(obj):
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"<git-error:{e}>"


def ids(lst):
    return torch.tensor([lst], dtype=torch.long, device=DEVICE)


def build_cache(model, batch_size):
    return Mamba2Cache(model.config, batch_size, dtype=DTYPE, device=model.device)


@torch.no_grad()
def prefill(model, cache, id_tensor):
    L = id_tensor.shape[1]
    cp = torch.arange(0, L, device=model.device)   # cp[0]==0 => prefill/chunked-ssd
    out = model(input_ids=id_tensor, cache_params=cache, cache_position=cp, use_cache=True)
    return out.logits


@torch.no_grad()
def decode_step(model, cache, token_col, offset):
    cp = torch.tensor([offset], device=model.device)  # cp[0]>0 => single-token decode
    out = model(input_ids=token_col, cache_params=cache, cache_position=cp, use_cache=True)
    return out.logits


@torch.no_grad()
def greedy_continue(model, cache, frontier_token, start_cp, N):
    """Feed frontier_token, then its own argmax outputs, N steps. Returns
    (stacked per-step logits [B,N,V], generated token list [N])."""
    logits_steps, tokens = [], []
    cur, cp = int(frontier_token), start_cp
    for _ in range(N):
        lg = decode_step(model, cache, ids([cur]), cp)        # [B,1,V]
        col = lg[:, 0, :]
        logits_steps.append(col)
        cur = int(col[0].argmax().item())
        tokens.append(cur)
        cp += 1
    return torch.stack(logits_steps, dim=1), tokens


@torch.no_grad()
def forced_continue(model, cache, tokens, start_cp):
    """Teacher-forced: feed the given tokens, cp advancing. Returns logits [B,len,V]."""
    cols = []
    cp = start_cp
    for t in tokens:
        lg = decode_step(model, cache, ids([int(t)]), cp)
        cols.append(lg[:, 0, :])
        cp += 1
    return torch.stack(cols, dim=1)


def snapshot_state(cache):
    return {"conv_states": cache.conv_states.detach().clone().cpu(),
            "ssm_states": cache.ssm_states.detach().clone().cpu()}


def make_snapshot(cache, frontier_token, next_cp):
    """Full continuation snapshot: recurrent cache state + generation frontier."""
    s = snapshot_state(cache)
    s["frontier_token"] = int(frontier_token)
    s["next_cache_position"] = int(next_cp)
    return s


def restore_state(model, snap):
    bsz = snap["conv_states"].shape[1]
    c = build_cache(model, bsz)
    c.conv_states.copy_(snap["conv_states"].to(c.conv_states.device, c.conv_states.dtype))
    c.ssm_states.copy_(snap["ssm_states"].to(c.ssm_states.device, c.ssm_states.dtype))
    return c


def snap_hash(snap):
    h = hashlib.sha256()
    for k in ("conv_states", "ssm_states"):
        t = snap[k].contiguous().cpu()
        h.update(k.encode()); h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def cmp_logits(a, b):
    a = a.detach().cpu(); b = b.detach().cpu()
    bit_exact = torch.equal(a, b)
    af, bf = a.float(), b.float()
    max_abs = (af - bf).abs().max().item()
    mean_abs = (af - bf).abs().mean().item()
    denom = bf.abs().max().item()
    rel = (max_abs / denom) if denom > 0 else 0.0
    agree = (af.argmax(-1) == bf.argmax(-1)).float().mean().item()
    if bit_exact:
        cls = "BIT_EXACT"
    elif agree == 1.0 and max_abs <= NE_MAXABS and mean_abs <= NE_MEANABS:
        cls = "NUMERICALLY_EQUIVALENT"
    elif agree == 1.0 and max_abs <= BD_MAXABS:
        cls = "BOUNDED_DIFFERENCE"
    else:
        cls = "NOT_EQUIVALENT"
    return {"class": cls, "bit_exact": bit_exact, "max_abs_err": max_abs,
            "mean_abs_err": mean_abs, "rel_err": rel, "argmax_agreement": agree}


def cmp_state_exact(snapA, snapB):
    out, alleq = {}, True
    for k in ("conv_states", "ssm_states"):
        eq = torch.equal(snapA[k], snapB[k])
        out[k] = {"exact_equal": eq, "shape": list(snapA[k].shape), "dtype": str(snapA[k].dtype)}
        alleq = alleq and eq
    out["all_exact"] = alleq
    return out


@torch.no_grad()
def weight_fingerprint(model):
    h = hashlib.sha256()
    n_params, total = 0, 0
    for name, p in model.named_parameters():
        n_params += 1; total += p.numel()
        h.update(f"{name}|{tuple(p.shape)}|{p.dtype}|{float(p.float().sum()):.6e}|"
                 f"{float(p.float().pow(2).sum()):.6e}".encode())
    return {"fingerprint_sha256": h.hexdigest(), "n_param_tensors": n_params, "total_elements": total}


def load_model():
    m = Mamba2ForCausalLM.from_pretrained(REPO_ID, revision=REVISION, torch_dtype=DTYPE).to(DEVICE).eval()
    for blk in m.backbone.layers:                    # pin chunk size explicitly
        blk.mixer.chunk_size = PINNED_CHUNK_SIZE
    m.config.chunk_size = PINNED_CHUNK_SIZE
    return m


# ------------------------------------------------------------------ main
def main():
    t_start = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(0)
    runner_path = os.path.abspath(__file__)
    results = {"packet": "RNN-06A2-MAMBA-CONTINUATION", "kind": "continuation_lifecycle_results"}

    # ----- challenge set load + re-verify sha (EOL-independent, canonical) -----
    with open(CHALLENGES_PATH) as f:
        ch = json.load(f)
    recorded_sha = ch.pop("lifecycleQualificationSetSha256")
    recomputed_sha = sha256_of_obj(ch)
    challenge_sha_ok = (recorded_sha == recomputed_sha)
    results["challenge_set"] = {
        "path": CHALLENGES_PATH, "recorded_sha256": recorded_sha,
        "recomputed_sha256": recomputed_sha, "sha_match": challenge_sha_ok,
        "generator_version": ch["generator_version"], "master_seed": ch["master_seed"]}
    assert challenge_sha_ok, "challenge set sha mismatch — refusing to run"

    # ----- executed-source identity (before outcomes) -----
    modeling_src = modeling_mamba2.__file__
    config_src = configuration_mamba2.__file__
    identity = {
        "runner_file": runner_path, "runner_source_sha256": sha256_file(runner_path),
        "runner_git_blob": git("hash-object", runner_path),
        "runner_git_tracked_dirty": git("status", "--porcelain", "--", runner_path),
        "git_head": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "modeling_mamba2_file": modeling_src, "modeling_mamba2_sha256": sha256_file(modeling_src),
        "configuration_mamba2_file": config_src, "configuration_mamba2_sha256": sha256_file(config_src),
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "repo_id": REPO_ID, "revision": REVISION, "dtype": str(DTYPE),
        "pinned_chunk_size": PINNED_CHUNK_SIZE,
        "mamba_ssm_present": modeling_mamba2.is_mamba_2_ssm_available(),
        "causal_conv1d_present": modeling_mamba2.is_causal_conv1d_available(),
        "is_fast_path_available": modeling_mamba2.is_fast_path_available,
        "challenge_set_sha256": recorded_sha,
        "protocol_files": {
            "PRE_REGISTRATION.md": sha256_file(os.path.join(OUTDIR, "PRE_REGISTRATION.md")),
            "CONTINUATION_CONTRACT.md": sha256_file(os.path.join(OUTDIR, "CONTINUATION_CONTRACT.md")),
        },
    }
    results["executed_source_identity"] = identity
    print("[identity] modeling sha256:", identity["modeling_mamba2_sha256"])
    print("[identity] is_fast_path_available:", identity["is_fast_path_available"])
    assert identity["is_fast_path_available"] is False, "fast path unexpectedly available"

    # ----- disjointness witness vs 06A exact sequences -----
    a06_seqs = {tuple(x) for x in [
        [11, 42, 7, 128, 256, 3, 99, 64, 15, 201, 88, 333],
        [11, 42, 7, 128, 256, 3, 99, 64], [22, 19, 300, 7, 45, 120, 8, 260],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [11, 42, 7, 128, 256, 3, 99, 64, 15, 201, 88, 333, 5, 6, 7, 8, 9, 10, 44, 77]]}
    my_seqs = ([tuple(s) for s in ch["determinism_seqs"]]
               + [tuple(s) for s in ch["checkpoint_seqs"]]
               + [tuple(ch["branch"]["prefix"])]
               + [tuple(ch["isolation"][k]) for k in ("P", "Q1", "Q2")])
    results["challenge_set"]["disjoint_from_06A_exact_sequences"] = (
        len(a06_seqs & set(my_seqs)) == 0)

    # ----- load model -----
    t0 = time.time(); model = load_model(); results["model_load_time_s"] = time.time() - t0
    cfg = model.config
    results["config_hash_sha256"] = hashlib.sha256(
        json.dumps(cfg.to_dict(), sort_keys=True, default=str).encode()).hexdigest()
    results["effective_chunk_size"] = [model.backbone.layers[0].mixer.chunk_size, cfg.chunk_size]
    fp_before = weight_fingerprint(model); results["weight_fingerprint_before"] = fp_before

    # ----- full-module state inventory -----
    c0 = build_cache(model, 1)
    inv = {"conv_states": {"shape": list(c0.conv_states.shape),
                           "bytes": c0.conv_states.numel() * c0.conv_states.element_size()},
           "ssm_states": {"shape": list(c0.ssm_states.shape),
                          "bytes": c0.ssm_states.numel() * c0.ssm_states.element_size()}}
    inv["total_state_bytes_per_sequence"] = inv["conv_states"]["bytes"] + inv["ssm_states"]["bytes"]
    inv["component_count"] = 2; inv["expected_total_bytes_bf16"] = 52002816
    inv["byte_accounting_ok"] = inv["total_state_bytes_per_sequence"] == 52002816
    results["state_inventory_measured"] = inv
    del c0

    matrix = []
    def row(claim, sub, cls, m):
        matrix.append({"claim": claim, "subcheck": sub, "result": cls, **m})

    B_BOUND = ch["boundaries"]              # [4,8,12]
    N_CONT = 6                              # greedy continuation length
    PREF = ch["prefix_len"]

    # ---- CLAIM A: fresh determinism ----
    A_rows = []
    for i, s in enumerate(ch["determinism_seqs"]):
        ca = build_cache(model, 1); lga = prefill(model, ca, ids(s)); sa = snapshot_state(ca)
        cb = build_cache(model, 1); lgb = prefill(model, cb, ids(s)); sb = snapshot_state(cb)
        lg = cmp_logits(lga, lgb); st = cmp_state_exact(sa, sb)
        A_rows.append({"seq_index": i, "logits": lg, "state_all_exact": st["all_exact"]})
        row("A", f"logits_seq{i}", lg["class"], lg)
        row("A", f"state_seq{i}", "PASS" if st["all_exact"] else "FAIL",
            {"argmax_agreement": 1.0 if st["all_exact"] else 0.0})
    results["claim_A_determinism"] = A_rows
    A_ok = all(r["logits"]["class"] == "BIT_EXACT" and r["state_all_exact"] for r in A_rows)

    # ---- CLAIM B: checkpoint + frontier restore + continuation (greedy, in-memory) ----
    def checkpoint_greedy(seq_tokens, t):
        """Return (ref_logits, ref_tokens, snap, cand_logits, cand_tokens): reference is
        uninterrupted greedy from state@t; candidate restores from snapshot ALONE."""
        ca = build_cache(model, 1); lg = prefill(model, ca, ids(seq_tokens[:t]))
        f0 = int(lg[0, -1, :].argmax().item())
        snap = make_snapshot(ca, f0, t)                    # frontier lives in snapshot
        ref_lg, ref_tok = greedy_continue(model, ca, f0, t, N_CONT)   # uninterrupted
        cb = restore_state(model, snap)
        cand_lg, cand_tok = greedy_continue(model, cb, snap["frontier_token"], t, N_CONT)
        return ref_lg, ref_tok, snap, cand_lg, cand_tok

    B_rows = []
    for i, s in enumerate(ch["checkpoint_seqs"]):
        ref_lg, ref_tok, _snap, cand_lg, cand_tok = checkpoint_greedy(s, 8)
        cmp = cmp_logits(ref_lg, cand_lg)
        tok_id = (ref_tok == cand_tok)
        B_rows.append({"seq_index": i, "boundary": 8, "logits": cmp, "tokens_identical": tok_id,
                       "ref_tokens": ref_tok, "cand_tokens": cand_tok})
        row("B", f"greedy_restore_seq{i}_t8", cmp["class"], cmp)
    results["claim_B_checkpoint_frontier_restore"] = B_rows
    B_ok = all(r["logits"]["class"] == "BIT_EXACT" and r["tokens_identical"] for r in B_rows)

    # ---- CLAIM D: multiple checkpoint boundaries (greedy) ----
    D_rows = []
    dseq = ch["checkpoint_seqs"][0]
    for t in B_BOUND:
        ref_lg, ref_tok, _snap, cand_lg, cand_tok = checkpoint_greedy(dseq, t)
        cmp = cmp_logits(ref_lg, cand_lg)
        D_rows.append({"boundary": t, "logits": cmp, "tokens_identical": ref_tok == cand_tok})
        row("D", f"greedy_restore_t{t}", cmp["class"], cmp)
    results["claim_D_multiple_boundaries"] = D_rows
    D_ok = all(r["logits"]["class"] == "BIT_EXACT" and r["tokens_identical"] for r in D_rows)

    # ---- CLAIM E: branch replay + F parent immutability ----
    bp = ch["branch"]["prefix"]; bt = ch["branch"]["boundary"]
    s1, s2 = ch["branch"]["stream_1"], ch["branch"]["stream_2"]
    cS = build_cache(model, 1); prefill(model, cS, ids(bp[:bt])); snapS = make_snapshot(cS, bp[bt] if bt < len(bp) else 0, bt)
    hash_pre = snap_hash(snapS)
    # independent references (fresh prefill + forced stream)
    def indep(stream):
        c = build_cache(model, 1); prefill(model, c, ids(bp[:bt]))
        return forced_continue(model, c, stream, bt)
    ref1, ref2 = indep(s1), indep(s2)
    r1 = forced_continue(model, restore_state(model, snapS), s1, bt)
    r2 = forced_continue(model, restore_state(model, snapS), s2, bt)
    # contamination: run stream_1 on a restore, then FRESH restore + stream_2 must equal r2
    cX = restore_state(model, snapS); _ = forced_continue(model, cX, s1, bt)
    r2_after = forced_continue(model, restore_state(model, snapS), s2, bt)
    hash_post = snap_hash(snapS)
    E_b1 = cmp_logits(r1, ref1); E_b2 = cmp_logits(r2, ref2); E_contam = cmp_logits(r2_after, r2)
    parent_immutable = (hash_pre == hash_post)
    results["claim_E_branch"] = {"branch1_vs_independent": E_b1, "branch2_vs_independent": E_b2,
                                 "b2_after_b1_vs_b2_alone": E_contam}
    results["claim_F_parent_immutable"] = {"parent_snapshot_unchanged": parent_immutable,
                                           "hash_pre": hash_pre, "hash_post": hash_post}
    row("E", "branch1_restore_vs_independent", E_b1["class"], E_b1)
    row("E", "branch2_restore_vs_independent", E_b2["class"], E_b2)
    row("E", "no_cross_branch_contamination", E_contam["class"], E_contam)
    row("F", "parent_snapshot_unchanged", "PASS" if parent_immutable else "FAIL",
        {"argmax_agreement": 1.0 if parent_immutable else 0.0})
    E_ok = all(x["class"] == "BIT_EXACT" for x in (E_b1, E_b2, E_contam))
    F_ok = parent_immutable

    # ---- CLAIM G: neighbor request isolation (primary) + diagnostic ----
    P, Q1, Q2 = ch["isolation"]["P"], ch["isolation"]["Q1"], ch["isolation"]["Q2"]
    gt = ch["isolation"]["boundary"]
    def batch_prefill(rows):
        c = build_cache(model, len(rows))
        lg = prefill(model, c, torch.tensor(rows, dtype=torch.long, device=DEVICE))
        return c, lg
    cPQ1, lgPQ1 = batch_prefill([P[:gt], Q1[:gt]])
    cPQ2, lgPQ2 = batch_prefill([P[:gt], Q2[:gt]])
    sPQ1, sPQ2 = snapshot_state(cPQ1), snapshot_state(cPQ2)
    # (a) P-row last-token logits invariant to neighbor
    G_logits = cmp_logits(lgPQ1[0:1, -1, :], lgPQ2[0:1, -1, :])
    # (b) P-row state slices invariant
    G_state = (torch.equal(sPQ1["conv_states"][:, 0:1], sPQ2["conv_states"][:, 0:1]) and
               torch.equal(sPQ1["ssm_states"][:, 0:1], sPQ2["ssm_states"][:, 0:1]))
    # (c) P-row restored greedy continuation invariant to neighbor
    def prow_snap(s_full, lg_full):
        snap = {"conv_states": s_full["conv_states"][:, 0:1].clone(),
                "ssm_states": s_full["ssm_states"][:, 0:1].clone()}
        f0 = int(lg_full[0, -1, :].argmax().item())
        return snap, f0
    snpA, fA = prow_snap(sPQ1, lgPQ1); snpB, fB = prow_snap(sPQ2, lgPQ2)
    contA, _ = greedy_continue(model, restore_state(model, snpA), fA, gt, N_CONT)
    contB, _ = greedy_continue(model, restore_state(model, snpB), fB, gt, N_CONT)
    G_cont = cmp_logits(contA, contB)
    # diagnostic (non-gating): P alone vs P in batch
    cPa, lgPa = batch_prefill([P[:gt]])
    G_diag = cmp_logits(lgPa[0:1, -1, :], lgPQ1[0:1, -1, :])
    results["claim_G_isolation"] = {
        "neighbor_invariance_logits": G_logits, "neighbor_invariance_state_exact": G_state,
        "neighbor_invariance_continuation": G_cont, "diagnostic_P_alone_vs_in_batch": G_diag}
    row("G", "neighbor_invariance_logits", G_logits["class"], G_logits)
    row("G", "neighbor_invariance_state_exact", "PASS" if G_state else "FAIL",
        {"argmax_agreement": 1.0 if G_state else 0.0})
    row("G", "neighbor_invariance_continuation", G_cont["class"], G_cont)
    row("G", "diagnostic_P_alone_vs_in_batch", G_diag["class"], G_diag)
    G_ok = (G_logits["class"] == "BIT_EXACT" and G_state and G_cont["class"] == "BIT_EXACT")

    # ---- CLAIM H: reset / fresh ----
    hseq = ch["determinism_seqs"][0][:8]
    cH = build_cache(model, 1); prefill(model, cH, ids(hseq)); cH.reset()
    snap_reset = snapshot_state(cH); snap_fresh = snapshot_state(build_cache(model, 1))
    H_reset_eq = cmp_state_exact(snap_reset, snap_fresh)
    zeros_ok = bool(snap_reset["conv_states"].abs().sum().item() == 0.0 and
                    snap_reset["ssm_states"].abs().sum().item() == 0.0)
    cH2 = build_cache(model, 1); prefill(model, cH2, ids(hseq)); cH2.reset()
    lg_reset = prefill(model, cH2, ids(Q1))
    cHf = build_cache(model, 1); lg_fresh = prefill(model, cHf, ids(Q1))
    H_prefill = cmp_logits(lg_reset, lg_fresh)
    results["claim_H_reset"] = {"reset_state_vs_fresh_exact": H_reset_eq, "reset_is_zeros": zeros_ok,
                                "prefill_after_reset_vs_fresh": H_prefill}
    row("H", "reset_state_vs_fresh_exact", "PASS" if H_reset_eq["all_exact"] else "FAIL",
        {"argmax_agreement": 1.0 if H_reset_eq["all_exact"] else 0.0})
    row("H", "prefill_after_reset_vs_fresh", H_prefill["class"], H_prefill)
    H_ok = H_reset_eq["all_exact"] and zeros_ok and H_prefill["class"] == "BIT_EXACT"

    # ---- CLAIM I: serialization round-trip ----
    cI = build_cache(model, 1); prefill(model, cI, ids(ch["checkpoint_seqs"][0])); snapI = snapshot_state(cI)
    buf = io.BytesIO(); torch.save(snapI, buf); nbytes = buf.tell(); buf.seek(0)
    snapI_loaded = torch.load(buf, map_location="cpu")
    I_eq = cmp_state_exact(snapI, snapI_loaded)
    I_meta = all(snapI[k].shape == snapI_loaded[k].shape and snapI[k].dtype == snapI_loaded[k].dtype
                 for k in ("conv_states", "ssm_states"))
    I_bytes = sum(snapI[k].numel() * snapI[k].element_size() for k in ("conv_states", "ssm_states"))
    results["claim_I_roundtrip"] = {"roundtrip_exact": I_eq, "meta_preserved": I_meta,
                                    "state_bytes": I_bytes, "buffer_bytes": nbytes,
                                    "byte_accounting_ok": I_bytes == 52002816}
    row("I", "roundtrip_state_exact", "PASS" if I_eq["all_exact"] else "FAIL",
        {"argmax_agreement": 1.0 if I_eq["all_exact"] else 0.0})
    row("I", "byte_accounting", "PASS" if I_bytes == 52002816 else "FAIL",
        {"argmax_agreement": 1.0 if I_bytes == 52002816 else 0.0})
    I_ok = I_eq["all_exact"] and I_meta and I_bytes == 52002816

    # ---- weight immutability (before destroy) ----
    fp_mid = weight_fingerprint(model)
    weights_immutable = fp_mid["fingerprint_sha256"] == fp_before["fingerprint_sha256"]
    results["weights_immutable_pre_destroy"] = weights_immutable
    results["training_mode_off"] = not model.training
    results["peak_vram_bytes_pre_C"] = torch.cuda.max_memory_allocated()

    # ---- CLAIM C: serialize -> DESTROY -> reload -> restore -> continue (greedy) ----
    cseq = ch["checkpoint_seqs"][1]; ct = 8
    cC = build_cache(model, 1); lgC = prefill(model, cC, ids(cseq[:ct]))
    f0C = int(lgC[0, -1, :].argmax().item())
    snapC = make_snapshot(cC, f0C, ct)
    ref_lg_C, ref_tok_C = greedy_continue(model, cC, f0C, ct, N_CONT)
    ref_lg_C_cpu = ref_lg_C.detach().cpu()                        # keep across destroy
    snap_path = os.path.join(OUTDIR, "state_snapshot_claimC.pt")
    torch.save(snapC, snap_path)
    t_ser0 = time.time(); torch.save(snapC, snap_path); ser_latency = time.time() - t_ser0
    snap_disk_bytes = os.path.getsize(snap_path); snap_disk_sha = sha256_file(snap_path)
    del model, cC, cPQ1, cPQ2, cPa, cH, cH2, cHf, cI, cS, cX
    gc.collect(); torch.cuda.empty_cache()
    t_re = time.time(); model2 = load_model(); reload_time = time.time() - t_re
    fp_M2 = weight_fingerprint(model2)
    reload_weights_match = fp_M2["fingerprint_sha256"] == fp_before["fingerprint_sha256"]
    t_ld = time.time(); snapC_loaded = torch.load(snap_path, map_location="cpu"); load_lat = time.time() - t_ld
    cC2 = restore_state(model2, snapC_loaded)
    # frontier obtained from the RESTORED SNAPSHOT ALONE (not any external tensor)
    cand_lg_C, cand_tok_C = greedy_continue(model2, cC2, snapC_loaded["frontier_token"], ct, N_CONT)
    C_cmp = cmp_logits(cand_lg_C, ref_lg_C_cpu)
    C_tok_id = cand_tok_C == ref_tok_C
    results["claim_C_destroy_restore"] = {
        "continuation_after_destroy_restore_vs_in_memory": C_cmp,
        "tokens_identical": C_tok_id, "frontier_from_snapshot_only": True,
        "snapshot_disk_bytes": snap_disk_bytes, "snapshot_disk_sha256": snap_disk_sha,
        "reload_weights_match_original": reload_weights_match,
        "latencies_s": {"serialize": ser_latency, "load": load_lat, "reload_model": reload_time}}
    row("C", "greedy_continue_after_destroy_restore", C_cmp["class"], C_cmp)
    row("C", "reload_weights_match_original", "PASS" if reload_weights_match else "FAIL",
        {"argmax_agreement": 1.0 if reload_weights_match else 0.0})
    C_ok = C_cmp["class"] == "BIT_EXACT" and C_tok_id and reload_weights_match
    try:
        os.remove(snap_path)
    except OSError:
        pass

    # ---- weight fingerprint after (reloaded model) ----
    results["weight_fingerprint_after_reload"] = fp_M2
    results["weights_immutable_across_reload"] = reload_weights_match

    # ---- GATE ----
    checks = {
        "A_ok": A_ok, "B_ok": B_ok, "C_ok": C_ok, "D_ok": D_ok, "E_ok": E_ok,
        "F_ok": F_ok, "G_ok": G_ok, "H_ok": H_ok, "I_ok": I_ok,
        "weights_immutable": weights_immutable and reload_weights_match,
        "full_module_state": inv["byte_accounting_ok"] and inv["component_count"] == 2,
        "challenge_sha_ok": challenge_sha_ok,
    }
    results["test_J_stochastic"] = "NOT_APPLICABLE_BY_CONTRACT"
    all_ok = all(checks.values())
    verdict = "QUALIFIED" if all_ok else "NOT_QUALIFIED"
    results["gate_checks"] = checks
    results["CONTINUATION_LIFECYCLE"] = verdict
    results["total_runtime_s"] = time.time() - t_start
    results["peak_vram_bytes_total"] = torch.cuda.max_memory_allocated()

    with open(os.path.join(OUTDIR, "CONTINUATION_RESULTS.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(OUTDIR, "CONTINUATION_MATRIX.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["claim", "subcheck", "result", "max_abs_err", "mean_abs_err", "rel_err",
                    "argmax_agreement"])
        for r in matrix:
            w.writerow([r.get("claim"), r.get("subcheck"), r.get("result"),
                        r.get("max_abs_err", ""), r.get("mean_abs_err", ""),
                        r.get("rel_err", ""), r.get("argmax_agreement", "")])

    print("\n==== GATE CHECKS ====")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"\nCONTINUATION_LIFECYCLE = {verdict}")
    print(f"runtime = {results['total_runtime_s']:.1f}s  peak_vram = "
          f"{results['peak_vram_bytes_total']/1e9:.2f} GB")


if __name__ == "__main__":
    main()
