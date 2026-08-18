#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06A-MAMBA — Frozen-backbone lifecycle qualification runner.

Qualifies whether AntonV/mamba2-1.3b-hf @ 703e19a4 on the pinned transformers
4.48.3 naive torch_forward backend (torch 2.6.0+cu124, bf16, no mamba_ssm/
causal_conv1d) exposes a COMPLETE, request-isolated recurrent state whose
checkpoint / restore / continuation / branch semantics are BIT_EXACT or
bounded-numerically-equivalent, under the preregistered contract in
runs/rnn/RNN-06A-MAMBA/PRE_REGISTRATION.md.

Self-records executed-source identity (repairs P0 NOT_PROVEN) INTO the results
JSON BEFORE outcomes. Emits LIFECYCLE_RESULTS.json + LIFECYCLE_MATRIX.csv and
mints exactly one FROZEN_BACKBONE_LIFECYCLE decision. No training, no push, no
Memory Caching, no historical-state, no RNN-06B mint.
"""
import csv
import gc
import hashlib
import io
import json
import os
import subprocess
import sys
import time

import torch
import transformers
from transformers import Mamba2ForCausalLM
from transformers.models.mamba2 import modeling_mamba2, configuration_mamba2
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06A-MAMBA")
REPO_ID = "AntonV/mamba2-1.3b-hf"
REVISION = "703e19a43f397c70315244a3424d79456b54fb34"
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Predeclared tolerances (PRE_REGISTRATION section 6)
NE_MAXABS, NE_MEANABS = 2e-2, 2e-3
BD_MAXABS = 5e-1

# ----- deterministic challenge inputs (id-space; PRE_REGISTRATION section 8) -----
BASE12 = [11, 42, 7, 128, 256, 3, 99, 64, 15, 201, 88, 333]   # A8 + B4
A8 = BASE12[:8]
SPLITS = [2, 5, 8]
B1 = [15, 201, 88, 333]
B2 = [400, 17, 290, 5]
P = [11, 42, 7, 128, 256, 3, 99, 64]
Q = [22, 19, 300, 7, 45, 120, 8, 260]
Q2 = [1, 2, 3, 4, 5, 6, 7, 8]
PREFIX20 = [11, 42, 7, 128, 256, 3, 99, 64, 15, 201, 88, 333, 5, 6, 7, 8, 9, 10, 44, 77]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"<git-error:{e}>"


def ids(lst, device):
    return torch.tensor([lst], dtype=torch.long, device=device)


def build_cache(model, batch_size):
    return Mamba2Cache(model.config, batch_size, dtype=DTYPE, device=model.device)


@torch.no_grad()
def prefill(model, cache, id_tensor):
    L = id_tensor.shape[1]
    cp = torch.arange(0, L, device=model.device)  # cp[0]==0 => prefill
    out = model(input_ids=id_tensor, cache_params=cache, cache_position=cp, use_cache=True)
    return out.logits  # [B, L, V]


@torch.no_grad()
def decode_step(model, cache, token_col, offset):
    cp = torch.tensor([offset], device=model.device)  # cp[0]>0 => single-token decode
    out = model(input_ids=token_col, cache_params=cache, cache_position=cp, use_cache=True)
    return out.logits  # [B, 1, V]


@torch.no_grad()
def continue_tokens(model, cache, last_logit, cont_ids, start_offset):
    """Return stacked next-token logits aligned to full-sequence positions:
    [last_logit] + one per continuation token. Shape [B, len(cont)+1, V]."""
    cols = [last_logit]
    offset = start_offset
    for t in cont_ids:
        lg = decode_step(model, cache, ids([t], model.device), offset)
        cols.append(lg[:, 0, :])
        offset += 1
    return torch.stack(cols, dim=1)


def snapshot(cache):
    return {
        "conv_states": cache.conv_states.detach().clone().cpu(),
        "ssm_states": cache.ssm_states.detach().clone().cpu(),
    }


def restore(model, snap):
    bsz = snap["conv_states"].shape[1]
    c = build_cache(model, bsz)
    c.conv_states.copy_(snap["conv_states"].to(c.conv_states.device, c.conv_states.dtype))
    c.ssm_states.copy_(snap["ssm_states"].to(c.ssm_states.device, c.ssm_states.dtype))
    return c


def snap_hash(snap):
    h = hashlib.sha256()
    for k in ("conv_states", "ssm_states"):
        t = snap[k].contiguous().cpu()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        # reinterpret raw bytes as uint8 (numpy has no bfloat16 dtype)
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def cmp_logits(a, b):
    """a,b: [..., V] tensors. Returns metrics + predeclared class."""
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
    out = {}
    alleq = True
    for k in ("conv_states", "ssm_states"):
        eq = torch.equal(snapA[k], snapB[k])
        out[k] = {"exact_equal": eq, "shape": list(snapA[k].shape),
                  "dtype": str(snapA[k].dtype)}
        alleq = alleq and eq
    out["all_exact"] = alleq
    return out


@torch.no_grad()
def weight_fingerprint(model):
    h = hashlib.sha256()
    n_params, total_elems = 0, 0
    for name, p in model.named_parameters():
        n_params += 1
        total_elems += p.numel()
        s = float(p.float().sum().item())
        ss = float(p.float().pow(2).sum().item())
        h.update(f"{name}|{tuple(p.shape)}|{p.dtype}|{s:.6e}|{ss:.6e}".encode())
    return {"fingerprint_sha256": h.hexdigest(), "n_param_tensors": n_params,
            "total_elements": total_elems}


def load_model():
    m = Mamba2ForCausalLM.from_pretrained(REPO_ID, revision=REVISION,
                                          torch_dtype=DTYPE).to(DEVICE).eval()
    return m


def main():
    t_start = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(0)
    runner_path = os.path.abspath(__file__)

    results = {"packet": "RNN-06A-MAMBA", "kind": "lifecycle_results"}

    # ---------------- IDENTITY FREEZE (self-recorded before outcomes) ----------
    modeling_src = modeling_mamba2.__file__
    config_src = configuration_mamba2.__file__
    identity = {
        "runner_file": runner_path,
        "runner_source_sha256": sha256_file(runner_path),
        "runner_git_blob": git("hash-object", runner_path),
        "runner_git_tracked_dirty": git("status", "--porcelain", "--", runner_path),
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "modeling_mamba2_file": modeling_src,
        "modeling_mamba2_sha256": sha256_file(modeling_src),
        "configuration_mamba2_file": config_src,
        "configuration_mamba2_sha256": sha256_file(config_src),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "repo_id": REPO_ID,
        "revision": REVISION,
        "dtype": str(DTYPE),
        "mamba_ssm_present": modeling_mamba2.is_mamba_2_ssm_available(),
        "causal_conv1d_present": modeling_mamba2.is_causal_conv1d_available(),
        "is_fast_path_available": modeling_mamba2.is_fast_path_available,
    }
    results["executed_source_identity"] = identity
    print("[identity] modeling sha256:", identity["modeling_mamba2_sha256"])
    print("[identity] is_fast_path_available:", identity["is_fast_path_available"])
    assert identity["is_fast_path_available"] is False, "fast path unexpectedly available"

    # ---------------- LOAD MODEL (M1) ------------------------------------------
    t0 = time.time()
    model = load_model()
    load_time = time.time() - t0
    cfg = model.config
    results["model_load_time_s"] = load_time
    results["config_hash_sha256"] = hashlib.sha256(
        json.dumps(cfg.to_dict(), sort_keys=True, default=str).encode()).hexdigest()

    # weight fingerprint BEFORE
    fp_before = weight_fingerprint(model)
    results["weight_fingerprint_before"] = fp_before

    # measured state bytes / contract re-verification (full-module inventory)
    c0 = build_cache(model, 1)
    inv = {
        "conv_states": {"shape": list(c0.conv_states.shape), "dtype": str(c0.conv_states.dtype),
                        "device": str(c0.conv_states.device),
                        "bytes": c0.conv_states.numel() * c0.conv_states.element_size()},
        "ssm_states": {"shape": list(c0.ssm_states.shape), "dtype": str(c0.ssm_states.dtype),
                       "device": str(c0.ssm_states.device),
                       "bytes": c0.ssm_states.numel() * c0.ssm_states.element_size()},
    }
    inv["total_state_bytes_per_sequence"] = inv["conv_states"]["bytes"] + inv["ssm_states"]["bytes"]
    inv["component_count"] = 2
    inv["expected_total_bytes_bf16"] = 52002816
    inv["byte_accounting_ok"] = (inv["total_state_bytes_per_sequence"] == inv["expected_total_bytes_bf16"])
    results["state_inventory_measured"] = inv
    del c0

    matrix = []  # rows: claim, subcheck, class/bool, metrics

    def row(claim, sub, cls, m):
        matrix.append({"claim": claim, "subcheck": sub, "result": cls, **m})

    # ---------------- CLAIM A: fresh determinism -------------------------------
    ca1 = build_cache(model, 1); lg1 = prefill(model, ca1, ids(BASE12, model.device)); s1 = snapshot(ca1)
    ca2 = build_cache(model, 1); lg2 = prefill(model, ca2, ids(BASE12, model.device)); s2 = snapshot(ca2)
    A_lg = cmp_logits(lg1, lg2)
    A_state = cmp_state_exact(s1, s2)
    results["claim_A_determinism"] = {"logits": A_lg, "state": A_state}
    row("A", "logits_rerun", A_lg["class"], A_lg)
    row("A", "state_rerun_exact", "PASS" if A_state["all_exact"] else "FAIL",
        {"argmax_agreement": 1.0 if A_state["all_exact"] else 0.0})

    # ---------------- CLAIM B: full-sequence vs segmented continuation ---------
    B_res = []
    for a in SPLITS:
        A_ids, Bcont = BASE12[:a], BASE12[a:]
        cf = build_cache(model, 1); lg_full = prefill(model, cf, ids(BASE12, model.device))
        full_slice = lg_full[:, a - 1:a + len(Bcont), :]  # [B, b+1, V]
        cs = build_cache(model, 1); lg_A = prefill(model, cs, ids(A_ids, model.device))
        seg = continue_tokens(model, cs, lg_A[:, -1, :], Bcont, start_offset=a)
        m = cmp_logits(seg, full_slice)
        # bonus: prefix-position identity (prefill(A) last logit vs full at a-1)
        prefix_pos = cmp_logits(lg_A[:, -1, :], lg_full[:, a - 1, :])
        B_res.append({"split_a": a, "cont_len": len(Bcont), "compare": m,
                      "prefix_position_identity": prefix_pos})
        row("B", f"segmented_vs_full_a{a}", m["class"], m)
    results["claim_B_full_vs_segmented"] = B_res

    # ---------------- CLAIM D: branch restore ----------------------------------
    cd = build_cache(model, 1); prefill(model, cd, ids(A8, model.device)); snapD = snapshot(cd)
    snapD_hash_pre = snap_hash(snapD)
    # independent fresh-prefill references for each branch
    def fresh_branch(cont):
        c = build_cache(model, 1); lgA = prefill(model, c, ids(A8, model.device))
        return continue_tokens(model, c, lgA[:, -1, :], cont, start_offset=len(A8))
    indep1 = fresh_branch(B1)
    indep2 = fresh_branch(B2)
    # restored branches from the shared snapshot
    cB1 = restore(model, snapD)
    lgA_dummy = None
    # need a "last_logit" seed for continue: recompute A's last logit once (deterministic)
    cseed = build_cache(model, 1); lgA_seed = prefill(model, cseed, ids(A8, model.device))
    seed_last = lgA_seed[:, -1, :]
    r1 = continue_tokens(model, cB1, seed_last, B1, start_offset=len(A8))
    cB2 = restore(model, snapD)
    r2 = continue_tokens(model, cB2, seed_last, B2, start_offset=len(A8))
    # contamination: run B1 on a restored cache, then restore fresh and run B2
    cX = restore(model, snapD); _ = continue_tokens(model, cX, seed_last, B1, start_offset=len(A8))
    cY = restore(model, snapD); r2_after = continue_tokens(model, cY, seed_last, B2, start_offset=len(A8))
    snapD_hash_post = snap_hash(snapD)
    D_b1 = cmp_logits(r1, indep1)
    D_b2 = cmp_logits(r2, indep2)
    D_contam = cmp_logits(r2_after, r2)
    D = {
        "branch1_vs_independent": D_b1,
        "branch2_vs_independent": D_b2,
        "b2_after_b1_vs_b2_alone": D_contam,
        "parent_snapshot_unchanged": (snapD_hash_pre == snapD_hash_post),
        "snapD_hash_pre": snapD_hash_pre, "snapD_hash_post": snapD_hash_post,
    }
    results["claim_D_branch"] = D
    row("D", "branch1_restore_vs_independent", D_b1["class"], D_b1)
    row("D", "branch2_restore_vs_independent", D_b2["class"], D_b2)
    row("D", "no_cross_branch_contamination", D_contam["class"], D_contam)
    row("D", "parent_snapshot_unchanged", "PASS" if D["parent_snapshot_unchanged"] else "FAIL",
        {"argmax_agreement": 1.0 if D["parent_snapshot_unchanged"] else 0.0})

    # ---------------- CLAIM E: request isolation -------------------------------
    cP = build_cache(model, 1); lgP_alone = prefill(model, cP, ids(P, model.device)); sP_alone = snapshot(cP)
    cQ = build_cache(model, 1); lgQ_alone = prefill(model, cQ, ids(Q, model.device))
    cPQ = build_cache(model, 2)
    lgPQ = prefill(model, cPQ, torch.tensor([P, Q], dtype=torch.long, device=model.device))
    sPQ = snapshot(cPQ)
    cPQ2 = build_cache(model, 2)
    lgPQ2 = prefill(model, cPQ2, torch.tensor([P, Q2], dtype=torch.long, device=model.device))
    sPQ2 = snapshot(cPQ2)
    # neighbor-invariance (load-bearing, expect BIT_EXACT): P row must not depend on neighbor
    E_leak_logits = cmp_logits(lgPQ[0:1], lgPQ2[0:1])
    # per-row state slice invariance
    conv_P_pq = sPQ["conv_states"][:, 0:1]; conv_P_pq2 = sPQ2["conv_states"][:, 0:1]
    ssm_P_pq = sPQ["ssm_states"][:, 0:1]; ssm_P_pq2 = sPQ2["ssm_states"][:, 0:1]
    E_leak_state = (torch.equal(conv_P_pq, conv_P_pq2) and torch.equal(ssm_P_pq, ssm_P_pq2))
    # supporting: alone vs in-batch (may be NUMERICALLY_EQUIVALENT due to batch GEMM)
    E_alone_P = cmp_logits(lgP_alone, lgPQ[0:1])
    E_alone_Q = cmp_logits(lgQ_alone, lgPQ[1:2])
    E = {
        "neighbor_invariance_logits": E_leak_logits,
        "neighbor_invariance_state_exact": E_leak_state,
        "P_alone_vs_in_batch": E_alone_P,
        "Q_alone_vs_in_batch": E_alone_Q,
    }
    results["claim_E_isolation"] = E
    row("E", "neighbor_invariance_logits", E_leak_logits["class"], E_leak_logits)
    row("E", "neighbor_invariance_state_exact", "PASS" if E_leak_state else "FAIL",
        {"argmax_agreement": 1.0 if E_leak_state else 0.0})
    row("E", "P_alone_vs_in_batch", E_alone_P["class"], E_alone_P)
    row("E", "Q_alone_vs_in_batch", E_alone_Q["class"], E_alone_Q)

    # ---------------- CLAIM F: reset / fresh state -----------------------------
    cF = build_cache(model, 1); prefill(model, cF, ids(A8, model.device)); cF.reset()
    snap_reset = snapshot(cF)
    snap_fresh = snapshot(build_cache(model, 1))
    F_reset_eq = cmp_state_exact(snap_reset, snap_fresh)
    zeros_ok = bool((snap_reset["conv_states"].abs().sum().item() == 0.0) and
                    (snap_reset["ssm_states"].abs().sum().item() == 0.0))
    # prefill on reset cache == prefill on fresh cache
    cF2 = build_cache(model, 1); prefill(model, cF2, ids(A8, model.device)); cF2.reset()
    lg_reset = prefill(model, cF2, ids(Q, model.device))
    cFr = build_cache(model, 1); lg_fresh = prefill(model, cFr, ids(Q, model.device))
    F_prefill = cmp_logits(lg_reset, lg_fresh)
    F = {"reset_state_vs_fresh_exact": F_reset_eq, "reset_is_zeros": zeros_ok,
         "prefill_after_reset_vs_fresh": F_prefill}
    results["claim_F_reset"] = F
    row("F", "reset_state_vs_fresh_exact", "PASS" if F_reset_eq["all_exact"] else "FAIL",
        {"argmax_agreement": 1.0 if F_reset_eq["all_exact"] else 0.0})
    row("F", "prefill_after_reset_vs_fresh", F_prefill["class"], F_prefill)

    # ---------------- CLAIM G: state round-trip (via bytes) --------------------
    cG = build_cache(model, 1); prefill(model, cG, ids(BASE12, model.device)); snapG = snapshot(cG)
    buf = io.BytesIO(); torch.save(snapG, buf); nbytes = buf.tell(); buf.seek(0)
    snapG_loaded = torch.load(buf, map_location="cpu")
    G_eq = cmp_state_exact(snapG, snapG_loaded)
    G_meta_ok = all(
        snapG[k].shape == snapG_loaded[k].shape and snapG[k].dtype == snapG_loaded[k].dtype
        for k in ("conv_states", "ssm_states")
    )
    G_bytes = sum(snapG[k].numel() * snapG[k].element_size() for k in ("conv_states", "ssm_states"))
    G = {"roundtrip_exact": G_eq, "meta_preserved": G_meta_ok, "component_count": 2,
         "state_bytes": G_bytes, "buffer_bytes": nbytes,
         "byte_accounting_ok": (G_bytes == inv["expected_total_bytes_bf16"])}
    results["claim_G_roundtrip"] = G
    row("G", "roundtrip_state_exact", "PASS" if G_eq["all_exact"] else "FAIL",
        {"argmax_agreement": 1.0 if G_eq["all_exact"] else 0.0})
    row("G", "byte_accounting", "PASS" if G["byte_accounting_ok"] else "FAIL",
        {"argmax_agreement": 1.0 if G["byte_accounting_ok"] else 0.0})

    # ---------------- CHUNK-SIZE identity sub-experiment (section 9) ------------
    def set_chunk(cs):
        for blk in model.backbone.layers:
            blk.mixer.chunk_size = cs
    native_cs = cfg.chunk_size
    c256 = build_cache(model, 1); lg_cs256 = prefill(model, c256, ids(PREFIX20, model.device))
    set_chunk(8)
    c8 = build_cache(model, 1); lg_cs8 = prefill(model, c8, ids(PREFIX20, model.device))
    set_chunk(native_cs)
    chunk_cmp = cmp_logits(lg_cs8, lg_cs256)
    chunk_is_identity = chunk_cmp["class"] not in ("BIT_EXACT", "NUMERICALLY_EQUIVALENT")
    results["chunk_size_experiment"] = {
        "native_chunk_size": native_cs, "compared_chunk_size": 8,
        "prefix_len": len(PREFIX20), "compare": chunk_cmp,
        "CHUNK_SIZE_IS_PART_OF_EXECUTION_IDENTITY": chunk_is_identity,
    }
    row("chunk", "cs8_vs_cs256_len20", chunk_cmp["class"], chunk_cmp)

    # weight fingerprint AFTER (immutability across all M1 ops)
    fp_after = weight_fingerprint(model)
    results["weight_fingerprint_after"] = fp_after
    results["weights_immutable_M1"] = (fp_after["fingerprint_sha256"] == fp_before["fingerprint_sha256"])
    results["training_mode_off"] = (not model.training)
    results["peak_vram_bytes_before_claimC"] = torch.cuda.max_memory_allocated()

    # ---------------- CLAIM C: serialize -> DESTROY runtime -> restore ---------
    # in-memory reference: prefill(A8) + step(B1), no destroy
    cC = build_cache(model, 1); lgA_C = prefill(model, cC, ids(A8, model.device))
    snapC = snapshot(cC)  # state right after prefill(A8), before stepping
    cont_ref = continue_tokens(model, cC, lgA_C[:, -1, :], B1, start_offset=len(A8))
    seed_last_C = lgA_C[:, -1, :].detach().clone()
    # serialize to disk, record bytes + sha
    snap_path = os.path.join(OUTDIR, "state_snapshot_claimC.pt")
    torch.save(snapC, snap_path)
    t_ser = time.time()
    torch.save(snapC, snap_path)
    ser_latency = time.time() - t_ser
    snap_disk_bytes = os.path.getsize(snap_path)
    snap_disk_sha = sha256_file(snap_path)
    # DESTROY runtime
    del model, cC, cP, cQ, cPQ, cPQ2, cF, cF2, cFr, cG, cd, cB1, cB2, cX, cY, cseed
    gc.collect(); torch.cuda.empty_cache()
    # reconstruct clean context (M2)
    t_re = time.time()
    model2 = load_model()
    reload_time = time.time() - t_re
    fp_M2 = weight_fingerprint(model2)
    reload_weights_match = (fp_M2["fingerprint_sha256"] == fp_before["fingerprint_sha256"])
    # deserialize + restore + continue
    t_ld = time.time()
    snapC_loaded = torch.load(snap_path, map_location="cpu")
    restore_ld = time.time() - t_ld
    t_rest = time.time()
    cC2 = restore(model2, snapC_loaded)
    restore_latency = time.time() - t_rest
    t_cont = time.time()
    cont_cand = continue_tokens(model2, cC2, seed_last_C.to(model2.device), B1, start_offset=len(A8))
    cont_latency = time.time() - t_cont
    C_cmp = cmp_logits(cont_cand, cont_ref)
    C = {
        "continuation_after_destroy_restore_vs_in_memory": C_cmp,
        "snapshot_disk_bytes": snap_disk_bytes,
        "snapshot_disk_sha256": snap_disk_sha,
        "reload_weights_match_original": reload_weights_match,
        "latencies_s": {"serialize": ser_latency, "load": restore_ld,
                        "restore": restore_latency, "continue": cont_latency,
                        "reload_model": reload_time},
    }
    results["claim_C_serialize_destroy_restore"] = C
    row("C", "continue_after_destroy_restore_vs_inmem", C_cmp["class"], C_cmp)
    row("C", "reload_weights_match_original", "PASS" if reload_weights_match else "FAIL",
        {"argmax_agreement": 1.0 if reload_weights_match else 0.0})

    # cleanup snapshot .pt (derived, large-ish; keep sha in results)
    try:
        os.remove(snap_path)
    except OSError:
        pass

    # ---------------- GATE (PRE_REGISTRATION section 7) ------------------------
    def is_exact_state(d):
        return d.get("all_exact", False)

    checks = {
        "A_ok": A_lg["class"] in ("BIT_EXACT", "NUMERICALLY_EQUIVALENT") and A_state["all_exact"],
        "B_ok": all(r["compare"]["argmax_agreement"] == 1.0 and
                    r["compare"]["class"] != "NOT_EQUIVALENT" for r in B_res),
        "C_ok": C_cmp["class"] == "BIT_EXACT" and reload_weights_match,
        "D_ok": (D_b1["class"] == "BIT_EXACT" and D_b2["class"] == "BIT_EXACT" and
                 D_contam["class"] == "BIT_EXACT" and D["parent_snapshot_unchanged"]),
        "E_ok": (E_leak_logits["class"] == "BIT_EXACT" and E_leak_state),
        "F_ok": (F_reset_eq["all_exact"] and zeros_ok and
                 F_prefill["class"] in ("BIT_EXACT", "NUMERICALLY_EQUIVALENT")),
        "G_ok": (G_eq["all_exact"] and G_meta_ok and G["byte_accounting_ok"]),
        "weights_immutable": results["weights_immutable_M1"],
        "full_module_not_partial": inv["byte_accounting_ok"] and inv["component_count"] == 2,
    }
    all_ok = all(checks.values())
    verdict = "QUALIFIED" if all_ok else "NOT_QUALIFIED"
    results["gate_checks"] = checks
    results["FROZEN_BACKBONE_LIFECYCLE"] = verdict
    results["total_runtime_s"] = time.time() - t_start
    results["peak_vram_bytes_total"] = torch.cuda.max_memory_allocated()

    # ---------------- write artifacts ------------------------------------------
    with open(os.path.join(OUTDIR, "LIFECYCLE_RESULTS.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    fields = ["claim", "subcheck", "result", "class", "bit_exact", "max_abs_err",
              "mean_abs_err", "rel_err", "argmax_agreement"]
    with open(os.path.join(OUTDIR, "LIFECYCLE_MATRIX.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["claim", "subcheck", "result", "max_abs_err", "mean_abs_err",
                    "rel_err", "argmax_agreement"])
        for r in matrix:
            w.writerow([r.get("claim"), r.get("subcheck"), r.get("result"),
                        r.get("max_abs_err", ""), r.get("mean_abs_err", ""),
                        r.get("rel_err", ""), r.get("argmax_agreement", "")])

    print("\n==== GATE CHECKS ====")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"\nFROZEN_BACKBONE_LIFECYCLE = {verdict}")
    print(f"runtime = {results['total_runtime_s']:.1f}s  peak_vram = "
          f"{results['peak_vram_bytes_total']/1e9:.2f} GB")


if __name__ == "__main__":
    main()
