#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06T Section 4 — true end-to-end economics (capture included).

Measures FINAL-only (fused prefill), FINAL-only (step path), and recovery-enabled (single-pass step
trajectory + in-run capture + K+1 readouts + MAX_CONFIDENCE selection). Splits compile/cold/warm.
Mints END_TO_END_RECOVERY_UTILITY against the frozen envelope.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import psutil
from transformers import AutoTokenizer
from mamba_ssm.utils.generation import InferenceParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_06t_lib as L      # noqa: E402
import rnn_06d_lib as D6     # noqa: E402

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06T")
TOK_ID = "EleutherAI/gpt-neox-20b"
POOL_SEED = 20260817
BATCH = 16
M = 192
SCHEDULE = [38, 76, 115, 153]
CTX_LEN = 4 * M
CAP_BOUNDS = [4 * (s + 1) for s in SCHEDULE] + [CTX_LEN]
WARM_ITERS = 6
COST_ENVELOPE_MS = 1000.0


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


def sync():
    torch.cuda.synchronize()


@torch.no_grad()
def final_only_fused(model, ctx, vt):
    """Normal deployment: fused/chunked prefill over the whole context (no inference_params), 1 readout."""
    out = model(ctx).logits[:, -1, :]         # no inference_params -> mem-eff/chunked prefill path
    # readout: append 2-token query via a fresh inference_params decode from a re-prefilled state is not
    # what a normal FINAL-only deployment does; a normal deployment just continues generation. We time
    # the full-context forward as the FINAL-only cost (the dominant term); the 2-token query is common
    # to all arms and measured within the trajectory arms.
    return out


@torch.no_grad()
def final_only_step(model, ctx, q, vt):
    snaps = L.run_trajectory(model, ctx, [CTX_LEN])
    pred, _ = L.readout(model, snaps[CTX_LEN], q, CTX_LEN, vt)
    return pred


@torch.no_grad()
def recovery_enabled(model, ctx, q, vt, timing=None):
    t = {}
    sync(); t0 = time.time()
    snaps = L.run_trajectory(model, ctx, CAP_BOUNDS)     # trajectory + in-run capture (clones)
    sync(); t["trajectory_plus_capture_s"] = time.time() - t0
    subs = []
    sync(); t0 = time.time()
    for s in SCHEDULE:
        pos = 4 * (s + 1)
        _, sub = L.readout(model, snaps[pos], q, pos, vt)
        subs.append(sub.cpu().numpy())
    _, subf = L.readout(model, snaps[CTX_LEN], q, CTX_LEN, vt)
    sync(); t["restore_readout_s"] = time.time() - t0
    t0 = time.time()
    pl = np.stack(subs, axis=1)                          # (B,K,V)
    def softmax(x):
        e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
    sel = softmax(pl).max(-1).argmax(1)                  # MAX_CONFIDENCE selection
    t["selection_s"] = time.time() - t0
    if timing is not None:
        timing.append(t)
    return sel


def med(xs):
    return float(np.median(xs))


def main():
    t_start = time.time(); runner = os.path.abspath(__file__)
    L.install_counters()
    import mamba_ssm  # noqa
    res = {"packet": "RNN-06T-ECON", "kind": "end_to_end_economics"}
    compile_t0 = time.time()
    tok = AutoTokenizer.from_pretrained(TOK_ID)
    pools, _ = L.build_official_pools(tok, POOL_SEED)
    vset = sorted(set(pools["scored_vals"])); vt = torch.tensor(vset, device=L.DEVICE, dtype=torch.long)
    model = L.load_model()
    res["compile_model_load_s"] = round(time.time() - compile_t0, 2)

    # real 3B contexts for a representative batch
    spec = json.load(open(os.path.join(OUTDIR, "T1_3B_QUALIFICATION_SPEC.json")))
    ex = spec["examples"][:BATCH]
    ctx_l, q_l = [], []
    for e in ex:
        toks, _ = D6.materialize_d0(e, M, pools)
        ctx_l.append(toks[:CTX_LEN]); q_l.append(toks[-2:])
    ctx = torch.tensor(ctx_l, device=L.DEVICE, dtype=torch.long)
    q = torch.tensor(q_l, device=L.DEVICE, dtype=torch.long)

    proc = psutil.Process()

    # cold (first call) then warm medians
    def bench(fn, *a):
        sync(); c0 = time.time(); fn(*a); sync(); cold = time.time() - c0
        warm = []
        for _ in range(WARM_ITERS):
            sync(); w0 = time.time(); fn(*a); sync(); warm.append(time.time() - w0)
        return cold, med(warm)

    torch.cuda.reset_peak_memory_stats()
    fo_fused_cold, fo_fused_warm = bench(final_only_fused, model, ctx, vt)
    vram_fused = torch.cuda.max_memory_allocated() / 1e9

    torch.cuda.reset_peak_memory_stats()
    fo_step_cold, fo_step_warm = bench(final_only_step, model, ctx, q, vt)
    vram_step = torch.cuda.max_memory_allocated() / 1e9

    torch.cuda.reset_peak_memory_stats()
    rec_cold, rec_warm = bench(recovery_enabled, model, ctx, q, vt)
    vram_rec = torch.cuda.max_memory_allocated() / 1e9
    # component breakdown (warm) from one instrumented recovery call
    timing = []
    recovery_enabled(model, ctx, q, vt, timing=timing)
    comp = timing[-1]

    peak_cpu_ram_gb = round(proc.memory_info().rss / 1e9, 3)
    snapshot_bytes = L.STATE_BYTES * len(SCHEDULE) * BATCH

    # per-query (batch-amortized) ms
    def pq(sec):
        return 1000.0 * sec / BATCH
    added_vs_fused_ms = pq(rec_warm - fo_fused_warm)
    added_vs_step_ms = pq(rec_warm - fo_step_warm)

    # quality from 3B
    b3b = json.load(open(os.path.join(OUTDIR, "T1_3B_RESULTS.json")))
    q_delta = b3b["arms"]["MAX_CONFIDENCE"]["vs_final"]["delta"]
    n_recovered = b3b["arms"]["MAX_CONFIDENCE"]["n_recovered_vs_final"]
    n_harmed = b3b["arms"]["MAX_CONFIDENCE"]["n_harmed_vs_final"]
    net_recovery = n_recovered - n_harmed
    N_3b = b3b["arms"]["FINAL"] and 192

    res["timings_s"] = {
        "final_only_fused": {"cold": round(fo_fused_cold, 4), "warm_batch": round(fo_fused_warm, 4),
                             "warm_per_query_ms": round(pq(fo_fused_warm), 2)},
        "final_only_step": {"cold": round(fo_step_cold, 4), "warm_batch": round(fo_step_warm, 4),
                            "warm_per_query_ms": round(pq(fo_step_warm), 2)},
        "recovery_enabled": {"cold": round(rec_cold, 4), "warm_batch": round(rec_warm, 4),
                             "warm_per_query_ms": round(pq(rec_warm), 2)},
        "recovery_components_warm_s": {k: round(v, 4) for k, v in comp.items()}}
    res["added_latency"] = {"vs_final_fused_ms_per_query": round(added_vs_fused_ms, 2),
                            "vs_final_step_ms_per_query": round(added_vs_step_ms, 2),
                            "note": "vs_fused = full end-to-end recovery premium (step path + capture "
                            "+ restore/readout); vs_step isolates capture+restore/readout only"}
    res["memory"] = {"peak_vram_gb_final_fused": round(vram_fused, 3),
                     "peak_vram_gb_final_step": round(vram_step, 3),
                     "peak_vram_gb_recovery": round(vram_rec, 3),
                     "peak_cpu_ram_gb": peak_cpu_ram_gb,
                     "snapshot_bytes_batch": snapshot_bytes,
                     "snapshot_bytes_per_seq_per_K": L.STATE_BYTES,
                     "state_bytes_x_K_per_seq": L.STATE_BYTES * len(SCHEDULE)}
    snap_mib = snapshot_bytes / (1024 * 1024)
    res["derived"] = {"quality_delta_maxconf_vs_final_3b": q_delta,
                      "net_recovery_count_3b": net_recovery, "n_recovered_3b": n_recovered,
                      "n_harmed_3b": n_harmed,
                      "net_recovery_per_snapshot_MiB": round(net_recovery / snap_mib, 5),
                      "net_recovery_per_added_ms_vs_fused": round(net_recovery / max(1e-9, added_vs_fused_ms), 5),
                      "quality_delta_per_added_ms_vs_fused": round(q_delta / max(1e-9, added_vs_fused_ms), 6),
                      "snapshot_count_K": len(SCHEDULE)}
    res["intrinsic_06d_style_ms_per_query"] = round(pq(comp["restore_readout_s"]), 2)

    # gate
    if net_recovery <= 0:
        verdict = "NOT_QUALIFIED"
    elif q_delta >= 0.05 and added_vs_fused_ms <= COST_ENVELOPE_MS:
        verdict = "QUALIFIED"
    elif q_delta >= 0.05:
        verdict = "COST_FAIL"
    else:
        verdict = "NOT_QUALIFIED"
    res["cost_envelope_ms_per_query"] = COST_ENVELOPE_MS
    res["END_TO_END_RECOVERY_UTILITY"] = verdict
    res["executed_source_identity"] = {"runner_git_blob": git("hash-object", runner),
                                       "runner_dirty": git("status", "--porcelain", "--", runner),
                                       "git_head": git("rev-parse", "HEAD"), "repo_id": L.REPO_ID,
                                       "revision": L.REVISION, "mamba_ssm": mamba_ssm.__version__,
                                       "protocol_sha256": sha256_file(os.path.join(OUTDIR, "T1_ECON_PRE_REGISTRATION.md"))}
    res["total_runtime_s"] = round(time.time() - t_start, 1)
    json.dump(res, open(os.path.join(OUTDIR, "T1_ECONOMICS.json"), "w"), indent=2, default=str)

    print(f"FINAL_fused warm={res['timings_s']['final_only_fused']['warm_per_query_ms']}ms/q")
    print(f"FINAL_step  warm={res['timings_s']['final_only_step']['warm_per_query_ms']}ms/q")
    print(f"RECOVERY    warm={res['timings_s']['recovery_enabled']['warm_per_query_ms']}ms/q")
    print(f"components(warm s)={res['timings_s']['recovery_components_warm_s']}")
    print(f"added vs fused={added_vs_fused_ms:.1f}ms/q  vs step={added_vs_step_ms:.1f}ms/q "
          f"(envelope {COST_ENVELOPE_MS}ms)")
    print(f"intrinsic(06D-style restore+readout)={res['intrinsic_06d_style_ms_per_query']}ms/q")
    print(f"snapshot_bytes_batch={snapshot_bytes} vram rec={vram_rec:.2f}GB")
    print(f"quality_delta={q_delta} net_recovery={net_recovery}")
    print(f"END_TO_END_RECOVERY_UTILITY = {verdict}")


if __name__ == "__main__":
    main()
