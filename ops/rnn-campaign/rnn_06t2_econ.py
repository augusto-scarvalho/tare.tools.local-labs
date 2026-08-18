#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06T2-T1R — apples-to-apples end-to-end economics (Section 13).

Every timed arm executes the SAME semantic task: same context + same target query + same constrained
scored-answer readout, returning the same scored answer. Fixes the RNN-06T defect where FINAL_fused
did NOT execute the query.
  * FINAL_FUSED_EQUIVALENT_WORK  = fused/chunked prefill of context, then step the 2-token query -> scored answer
  * FINAL_STEP_EQUIVALENT_WORK   = single-pass step trajectory to FINAL, then query readout -> scored answer
  * RECOVERY_ENABLED_EQUIVALENT_WORK = step trajectory + in-run capture(K) + K+1 restore/readouts + MAX_CONF selection -> scored answer
Primary utility comparator (frozen): RECOVERY_ENABLED - FINAL_STEP (marginal cost of enabling recovery
on the capture-capable step path). Gate on p95 warm within ENVELOPE_MS=250. Persist RAW warm samples.
Multiple process starts supported via argv process index; a final aggregate mints the verdict.
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
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06T2")
TOK_ID = "EleutherAI/gpt-neox-20b"
POOL_SEED = 20260817
BATCH = 16
M = 192
SCHEDULE = D6.schedule_slots(M, 4)
CTX_LEN = 4 * M
CAP_BOUNDS = [4 * (s + 1) for s in SCHEDULE] + [CTX_LEN]
WARM_ITERS = 40
ENVELOPE_MS = 250.0            # frozen (T1R_PRE_REGISTRATION.md); gate on p95 of RECOVERY-FINAL_STEP


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


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


@torch.no_grad()
def final_fused_equiv(model, ctx, q, vt):
    """Fused/chunked prefill (no inference_params) then step the 2-token query from the resulting state
    -> constrained scored answer. Uses a fresh cache seeded by re-running the last-token continuation.
    To keep the SAME scored-answer semantics without a mid-seq capture, we do a plain fused prefill of
    ctx+query[:-1] then read the constrained logits at the final query token."""
    full = torch.cat([ctx, q], dim=1)
    logits = model(full).logits[:, -1, :].float()      # fused/chunked prefill over ctx+query
    sub = logits.index_select(1, vt)
    return vt[sub.argmax(-1)]


@torch.no_grad()
def final_step_equiv(model, ctx, q, vt):
    snaps = L.run_trajectory(model, ctx, [CTX_LEN])
    pred, _ = L.readout(model, snaps[CTX_LEN], q, CTX_LEN, vt)
    return pred


@torch.no_grad()
def recovery_equiv(model, ctx, q, vt, timing=None):
    t = {}
    sync(); t0 = time.time()
    snaps = L.run_trajectory(model, ctx, CAP_BOUNDS)
    sync(); t["trajectory_plus_capture_s"] = time.time() - t0
    sync(); t0 = time.time()
    subs = []
    for s in SCHEDULE:
        pos = 4 * (s + 1)
        _, sub = L.readout(model, snaps[pos], q, pos, vt)
        subs.append(sub.cpu().numpy())
    _, subf = L.readout(model, snaps[CTX_LEN], q, CTX_LEN, vt)
    sync(); t["restore_readout_s"] = time.time() - t0
    t0 = time.time()
    pl = np.stack(subs, axis=1)
    sel = softmax(pl).max(-1).argmax(1)
    pred = pl.argmax(-1)[np.arange(pl.shape[0]), sel]
    t["selection_s"] = time.time() - t0
    # state-copy / GPU->CPU already inside restore_readout (sub.cpu()); measure a dedicated copy sample
    sync(); t0 = time.time()
    _ = L.serialize_state(snaps[CAP_BOUNDS[0]])          # GPU->CPU snapshot copy sample
    sync(); t["gpu_to_cpu_snapshot_copy_s"] = time.time() - t0
    if timing is not None:
        timing.append(t)
    return torch.tensor(pred)


def stats(xs):
    a = np.array(xs, float)
    return {"n": len(a), "median": float(np.median(a)), "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)), "p95": float(np.percentile(a, 95)),
            "min": float(a.min()), "max": float(a.max())}


def main():
    pidx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    t_start = time.time(); runner = os.path.abspath(__file__)
    L.install_counters()
    import mamba_ssm  # noqa
    compile_t0 = time.time()
    tok = AutoTokenizer.from_pretrained(TOK_ID)
    pools, _ = L.build_official_pools(tok, POOL_SEED)
    vset = sorted(set(pools["scored_vals"])); vt = torch.tensor(vset, device=L.DEVICE, dtype=torch.long)
    model = L.load_model()
    compile_model_load_s = time.time() - compile_t0

    spec = json.load(open(os.path.join(OUTDIR, "T1R_WIDE_QUAL_SPEC.json")))
    ex = spec["examples"][:BATCH]
    ctx_l, q_l = [], []
    for e in ex:
        toks, _ = D6.materialize_d0(e, M, pools)
        ctx_l.append(toks[:CTX_LEN]); q_l.append(toks[-2:])
    ctx = torch.tensor(ctx_l, device=L.DEVICE, dtype=torch.long)
    q = torch.tensor(q_l, device=L.DEVICE, dtype=torch.long)
    proc = psutil.Process()

    def bench(fn):
        sync(); c0 = time.time(); fn(model, ctx, q, vt); sync(); cold = time.time() - c0
        warm = []
        for _ in range(WARM_ITERS):
            sync(); w0 = time.time(); fn(model, ctx, q, vt); sync(); warm.append(time.time() - w0)
        return cold, warm

    torch.cuda.reset_peak_memory_stats()
    ff_cold, ff_warm = bench(final_fused_equiv); vram_fused = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    fs_cold, fs_warm = bench(final_step_equiv); vram_step = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    rec_cold, rec_warm = bench(recovery_equiv); vram_rec = torch.cuda.max_memory_allocated() / 1e9

    timing = []; recovery_equiv(model, ctx, q, vt, timing=timing); comp = timing[-1]

    # per-query (batch-amortized) ms
    def pq(sec):
        return 1000.0 * sec / BATCH
    ff_pq = [pq(s) for s in ff_warm]; fs_pq = [pq(s) for s in fs_warm]; rec_pq = [pq(s) for s in rec_warm]
    added_vs_step = [r - s for r, s in zip(rec_pq, fs_pq)]
    added_vs_fused = [r - f for r, f in zip(rec_pq, ff_pq)]

    out = {"packet": "RNN-06T2-ECON", "process_index": pidx, "batch": BATCH,
           "compile_model_load_s": round(compile_model_load_s, 2),
           "warm_per_query_ms": {"FINAL_FUSED_EQUIVALENT_WORK": stats(ff_pq),
                                 "FINAL_STEP_EQUIVALENT_WORK": stats(fs_pq),
                                 "RECOVERY_ENABLED_EQUIVALENT_WORK": stats(rec_pq)},
           "cold_batch_s": {"final_fused": round(ff_cold, 4), "final_step": round(fs_cold, 4),
                            "recovery": round(rec_cold, 4)},
           "added_latency_per_query_ms": {"vs_final_step_primary": stats(added_vs_step),
                                          "vs_final_fused_descriptive": stats(added_vs_fused)},
           "recovery_components_warm_s": {k: round(v, 5) for k, v in comp.items()},
           "recovery_components_per_query_ms": {k: round(pq(v), 3) for k, v in comp.items()},
           "raw_warm_per_query_ms": {"final_fused": [round(x, 4) for x in ff_pq],
                                     "final_step": [round(x, 4) for x in fs_pq],
                                     "recovery": [round(x, 4) for x in rec_pq],
                                     "added_vs_step": [round(x, 4) for x in added_vs_step]},
           "memory": {"peak_vram_gb_final_fused": round(vram_fused, 3),
                      "peak_vram_gb_final_step": round(vram_step, 3),
                      "peak_vram_gb_recovery": round(vram_rec, 3),
                      "peak_cpu_ram_gb": round(proc.memory_info().rss / 1e9, 3),
                      "snapshot_bytes_per_seq_per_K": L.STATE_BYTES,
                      "snapshot_bytes_batch_KplusFinal": L.STATE_BYTES * (len(SCHEDULE) + 1) * BATCH},
           "envelope_ms_per_query": ENVELOPE_MS, "envelope_gate_statistic": "p95(added_vs_final_step)",
           "executed_source_identity": {"runner_git_blob": git("hash-object", runner),
                                        "runner_dirty": git("status", "--porcelain", "--", runner),
                                        "git_head": git("rev-parse", "HEAD"), "repo_id": L.REPO_ID,
                                        "revision": L.REVISION, "mamba_ssm": mamba_ssm.__version__,
                                        "protocol_sha256": sha256_file(os.path.join(OUTDIR, "T1R_PRE_REGISTRATION.md"))},
           "fast_path_active": bool(not any(L.fallback_reachable().values()) and L.KCOUNT["selective_state_update"] > 0),
           "total_runtime_s": round(time.time() - t_start, 1)}
    json.dump(out, open(os.path.join(OUTDIR, f"T1R_ECON_run{pidx}.json"), "w"), indent=2, default=str)
    print(f"[econ p{pidx}] FUSED={out['warm_per_query_ms']['FINAL_FUSED_EQUIVALENT_WORK']['median']:.2f}ms "
          f"STEP={out['warm_per_query_ms']['FINAL_STEP_EQUIVALENT_WORK']['median']:.2f}ms "
          f"REC={out['warm_per_query_ms']['RECOVERY_ENABLED_EQUIVALENT_WORK']['median']:.2f}ms")
    print(f"[econ p{pidx}] added vs STEP: median={out['added_latency_per_query_ms']['vs_final_step_primary']['median']:.2f} "
          f"p95={out['added_latency_per_query_ms']['vs_final_step_primary']['p95']:.2f}ms "
          f"(envelope {ENVELOPE_MS}, gate p95)")
    print(f"[econ p{pidx}] added vs FUSED(descriptive): median={out['added_latency_per_query_ms']['vs_final_fused_descriptive']['median']:.2f}ms")


if __name__ == "__main__":
    main()
