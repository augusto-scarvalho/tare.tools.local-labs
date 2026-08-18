#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06T2-E1 — economics semantic closure (append-only successor to ops/rnn_06t2_econ.py).

Fixes ONE false-green: the recovery arm returned a scored-vocabulary COLUMN INDEX while
FINAL_FUSED/FINAL_STEP returned scored VALUE TOKEN IDs. Here every arm returns the SAME scored
VALUE TOKEN ID domain, proven by executable output-domain assertions run before any timing is trusted.
Timing uses randomized/interleaved cycles (one iter of each arm per cycle, shuffled order) across >=2
clean process starts. Historical econ files are untouched; outputs land in runs/rnn/RNN-06T2-E1/.

Does NOT rerun lifecycle or synthetic recovery qualification.
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
from mamba_ssm.utils.generation import InferenceParams  # noqa: F401 (parity w/ historical env)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_06t_lib as L      # noqa: E402
import rnn_06d_lib as D6     # noqa: E402

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06T2-E1")
HIST_DIR = os.path.join(REPO, "runs", "rnn", "RNN-06T2")
TOK_ID = "EleutherAI/gpt-neox-20b"
POOL_SEED = 20260817
E1_SHUFFLE_SEED_BASE = 20261200
BATCH = 16
M = 192
SCHEDULE = D6.schedule_slots(M, 4)
CTX_LEN = 4 * M
CAP_BOUNDS = [4 * (s + 1) for s in SCHEDULE] + [CTX_LEN]
WARM_ITERS = 40
ENVELOPE_MS = 250.0            # frozen in T1R_PRE_REGISTRATION.md BEFORE T1R outcomes; reused verbatim


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
    """Fused/chunked prefill over ctx+query, read constrained logits at the final query token,
    return the scored VALUE TOKEN ID (element of vt)."""
    full = torch.cat([ctx, q], dim=1)
    logits = model(full).logits[:, -1, :].float()
    sub = logits.index_select(1, vt)
    return vt[sub.argmax(-1)]                         # -> token ids


@torch.no_grad()
def final_step_equiv(model, ctx, q, vt):
    snaps = L.run_trajectory(model, ctx, [CTX_LEN])
    pred, _ = L.readout(model, snaps[CTX_LEN], q, CTX_LEN, vt)
    return pred                                        # L.readout returns vt[argmax] -> token ids


@torch.no_grad()
def recovery_equiv(model, ctx, q, vt, timing=None, return_colidx=False):
    """Step trajectory + in-run capture(K) + K+1 restore/readouts + MAX_CONF selection.
    FIX: return the scored VALUE TOKEN ID (vt[selected column]) instead of the raw column index, so the
    output domain matches FINAL_FUSED/FINAL_STEP. `return_colidx=True` returns the pre-fix column index
    (used only by the output-domain assertion to prove the fix changed the domain)."""
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
    pl = np.stack(subs, axis=1)                        # (B, K, V) constrained logits per snapshot
    sel = softmax(pl).max(-1).argmax(1)                # MAX_CONFIDENCE snapshot selection
    col_idx = pl.argmax(-1)[np.arange(pl.shape[0]), sel]   # column index into vt (the OLD, buggy output)
    vt_np = vt.detach().cpu().numpy()
    pred_tok = vt_np[col_idx]                          # FIX: map column index -> scored VALUE TOKEN ID
    t["selection_s"] = time.time() - t0
    sync(); t0 = time.time()
    _ = L.serialize_state(snaps[CAP_BOUNDS[0]])
    sync(); t["gpu_to_cpu_snapshot_copy_s"] = time.time() - t0
    if timing is not None:
        timing.append(t)
    if return_colidx:
        return torch.tensor(pred_tok), torch.tensor(col_idx)
    return torch.tensor(pred_tok)                      # -> token ids


def stats(xs):
    a = np.array(xs, float)
    return {"n": len(a), "median": float(np.median(a)), "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)), "p95": float(np.percentile(a, 95)),
            "min": float(a.min()), "max": float(a.max())}


def output_domain_assertions(model, ctx, q, vt):
    """Run each arm once; assert all arms return scored VALUE TOKEN IDs in the frozen vt set.
    Returns a dict of evidence (also proves the OLD recovery column-index output was out-of-domain)."""
    vt_set = set(int(x) for x in vt.detach().cpu().numpy().tolist())
    vt_min, vt_max = int(vt.min()), int(vt.max())
    n_vt = int(vt.numel())

    ff = final_fused_equiv(model, ctx, q, vt).detach().cpu().numpy().tolist()
    fs = final_step_equiv(model, ctx, q, vt).detach().cpu().numpy().tolist()
    rec_tok_t, rec_col_t = recovery_equiv(model, ctx, q, vt, return_colidx=True)
    rec = rec_tok_t.detach().cpu().numpy().tolist()
    rec_col = rec_col_t.detach().cpu().numpy().tolist()

    def all_in(xs):
        return all(int(x) in vt_set for x in xs)

    ev = {
        "vt_size": n_vt, "vt_min": vt_min, "vt_max": vt_max,
        "final_fused_all_in_vt": all_in(ff),
        "final_step_all_in_vt": all_in(fs),
        "recovery_all_in_vt": all_in(rec),
        "recovery_old_colidx_all_in_vt": all_in(rec_col),      # expected False -> proves the fix mattered
        "recovery_fix_changed_output": bool(any(int(a) != int(b) for a, b in zip(rec, rec_col))),
        "final_fused_sample": ff[:8], "final_step_sample": fs[:8],
        "recovery_sample_tokenid": rec[:8], "recovery_old_colidx_sample": rec_col[:8],
        "cross_arm_domain_identical": True,  # all arms constrained to the same vt set by construction
    }
    # Hard executable assertions (DOMAIN_MEMBERSHIP, DTYPE_RANGE, NOT_COLUMN_INDEX)
    assert ev["final_fused_all_in_vt"], "FINAL_FUSED produced an out-of-domain output"
    assert ev["final_step_all_in_vt"], "FINAL_STEP produced an out-of-domain output"
    assert ev["recovery_all_in_vt"], "RECOVERY produced an out-of-domain output (fix failed)"
    assert min(min(ff), min(fs), min(rec)) >= vt_min and max(max(ff), max(fs), max(rec)) <= vt_max, \
        "DTYPE_RANGE violated"
    # NOT_COLUMN_INDEX: because vt token ids are >> len(vt), column indices are (almost surely) out of
    # the vt set; assert the OLD output was out-of-domain, i.e. the false-green was real and is now closed.
    assert not ev["recovery_old_colidx_all_in_vt"], \
        "column-index output was already in vt-domain; NOT_COLUMN_INDEX proof inconclusive"
    ev["ASSERTIONS_PASSED"] = True
    return ev


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

    spec = json.load(open(os.path.join(HIST_DIR, "T1R_WIDE_QUAL_SPEC.json")))
    ex = spec["examples"][:BATCH]
    ctx_l, q_l = [], []
    for e in ex:
        toks, _ = D6.materialize_d0(e, M, pools)
        ctx_l.append(toks[:CTX_LEN]); q_l.append(toks[-2:])
    ctx = torch.tensor(ctx_l, device=L.DEVICE, dtype=torch.long)
    q = torch.tensor(q_l, device=L.DEVICE, dtype=torch.long)
    proc = psutil.Process()

    # ---- Output-domain assertions FIRST (before timing is trusted) ----
    domain_ev = output_domain_assertions(model, ctx, q, vt)

    # ---- Randomized / interleaved timing ----
    arms = {"final_fused": final_fused_equiv, "final_step": final_step_equiv, "recovery": recovery_equiv}
    rng = np.random.default_rng(E1_SHUFFLE_SEED_BASE + pidx)
    warm = {k: [] for k in arms}
    vram = {}
    # cold pass + peak vram per arm (fixed order, once)
    for name, fn in arms.items():
        torch.cuda.reset_peak_memory_stats()
        sync(); c0 = time.time(); fn(model, ctx, q, vt); sync()
        vram[name] = torch.cuda.max_memory_allocated() / 1e9
    # interleaved warm cycles: one timed iter per arm per cycle, shuffled order
    order_log = []
    for _ in range(WARM_ITERS):
        names = list(arms.keys()); rng.shuffle(names); order_log.append(names[:])
        for name in names:
            sync(); w0 = time.time(); arms[name](model, ctx, q, vt); sync()
            warm[name].append(time.time() - w0)

    timing = []; recovery_equiv(model, ctx, q, vt, timing=timing); comp = timing[-1]

    def pq(sec):
        return 1000.0 * sec / BATCH
    ff_pq = [pq(s) for s in warm["final_fused"]]
    fs_pq = [pq(s) for s in warm["final_step"]]
    rec_pq = [pq(s) for s in warm["recovery"]]
    added_vs_step = [r - s for r, s in zip(rec_pq, fs_pq)]
    added_vs_fused = [r - f for r, f in zip(rec_pq, ff_pq)]

    out = {"packet": "RNN-06T2-E1-ECON", "process_index": pidx, "batch": BATCH,
           "timing_method": "randomized_interleaved_cycles_one_iter_per_arm",
           "shuffle_seed": E1_SHUFFLE_SEED_BASE + pidx,
           "compile_model_load_s": round(compile_model_load_s, 2),
           "output_domain_evidence": domain_ev,
           "warm_per_query_ms": {"FINAL_FUSED_EQUIVALENT_WORK": stats(ff_pq),
                                 "FINAL_STEP_EQUIVALENT_WORK": stats(fs_pq),
                                 "RECOVERY_ENABLED_EQUIVALENT_WORK": stats(rec_pq)},
           "added_latency_per_query_ms": {"vs_final_step_primary": stats(added_vs_step),
                                          "vs_final_fused_descriptive": stats(added_vs_fused)},
           "recovery_components_per_query_ms": {k: round(pq(v), 3) for k, v in comp.items()},
           "raw_warm_per_query_ms": {"final_fused": [round(x, 4) for x in ff_pq],
                                     "final_step": [round(x, 4) for x in fs_pq],
                                     "recovery": [round(x, 4) for x in rec_pq],
                                     "added_vs_step": [round(x, 4) for x in added_vs_step]},
           "cycle_order_log_first5": order_log[:5],
           "memory": {"peak_vram_gb_final_fused": round(vram["final_fused"], 3),
                      "peak_vram_gb_final_step": round(vram["final_step"], 3),
                      "peak_vram_gb_recovery": round(vram["recovery"], 3),
                      "peak_cpu_ram_gb": round(proc.memory_info().rss / 1e9, 3)},
           "envelope_ms_per_query": ENVELOPE_MS, "envelope_gate_statistic": "p95(added_vs_final_step)",
           "executed_source_identity": {"runner_git_blob": git("hash-object", runner),
                                        "runner_dirty": git("status", "--porcelain", "--", runner),
                                        "git_head": git("rev-parse", "HEAD"), "repo_id": L.REPO_ID,
                                        "revision": L.REVISION, "mamba_ssm": mamba_ssm.__version__,
                                        "protocol_sha256": sha256_file(os.path.join(OUTDIR, "E1_PRE_REGISTRATION.md"))},
           "fast_path_active": bool(not any(L.fallback_reachable().values()) and L.KCOUNT["selective_state_update"] > 0),
           "total_runtime_s": round(time.time() - t_start, 1)}
    json.dump(out, open(os.path.join(OUTDIR, f"E1_ECON_run{pidx}.json"), "w"), indent=2, default=str)
    print(f"[e1 econ p{pidx}] ASSERTIONS_PASSED={domain_ev['ASSERTIONS_PASSED']} "
          f"rec_all_in_vt={domain_ev['recovery_all_in_vt']} old_colidx_in_vt={domain_ev['recovery_old_colidx_all_in_vt']}")
    print(f"[e1 econ p{pidx}] FUSED={out['warm_per_query_ms']['FINAL_FUSED_EQUIVALENT_WORK']['median']:.2f}ms "
          f"STEP={out['warm_per_query_ms']['FINAL_STEP_EQUIVALENT_WORK']['median']:.2f}ms "
          f"REC={out['warm_per_query_ms']['RECOVERY_ENABLED_EQUIVALENT_WORK']['median']:.2f}ms")
    print(f"[e1 econ p{pidx}] added vs STEP p95={out['added_latency_per_query_ms']['vs_final_step_primary']['p95']:.2f}ms "
          f"(envelope {ENVELOPE_MS}); added vs FUSED median={out['added_latency_per_query_ms']['vs_final_fused_descriptive']['median']:.2f}ms")


if __name__ == "__main__":
    main()
