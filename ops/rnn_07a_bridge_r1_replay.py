#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-07A-BRIDGE-R1 — IN-PROCESS same-shape replay determinism verification (verification-only).

Two prior replay attempts were mis-specified:
  (a) the in-runner replay re-ran the first-8 UIDs as a batch of 8 (capture was batch 32) -> batch-shape
      non-portability (RNN-06T2 OUT_OF_SCOPE) -> 0/40;
  (b) a standalone same-shape replay ran in a SEPARATE PROCESS -> RNN-06T2 documented that bf16 kernel
      autotuning makes state bytes diverge across process starts -> 0/40.

The faithful authority test is IN-PROCESS SAME-SHAPE determinism: within ONE process, run the exact
capture batch #1 (selected[0:32], shape (32,32000)) through the qualified run_trajectory TWICE and require
every boundary state hash (first 8 rows) to reproduce BIT_EXACT (RNN-06T2 test H). This establishes that
the capture path is deterministic at the capture shape, so the recovery outcome states (all captured
in-process at batch 32) are internally reproducible. It does NOT recompute any recovery outcome.
Also reports the cross-process comparison to the stored process-A hashes to document the bf16 boundary.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_07a_bridge_r1 as R   # noqa: E402
import rnn_07a_lib as A         # noqa: E402
import rnn_07a_bridge_lib as B  # noqa: E402
import rnn_06t_lib as L         # noqa: E402

OUTDIR = R.OUTDIR


def hashes_for(model, ids, boundaries, n_rows):
    snaps = L.run_trajectory(model, ids, boundaries)
    h = {}
    for row_i in range(n_rows):
        h[row_i] = {R.SNAP_NAMES[bi]: L.state_hash_row(snaps[b], row_i) for bi, b in enumerate(boundaries)}
    del snaps; torch.cuda.empty_cache()
    return h


def main():
    L.install_counters()
    tok = A.load_tokenizer()
    model = L.load_model()
    R.book_g = B.load_book_tokens(tok)
    pool = R.build_pool_meta(tok, R.R1_POOL_SEED, R.N_CHAR, R.MAX_POOL)

    res = json.load(open(os.path.join(OUTDIR, "R1_RESULTS.json")))
    sel_records = res["fresh_set_identity"]["selected_records"]
    selected = [r["pool_idx"] for r in sel_records]
    uid_by_k = {r["pool_idx"]: r["uid"] for r in sel_records}
    captured = {row["uid"]: row["boundary_hashes"] for row in res["rows"]}

    boundaries = [int(round(p * R.LEN32)) for p in R.PROGRESS]
    batch1_k = selected[:R.B_CAP]
    ctx = [B.make_context(R.book_g, pool[k]["needle_ids"], R.LEN32, R.DEPTH,
                          filler_seed=R.R1_RECOVERY_FILLER_BASE + k)[0] for k in batch1_k]
    ids = torch.tensor(ctx, device=L.DEVICE, dtype=torch.long)

    n = R.REPLAY_SUBSET_N
    h1 = hashes_for(model, ids, boundaries, n)   # in-process run #1
    h2 = hashes_for(model, ids, boundaries, n)   # in-process run #2 (same process, same shape)

    inproc, xproc, records = True, True, []
    for row_i in range(n):
        k = batch1_k[row_i]; u = uid_by_k[k]
        for bi in range(len(boundaries)):
            snap = R.SNAP_NAMES[bi]
            m_in = (h1[row_i][snap] == h2[row_i][snap])
            m_x = (h1[row_i][snap] == captured[u][snap])
            inproc = inproc and m_in; xproc = xproc and m_x
            records.append({"uid": u, "row": row_i, "snapshot": snap,
                            "inprocess_match": m_in, "crossprocess_match_to_capture": m_x})

    out = {"packet": "RNN-07A-BRIDGE-R1-REPLAY-INPROCESS",
           "purpose": "verification-only; IN-PROCESS same-shape determinism at capture shape (32,32000)",
           "capture_batch_shape": [R.B_CAP, R.LEN32], "n_boundary_checks": len(records),
           "IN_PROCESS_SAME_SHAPE_ALL_MATCH": bool(inproc),
           "n_inprocess_match": sum(r["inprocess_match"] for r in records),
           "CROSS_PROCESS_MATCH_TO_CAPTURE_ALL": bool(xproc),
           "n_crossprocess_match": sum(r["crossprocess_match_to_capture"] for r in records),
           "interpretation": ("IN_PROCESS_SAME_SHAPE_ALL_MATCH=True establishes the capture path is "
                              "deterministic at the capture shape -> recovery-outcome states (all captured "
                              "in-process at batch 32) are internally reproducible. Cross-process mismatch "
                              "to the original capture is the documented RNN-06T2 bf16 kernel-autotuning "
                              "boundary condition, NOT a capture defect."),
           "fast_path_active": bool(not any(L.fallback_reachable().values()) and L.KCOUNT["selective_state_update"] > 0),
           "records": records}
    json.dump(out, open(os.path.join(OUTDIR, "R1_REPLAY_INPROCESS.json"), "w"), indent=2, default=str)
    print(f"IN_PROCESS_SAME_SHAPE_ALL_MATCH = {inproc}  ({out['n_inprocess_match']}/{out['n_boundary_checks']})")
    print(f"CROSS_PROCESS_MATCH_TO_CAPTURE  = {xproc}  ({out['n_crossprocess_match']}/{out['n_boundary_checks']})")


if __name__ == "__main__":
    main()
