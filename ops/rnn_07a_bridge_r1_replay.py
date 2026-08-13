#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-07A-BRIDGE-R1 — SAME-BATCH-SHAPE replay verification (verification-only; preserves R1 outcome).

The in-runner replay (R1_RESULTS.json temporal_identity_replay) re-ran the first-8 UIDs as a batch of 8,
whereas they were captured in a batch of 32 (capture batch #1 = selected[0:32]). RNN-06T2 established
BATCH_SHAPE_NUMERICAL_PORTABILITY = OUT_OF_SCOPE (batch-size states are not bit-portable), so that replay
was mis-specified (not "same path") and reported 0/40 — a verification artifact, NOT a capture defect.

This script re-runs the EXACT capture-batch #1 (selected[0:32], shape (32, 32000)) and compares the
first-8 rows' boundary state hashes to the captured hashes in R1_RESULTS.json. In-process same-shape
same-position replay should reproduce every boundary hash BIT_EXACT (RNN-06T2 test H). This establishes
capture reproducibility / recovery authority. It does NOT recompute any recovery outcome.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_07a_bridge_r1 as R   # noqa: E402  (deterministic pool/selection/context rebuild)
import rnn_07a_lib as A         # noqa: E402
import rnn_07a_bridge_lib as B  # noqa: E402
import rnn_06t_lib as L         # noqa: E402

OUTDIR = R.OUTDIR


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
    # rebuild EXACT capture batch #1 (same shape 32, same row order)
    batch1_k = selected[:R.B_CAP]
    ctx = []
    for k in batch1_k:
        c, _ = B.make_context(R.book_g, pool[k]["needle_ids"], R.LEN32, R.DEPTH,
                              filler_seed=R.R1_RECOVERY_FILLER_BASE + k)
        ctx.append(c)
    ids = torch.tensor(ctx, device=L.DEVICE, dtype=torch.long)
    snaps = L.run_trajectory(model, ids, boundaries)

    # compare first REPLAY_SUBSET_N rows (frozen subset) — same batch shape (32) and same row position
    records, all_match = [], True
    for row_i in range(R.REPLAY_SUBSET_N):
        k = batch1_k[row_i]; u = uid_by_k[k]
        for bi, b in enumerate(boundaries):
            h2 = L.state_hash_row(snaps[b], row_i)
            h1 = captured[u][R.SNAP_NAMES[bi]]
            m = (h1 == h2); all_match = all_match and m
            records.append({"uid": u, "row": row_i, "snapshot": R.SNAP_NAMES[bi], "match": m,
                            "captured": h1[:16], "replay": h2[:16]})
    out = {"packet": "RNN-07A-BRIDGE-R1-REPLAY-SAMESHAPE",
           "purpose": "verification-only; preserves R1 outcome; same batch shape (32) as capture batch #1",
           "replay_batch_shape": list(ids.shape), "capture_batch_shape": [R.B_CAP, R.LEN32],
           "in_runner_replay_batch_shape": [R.REPLAY_SUBSET_N, R.LEN32],
           "explanation": "in-runner replay used batch 8 vs capture batch 32 -> batch-shape non-portability "
                          "(RNN-06T2 OUT_OF_SCOPE) -> 0/40; this same-shape replay is the faithful test.",
           "n_boundary_checks": len(records), "n_match": sum(r["match"] for r in records),
           "SAME_SHAPE_REPLAY_ALL_MATCH": bool(all_match),
           "fast_path_active": bool(not any(L.fallback_reachable().values()) and L.KCOUNT["selective_state_update"] > 0),
           "records": records}
    json.dump(out, open(os.path.join(OUTDIR, "R1_REPLAY_SAMESHAPE.json"), "w"), indent=2, default=str)
    print(f"SAME_SHAPE_REPLAY_ALL_MATCH = {all_match}  ({out['n_match']}/{out['n_boundary_checks']})")


if __name__ == "__main__":
    main()
