#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06B3 — bounded EXPLORATORY calibration for the ORDER-STABLE subpacked state-load design.

Predeclared candidate family (chosen BEFORE running): M ∈ {128, 192}, MIN_SENTINEL_RESERVE ∈
{16, 32}, sentinel = REPEAT1 (carried from B2), DS arm, coarse subpacked U grid (never U=M).
Chooses M / reserve / U-grid / batching for the frozen B3 qualification. Verifies the nested
binding-identity invariant for the calibration grid. NOT a gate; mints nothing. No cherry-picking.

b3CalibrationSetSha256 recorded. Records executed-source identity. cs=32.
At most ONE append-only grid extension is permitted (justified before executing it).
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import torch
from transformers import Mamba2ForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_06b3_lib as lib  # noqa: E402

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06B3-MAMBA-CONTROLLED-STATE-LOAD-PERTURBATION")
REPO_ID = "AntonV/mamba2-1.3b-hf"
REVISION = "703e19a43f397c70315244a3424d79456b54fb34"
DEVICE, DTYPE, CHUNK = "cuda", torch.bfloat16, 32
AUTOBATCH = 1536
CALIB_SEED = 20260816

CAND_M = [128, 192]
CAND_RESERVE = [16, 32]
N_CALIB = 48
SENTINEL = "REPEAT1"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*a):
    try:
        return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as e:
        return f"<git-error:{e}>"


@torch.no_grad()
def score(model, prompts, golds, vtensor, vset, bs):
    n = len(prompts); k = 0
    for b in range(0, n, bs):
        ids = torch.tensor(prompts[b:b + bs], device=DEVICE, dtype=torch.long)
        last = model(ids).logits[:, -1, :].float()
        con = vtensor[last.index_select(1, vtensor).argmax(-1)]
        for j in range(ids.shape[0]):
            if int(con[j]) == int(golds[b + j]):
                k += 1
    return k, n


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(0)
    t0 = time.time()
    runner = os.path.abspath(__file__)
    libpath = os.path.join(os.path.dirname(runner), "rnn_06b3_lib.py")
    b2libpath = os.path.join(os.path.dirname(runner), "rnn_06b2_lib.py")
    from transformers.models.mamba2 import modeling_mamba2
    rec = {"packet": "RNN-06B3", "kind": "calibration_exploratory",
           "candidate_family": {"M": CAND_M, "reserve": CAND_RESERVE, "sentinel": SENTINEL,
                                "arm": "DS", "n_calib": N_CALIB}}
    rec["executed_source_identity"] = {
        "runner_sha256": sha256_file(runner), "runner_git_blob": git("hash-object", runner),
        "runner_dirty": git("status", "--porcelain", "--", runner),
        "lib_sha256": sha256_file(libpath), "lib_git_blob": git("hash-object", libpath),
        "b2lib_sha256": sha256_file(b2libpath),
        "git_head": git("rev-parse", "HEAD"),
        "modeling_mamba2_sha256": sha256_file(modeling_mamba2.__file__),
        "transformers": __import__("transformers").__version__, "torch": torch.__version__,
        "repo_id": REPO_ID, "revision": REVISION, "chunk_size": CHUNK, "dtype": str(DTYPE),
        "is_fast_path_available": modeling_mamba2.is_fast_path_available}
    assert modeling_mamba2.is_fast_path_available is False

    tok = AutoTokenizer.from_pretrained(REPO_ID, revision=REVISION)
    model = Mamba2ForCausalLM.from_pretrained(REPO_ID, revision=REVISION, torch_dtype=DTYPE).to(DEVICE).eval()
    for blk in model.backbone.layers:
        blk.mixer.chunk_size = CHUNK
    model.config.chunk_size = CHUNK
    pools, pool_meta = lib.build_pools(tok, CALIB_SEED)
    rec["pools"] = pool_meta
    vset = set(pools["scored_vals"]); vtensor = torch.tensor(sorted(vset), device=DEVICE, dtype=torch.long)
    rec["chance"] = 1.0 / len(vset)

    # calibration specs per candidate M
    specs_by_M = {M: [lib.build_example_spec(CALIB_SEED, i, 0, M) for i in range(N_CALIB)]
                  for M in CAND_M}
    calib_abstract = {"calib_seed": CALIB_SEED, "n_calib": N_CALIB, "cand_M": CAND_M,
                      "cand_reserve": CAND_RESERVE, "sentinel": SENTINEL,
                      "specs": {str(M): specs_by_M[M] for M in CAND_M}}
    rec["b3CalibrationSetSha256"] = lib.sha256_of_obj(calib_abstract)

    surface = []
    nested_all_pass = True
    torch.cuda.reset_peak_memory_stats()
    for M in CAND_M:
        for reserve in CAND_RESERVE:
            u_max = M - reserve
            U_grid = sorted(set([u for u in [1, 8, 24, 48, 72, 96, 128, u_max] if u <= u_max]))
            # nested-identity self-check for this grid (DS)
            ok, detail = lib.nested_identity_check(specs_by_M[M][0], M, U_grid, "DS", pools)
            nested_all_pass = nested_all_pass and ok
            for U in U_grid:
                prompts, golds = [], []
                for spec in specs_by_M[M]:
                    p, g = lib.materialize_b3(spec, M, U, "DS", pools, reserve)
                    prompts.append(p); golds.append(g)
                seq_len = len(prompts[0])
                assert all(len(p) == seq_len for p in prompts)
                bs = max(1, min(64, AUTOBATCH // seq_len))
                k, n = score(model, prompts, golds, vtensor, vset, bs)
                surface.append({"M": M, "reserve": reserve, "U": U, "seq_len": seq_len,
                                "sentinel_slots": M - U, "n": n, "correct": k, "acc": k / n})
                print(f"[b3calib] M={M} res={reserve} U={U:3d} sent={M-U:3d} len={seq_len} "
                      f"acc={k/n:.3f} ({k}/{n})", file=sys.stderr)
    rec["surface"] = surface
    rec["nested_identity_all_pass"] = nested_all_pass
    rec["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    rec["runtime_s"] = round(time.time() - t0, 1)

    # feasibility per (M,reserve): low-load acc, max-subpacked acc, drop
    summ = {}
    for M in CAND_M:
        for reserve in CAND_RESERVE:
            rows = [r for r in surface if r["M"] == M and r["reserve"] == reserve]
            if not rows:
                continue
            accs = {r["U"]: r["acc"] for r in rows}
            low = accs[min(accs)]; high = accs[max(accs)]
            summ[f"M{M}_res{reserve}"] = {"low_load_acc": low, "max_subpacked_U": max(accs),
                                          "max_subpacked_acc": high, "drop": round(low - high, 3),
                                          "competent": low >= 0.75, "material_drop_ge_0.15": (low - high) >= 0.15}
    rec["feasibility_summary"] = summ
    with open(os.path.join(OUTDIR, "B3_CALIBRATION.json"), "w") as f:
        json.dump(rec, f, indent=2, default=str)
    print("\n== feasibility ==", file=sys.stderr)
    for k, v in summ.items():
        print(f"  {k}: low={v['low_load_acc']:.3f} maxU={v['max_subpacked_U']} "
              f"high={v['max_subpacked_acc']:.3f} drop={v['drop']} material={v['material_drop_ge_0.15']}",
              file=sys.stderr)
    print(f"nested_identity_all_pass={nested_all_pass}", file=sys.stderr)
    print(f"b3CalibrationSetSha256={rec['b3CalibrationSetSha256']}", file=sys.stderr)


if __name__ == "__main__":
    main()
