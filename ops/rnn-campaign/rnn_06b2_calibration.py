#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06B2 — bounded EXPLORATORY calibration for the fixed-length state-load construction.

Sweeps candidate fixed slot count M x sentinel scheme x a coarse unique-load ladder (DS arm
only) on the frozen subject to find a feasible surface (low-load competent, high-load capable
of measurable loss, fixed length feasible). Chooses M / dose ladder / sentinel / batching for
the frozen B2 qualification. NOT a gate; mints nothing. No cherry-picking of examples.

b2CalibrationSetSha256 recorded. Records executed-source identity. cs=32.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import Mamba2ForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_06b2_lib as lib  # noqa: E402

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD")
REPO_ID = "AntonV/mamba2-1.3b-hf"
REVISION = "703e19a43f397c70315244a3424d79456b54fb34"
DEVICE, DTYPE, CHUNK = "cuda", torch.bfloat16, 32
AUTOBATCH = 1536
CALIB_SEED = 20260814

CAND_M = [48, 64, 96, 128]
CAND_SENT = ["REPEAT1", "CYCLE4"]
N_CALIB = 48


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
    from transformers.models.mamba2 import modeling_mamba2, configuration_mamba2
    rec = {"packet": "RNN-06B2", "kind": "calibration_exploratory"}
    rec["executed_source_identity"] = {
        "runner_sha256": sha256_file(runner), "runner_git_blob": git("hash-object", runner),
        "runner_dirty": git("status", "--porcelain", "--", runner),
        "lib_sha256": sha256_file(os.path.join(os.path.dirname(runner), "rnn_06b2_lib.py")),
        "lib_git_blob": git("hash-object", os.path.join(os.path.dirname(runner), "rnn_06b2_lib.py")),
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
    vset = set(pools["scored_vals"])
    vtensor = torch.tensor(sorted(vset), device=DEVICE, dtype=torch.long)
    rec["chance"] = 1.0 / len(vset)

    # calibration example specs (per candidate M) — abstract, deterministic
    specs_by_M = {M: [lib.build_example_spec(CALIB_SEED, i, 0, M) for i in range(N_CALIB)]
                  for M in CAND_M}
    calib_abstract = {"calib_seed": CALIB_SEED, "n_calib": N_CALIB, "cand_M": CAND_M,
                      "cand_sentinel": CAND_SENT,
                      "specs": {str(M): specs_by_M[M] for M in CAND_M}}
    rec["b2CalibrationSetSha256"] = lib.sha256_of_obj(calib_abstract)

    surface = []
    torch.cuda.reset_peak_memory_stats()
    for M in CAND_M:
        U_ladder = sorted(set([1, 2, 4, 8, 16, 24, 32, 48, 64, 96, M]))
        U_ladder = [u for u in U_ladder if u <= M]
        for sent in CAND_SENT:
            for U in U_ladder:
                prompts, golds = [], []
                for spec in specs_by_M[M]:
                    p, g = lib.materialize(spec, M, U, "DS", sent, pools)
                    prompts.append(p); golds.append(g)
                seq_len = len(prompts[0])
                bs = max(1, min(64, AUTOBATCH // seq_len))
                k, n = score(model, prompts, golds, vtensor, vset, bs)
                surface.append({"M": M, "sentinel": sent, "U": U, "seq_len": seq_len,
                                "n": n, "correct": k, "constrained_acc": k / n})
                print(f"[calib] M={M} {sent} U={U:3d} len={seq_len} acc={k/n:.3f} ({k}/{n})",
                      file=sys.stderr)
    rec["surface"] = surface
    rec["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    rec["runtime_s"] = round(time.time() - t0, 1)

    # feasibility summary per (M,sentinel): low-load acc (U=1), high-load acc (U=M), interior in (0.3,0.75)
    summ = {}
    for M in CAND_M:
        for sent in CAND_SENT:
            rows = [r for r in surface if r["M"] == M and r["sentinel"] == sent]
            accs = {r["U"]: r["constrained_acc"] for r in rows}
            low = accs.get(1, accs[min(accs)])
            high = accs[max(accs)]
            interior = sum(1 for u, a in accs.items() if 0.30 < a < 0.75)
            summ[f"M{M}_{sent}"] = {"low_load_acc": low, "high_load_acc": high,
                                    "drop": round(low - high, 3), "interior_doses": interior,
                                    "feasible": low >= 0.75 and high <= 0.45 and interior >= 2}
    rec["feasibility_summary"] = summ
    with open(os.path.join(OUTDIR, "B2_CALIBRATION.json"), "w") as f:
        json.dump(rec, f, indent=2, default=str)
    print("\n== feasibility summary ==", file=sys.stderr)
    for k, v in summ.items():
        print(f"  {k}: low={v['low_load_acc']:.3f} high={v['high_load_acc']:.3f} "
              f"interior={v['interior_doses']} feasible={v['feasible']}", file=sys.stderr)
    print(f"b2CalibrationSetSha256={rec['b2CalibrationSetSha256']}", file=sys.stderr)


if __name__ == "__main__":
    main()
