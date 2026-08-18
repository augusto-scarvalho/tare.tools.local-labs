"""A2 reconstruction gate: does `base + ThinkingCap-LoRA(λ=1)` actually reproduce the full
ThinkingCap fine-tune? This is the FIRST test the research protocol (§23.6) demands, and the
one the community LoRA card never did -- it ships with zero reconstruction metrics.

Why it gates everything downstream: the adapter is a rank-64 SVD of the full-FT weight delta.
The long-to-short literature (arXiv 2503.20641, 2410.21228) is explicit that a truncated-SVD
adapter reconstructs a full fine-tune ONLY when that delta is genuinely low-rank; a full FT's
delta often is not (the "intruder dimensions" result). If the rank-64 adapter does NOT
reconstruct ThinkingCap on its OWN origin base, then:
  * the DavidAU transfer is meaningless (you'd be transferring an adapter that doesn't even
    carry the behavior), and
  * we learn the concision lives outside the rank-64 subspace -- a real finding.

The gate is behavioral, not a weight norm: run the SAME prompts, greedy, through three arms
produced by a2_concision_bench with a shared --tag --subset:
  A) base            = qwen36-27b-dense                     (verbose reference)
  B) base + LoRA@1.0 = qwen36-27b-dense + thinkingcap-lora  (the reconstruction candidate)
  C) full ThinkingCap= thinkingcap-27b                      (the target behavior)

Two independent signals, both must agree for a PASS:
  1. CONCISION match: B's reasoning-token length tracks C, not A. (Does the adapter make the
     base concise like ThinkingCap?)
  2. OUTPUT fidelity: B's text is materially more similar to C than to A, per problem
     (difflib ratio on reasoning+answer). (Is B behaving like C, or just coincidentally
     short?) Similarity is a proxy for the logit-level agreement we cannot cheaply read on
     the pinned server; at greedy temp=0 a faithful adapter should track the target's actual
     wording, not merely its length.

    # produce the three arms first (small subset is enough for the gate):
    python a2_concision_bench.py --model qwen36-27b-dense --workload gsm8k --subset 30 --tag recon
    python a2_concision_bench.py --model qwen36-27b-dense --workload gsm8k --subset 30 --tag recon --lora thinkingcap-lora-r64 --lora-lambda 1.0
    python a2_concision_bench.py --model thinkingcap-27b   --workload gsm8k --subset 30 --tag recon
    # then:
    python a2_reconstruct_gate.py --tag recon --workload gsm8k
"""
from __future__ import annotations

import argparse
import json
import pathlib
from difflib import SequenceMatcher

import numpy as np

RUNS = pathlib.Path(__file__).parent / "runs" / "a2"


def _load(stem: str) -> dict[str, dict]:
    path = RUNS / f"{stem}.json"
    if not path.exists():
        raise SystemExit(f"missing arm: {path}\n(run a2_concision_bench for it first)")
    return {r["task_id"]: r for r in json.loads(path.read_text(encoding="utf-8"))
            if r.get("task_id")}


def _full(rec: dict) -> str:
    """The whole behavioral trace: reasoning + answer. Reconstruction should match BOTH."""
    return (rec.get("reasoning_text") or "") + "\n" + (rec.get("completion") or rec.get("text") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--base", default="qwen36-27b-dense")
    ap.add_argument("--cap", default="thinkingcap-27b")
    ap.add_argument("--lora", default="thinkingcap-lora-r64")
    ap.add_argument("--lora-lambda", type=float, default=1.0)
    # PASS thresholds. Conservative on purpose -- a marginal reconstruction is a fail, because
    # everything downstream inherits its fidelity.
    ap.add_argument("--len-tol", type=float, default=0.25,
                    help="B's median reasoning length must be within this frac of C's")
    ap.add_argument("--sim-margin", type=float, default=0.10,
                    help="median sim(B,C) must exceed median sim(B,A) by at least this")
    args = ap.parse_args()

    A = _load(f"{args.tag}__{args.base}__{args.workload}")
    B = _load(f"{args.tag}__{args.base}__{args.workload}__{args.lora}-l{args.lora_lambda}")
    C = _load(f"{args.tag}__{args.cap}__{args.workload}")
    ids = sorted(set(A) & set(B) & set(C))
    if not ids:
        raise SystemExit("no common task_ids across the three arms")

    def med_rt(arm):
        v = [arm[i]["reasoning_tokens"] for i in ids
             if arm[i].get("reasoning_tokens") is not None]
        return float(np.median(v)) if v else float("nan")

    a_rt, b_rt, c_rt = med_rt(A), med_rt(B), med_rt(C)

    # signal 1: concision. B should sit near C, far from A.
    len_ratio = b_rt / c_rt if c_rt else float("nan")   # 1.0 = exact length reconstruction
    concise_pass = abs(len_ratio - 1.0) <= args.len_tol

    # signal 2: per-problem output fidelity (B closer to C than to A).
    sim_bc, sim_ba = [], []
    for i in ids:
        fb, fc, fa = _full(B[i]), _full(C[i]), _full(A[i])
        sim_bc.append(SequenceMatcher(None, fb, fc).ratio())
        sim_ba.append(SequenceMatcher(None, fb, fa).ratio())
    m_bc, m_ba = float(np.median(sim_bc)), float(np.median(sim_ba))
    fidelity_pass = (m_bc - m_ba) >= args.sim_margin

    print(f"\n=== A2 reconstruction gate [{args.workload}] — {len(ids)} problems ===")
    print("\n-- signal 1: concision (reasoning-token medians) --")
    print(f"   A base            : {a_rt:.0f}")
    print(f"   B base+LoRA@{args.lora_lambda:<4}: {b_rt:.0f}")
    print(f"   C full ThinkingCap: {c_rt:.0f}")
    print(f"   B/C length ratio = {len_ratio:.2f}  (1.0 = exact)  "
          f"-> {'PASS' if concise_pass else 'FAIL'} (tol {args.len_tol})")
    print("\n-- signal 2: output fidelity (difflib ratio, per-problem median) --")
    print(f"   sim(B, C) = {m_bc:.3f}   sim(B, A) = {m_ba:.3f}   gap = {m_bc - m_ba:+.3f}")
    print(f"   -> {'PASS' if fidelity_pass else 'FAIL'} (margin {args.sim_margin})")

    verdict = concise_pass and fidelity_pass
    print(f"\n=== RECONSTRUCTION: {'PASS — LoRA reproduces ThinkingCap; transfer is meaningful' if verdict else 'FAIL — rank-64 SVD does not reconstruct the full FT; concision lives outside it'} ===")
    # A middle case worth naming explicitly: concise but not faithful = the adapter shortens
    # but does not reproduce ThinkingCap's actual reasoning (a different short model).
    if concise_pass and not fidelity_pass:
        print("   NOTE: concise but low-fidelity -- the adapter shortens output without "
              "reproducing ThinkingCap's reasoning. Treat any transfer result with caution.")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
