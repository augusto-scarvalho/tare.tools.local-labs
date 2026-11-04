#!/usr/bin/env python3
"""SPEC-01: Speculative Evolution Pipeline Probe.

Evaluates the speedup and acceptance rates of the Hybrid Speculative Engine
(N-Gram Trie + MTP Neural Proposer) across 50 structured syntax workloads.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.hybrid_speculative_engine import HybridSpeculativeEngine


def run_benchmark(num_trials: int = 50) -> dict:
    random.seed(20260824)

    # Patterns mimicking structured generation (JSON + Code + XML reasoning tags)
    motifs = [
        [100, 101, 102, 103, 104, 105],  # `<think>\nLet'`
        [200, 201, 202, 203, 204],        # `{"status": "ok",`
        [300, 301, 302, 303, 304, 305],  # `def forward(self, x):`
        [400, 401, 402, 403],             # `\n    return out\n`
    ]

    total_autoregressive_steps = 0
    total_hybrid_steps = 0
    total_hybrid_ngram_drafts = 0
    total_hybrid_mtp_drafts = 0
    total_tokens_generated = 0

    for trial in range(num_trials):
        # Build ground truth sequence with structured repeating elements
        ground_truth = []
        for _ in range(15):
            ground_truth.extend(random.choice(motifs))
            ground_truth.extend([random.randint(1000, 1100) for _ in range(2)])
        ground_truth.append(999)  # EOS

        prompt = ground_truth[:10]

        def verifier(ctx: list[int], draft: list[int]) -> tuple[int, int]:
            curr = len(ctx)
            acc = 0
            for i, t in enumerate(draft):
                if curr + i < len(ground_truth) and t == ground_truth[curr + i]:
                    acc += 1
                else:
                    break
            nxt = curr + acc
            return acc, ground_truth[nxt] if nxt < len(ground_truth) else 999

        def mtp_proposer(ctx: list[int], max_draft: int) -> list[int]:
            curr = len(ctx)
            # Simulated MTP: 75% accuracy
            draft = []
            for i in range(max_draft):
                if curr + i < len(ground_truth):
                    if random.random() < 0.75:
                        draft.append(ground_truth[curr + i])
                    else:
                        draft.append(random.randint(5000, 6000))
            return draft

        engine = HybridSpeculativeEngine(
            target_verifier=verifier,
            mtp_proposer=mtp_proposer,
            max_draft_len=4,
        )

        res = engine.generate(prompt_tokens=prompt, max_new_tokens=100, eos_token_id=999)

        gen_cnt = res["generated_count"]
        total_tokens_generated += gen_cnt
        total_autoregressive_steps += gen_cnt  # 1 step per token in autoregressive baseline
        total_hybrid_steps += res["target_verification_steps"]
        total_hybrid_ngram_drafts += res["ngram_drafts"]
        total_hybrid_mtp_drafts += res["mtp_drafts"]

    overall_speedup = total_tokens_generated / total_hybrid_steps
    mean_accepted = total_tokens_generated / total_hybrid_steps
    ngram_share = (total_hybrid_ngram_drafts / (total_hybrid_ngram_drafts + total_hybrid_mtp_drafts)) * 100.0

    return {
        "trials": num_trials,
        "total_tokens_generated": total_tokens_generated,
        "autoregressive_steps": total_autoregressive_steps,
        "hybrid_speculative_steps": total_hybrid_steps,
        "mean_accepted_tokens_per_step": round(mean_accepted, 2),
        "speculative_speedup_factor": round(overall_speedup, 2),
        "ngram_drafts": total_hybrid_ngram_drafts,
        "mtp_drafts": total_hybrid_mtp_drafts,
        "ngram_draft_share_pct": round(ngram_share, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SPEC-01 Speculative Evolution Probe")
    parser.add_argument("--output", default="runs/research/SPEC-01-SPECULATIVE-PIPELINE-2026-08-25/raw/receipt.json")
    args = parser.parse_args()

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== SPEC-01 Speculative Evolution Pipeline Probe ===", flush=True)
    res = run_benchmark(num_trials=50)

    print(f"Total Tokens Generated:       {res['total_tokens_generated']}")
    print(f"Autoregressive Steps:         {res['autoregressive_steps']}")
    print(f"Hybrid Speculative Steps:     {res['hybrid_speculative_steps']}")
    print(f"Mean Tokens Accepted / Step:  {res['mean_accepted_tokens_per_step']} tokens/step")
    print(f"Effective Speedup Factor:     {res['speculative_speedup_factor']}×")
    print(f"N-Gram Trie Draft Share:      {res['ngram_draft_share_pct']}%")

    gates = {
        "speedup_ge_1_80x": res["speculative_speedup_factor"] >= 1.80,
        "mean_accepted_ge_2_0": res["mean_accepted_tokens_per_step"] >= 2.0,
        "ngram_active_ge_20pct": res["ngram_draft_share_pct"] >= 20.0,
    }

    verdict = "PROMOTED" if all(gates.values()) else "REJECTED"

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Antigravity",
        "results": res,
        "gates": gates,
        "verdict": verdict,
    }

    out_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n==================================================", flush=True)
    print(f"  SPEC-01 SPECULATIVE PIPELINE VERDICT: {verdict}", flush=True)
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if verdict == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
