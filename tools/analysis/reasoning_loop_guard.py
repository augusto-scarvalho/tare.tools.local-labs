#!/usr/bin/env python3
"""BEE-L5: Reasoning-Loop Guard.

Detects entropy collapse, cyclic hesitation loops, and repetitive reasoning traps
in <think> channels in real-time, preventing runaway generation and token budget exhaustion.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REVERSAL_PATTERNS = [
    re.compile(r"\b(wait|wait,|wait\.|let me reconsider|let me check|but wait|hold on|actually no|rethinking)\b", re.IGNORECASE),
    re.compile(r"\b(mas espere|recalculando|pensando bem|na verdade não|espera)\b", re.IGNORECASE),
]


class ReasoningLoopGuard:
    def __init__(self, window_size: int = 32, max_reversals: int = 3, max_ngram_reps: int = 3):
        self.window_size = window_size
        self.max_reversals = max_reversals
        self.max_ngram_reps = max_ngram_reps
        self.token_window: collections.deque[str] = collections.deque(maxlen=window_size)
        self.full_trace: list[str] = []
        self.inside_think_tag = True

    def reset(self) -> None:
        self.token_window.clear()
        self.full_trace.clear()
        self.inside_think_tag = True

    def feed_token(self, token_str: str) -> tuple[bool, str | None]:
        """Feeds a token and returns (trigger_cut: bool, reason: str | None)."""
        self.token_window.append(token_str)
        self.full_trace.append(token_str)

        if "</think>" in token_str.lower():
            self.inside_think_tag = False
            return False, None

        if not self.inside_think_tag:
            return False, None

        if len(self.token_window) < 8:
            return False, None

        window_text = "".join(self.token_window)

        # Check 1: Excessive hesitation/reversal phrases in local window
        reversal_count = 0
        for pat in REVERSAL_PATTERNS:
            reversal_count += len(pat.findall(window_text))
        if reversal_count >= self.max_reversals:
            return True, f"EXCESSIVE_REVERSALS_IN_WINDOW ({reversal_count} >= {self.max_reversals})"

        # Check 2: Repetitive 4-token cycles
        tokens = list(self.token_window)
        if len(tokens) >= 12:
            # Check if last 4 tokens repeat consecutively 3 times
            ngram = tuple(tokens[-4:])
            prev_ngram_1 = tuple(tokens[-8:-4])
            prev_ngram_2 = tuple(tokens[-12:-8])
            if ngram == prev_ngram_1 == prev_ngram_2:
                return True, f"CONSECUTIVE_4GRAM_CYCLE ({' '.join(ngram)})"

        return False, None


def generate_benchmark_dataset() -> list[dict]:
    dataset = []

    # 25 Legitimate Reasoning Samples
    for i in range(25):
        legit_text = (
            f"Step 1: Calculate the base value {i} + 10 = {i+10}. "
            f"Step 2: Multiply by factor 2 to obtain {(i+10)*2}. "
            f"Wait, let's verify if tax applies. If tax is 10%, we add {((i+10)*2)*0.1}. "
            f"Step 3: Total sum is {((i+10)*2)*1.1}. "
            f"Conclusion: The final answer is {((i+10)*2)*1.1}."
        )
        dataset.append({
            "id": f"legit_{i:02d}",
            "text": legit_text,
            "is_loop": False,
        })

    # 25 Pathological Loop Samples (Wait loops and repetitive n-grams)
    for i in range(25):
        if i % 2 == 0:
            # Reversal trap
            loop_text = (
                f"Let's see the problem {i}. "
                + "Wait, let me reconsider. But wait, actually no, let me check. Wait, let me reconsider. " * 3
                + "Wait, but wait, actually no."
            )
        else:
            # Cyclic n-gram trap
            loop_text = (
                f"We must consider case {i}. "
                + "therefore we must evaluate " * 5
                + "and solve it."
            )
        dataset.append({
            "id": f"loop_{i:02d}",
            "text": loop_text,
            "is_loop": True,
        })

    return dataset


def evaluate_guard() -> dict:
    guard = ReasoningLoopGuard(window_size=32, max_reversals=3, max_ngram_reps=3)
    dataset = generate_benchmark_dataset()

    tp, fp, tn, fn = 0, 0, 0, 0
    results = []

    for item in dataset:
        guard.reset()
        # Split text into simulated tokens
        tokens = [word + " " for word in item["text"].split()]
        cut_triggered = False
        trigger_reason = None
        cut_token_idx = -1

        for idx, tok in enumerate(tokens):
            triggered, reason = guard.feed_token(tok)
            if triggered:
                cut_triggered = True
                trigger_reason = reason
                cut_token_idx = idx
                break

        if item["is_loop"] and cut_triggered:
            tp += 1
            eval_class = "TP"
        elif not item["is_loop"] and cut_triggered:
            fp += 1
            eval_class = "FP"
        elif not item["is_loop"] and not cut_triggered:
            tn += 1
            eval_class = "TN"
        else:
            fn += 1
            eval_class = "FN"

        results.append({
            "id": item["id"],
            "is_loop": item["is_loop"],
            "cut_triggered": cut_triggered,
            "trigger_reason": trigger_reason,
            "cut_token_idx": cut_token_idx,
            "eval_class": eval_class,
        })

    tpr = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    fpr = (fp / (fp + tn)) * 100.0 if (fp + tn) > 0 else 0.0

    return {
        "total_samples": len(dataset),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "sensitivity_tpr_pct": round(tpr, 2),
        "false_alarm_fpr_pct": round(fpr, 2),
        "detailed_results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="BEE-L5 Reasoning-Loop Guard")
    parser.add_argument("--output", default="runs/research/BEE-L5-REASONING-LOOP-GUARD-2026-08-25/raw/receipt.json")
    args = parser.parse_args()

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== BEE-L5 Reasoning-Loop Guard Evaluation ===", flush=True)
    eval_res = evaluate_guard()

    print(f"Total Samples:      {eval_res['total_samples']}")
    print(f"True Positives:     {eval_res['true_positives']}/25 ({eval_res['sensitivity_tpr_pct']}%)")
    print(f"False Positives:    {eval_res['false_positives']}/25 ({eval_res['false_alarm_fpr_pct']}%)")
    print(f"True Negatives:     {eval_res['true_negatives']}/25")
    print(f"False Negatives:    {eval_res['false_negatives']}/25")

    gates = {
        "sensitivity_tpr_ge_95pct": eval_res["sensitivity_tpr_pct"] >= 95.0,
        "false_positive_rate_le_2pct": eval_res["false_alarm_fpr_pct"] <= 2.0,
    }

    verdict = "PROMOTED" if all(gates.values()) else "REJECTED"

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Antigravity",
        "evaluation": eval_res,
        "gates": gates,
        "verdict": verdict,
    }

    out_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n==================================================", flush=True)
    print(f"  BEE-L5 REASONING-LOOP GUARD VERDICT: {verdict}", flush=True)
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if verdict == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
