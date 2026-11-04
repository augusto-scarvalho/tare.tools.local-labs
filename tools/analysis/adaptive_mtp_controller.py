#!/usr/bin/env python3
"""BEE-L3: Adaptive MTP Profit Controller.

Dynamically controls speculative decoding / multi-token prediction depth K
based on moving acceptance rates to maximize throughput during high-predictability
phases and eliminate latency penalties during stochastic reasoning phases.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import random
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class AdaptiveMTPController:
    def __init__(self, window_size: int = 16, gamma: float = 0.15, max_k: int = 4, probe_interval: int = 8):
        self.window_size = window_size
        self.gamma = gamma  # Draft step cost / Target step cost
        self.max_k = max_k
        self.probe_interval = probe_interval
        self.step_count = 0
        self.history: collections.deque[tuple[int, int]] = collections.deque(maxlen=window_size)
        self.current_depth = max_k

    def record_step(self, accepted_tokens: int, drafted_tokens: int) -> None:
        """Records an execution step with number of accepted tokens vs drafted tokens."""
        self.step_count += 1
        if drafted_tokens > 0:
            self.history.append((accepted_tokens, drafted_tokens))
        self._update_depth()

    def get_current_acceptance_rate(self) -> float:
        """Calculates moving acceptance rate over the window."""
        if not self.history:
            return 0.85  # Optimistic initial default
        total_accepted = sum(acc for acc, _ in self.history)
        total_drafted = sum(dft for _, dft in self.history)
        return total_accepted / total_drafted if total_drafted > 0 else 0.0

    def expected_speedup(self, alpha: float, k: int) -> float:
        """Calculates expected speedup under acceptance rate alpha and draft depth k."""
        if k == 0:
            return 1.0
        if math.isclose(alpha, 1.0):
            expected_accepted = 1.0 + k
        else:
            expected_accepted = (1.0 - (alpha ** (k + 1))) / (1.0 - alpha)
        
        step_cost = 1.0 + (k * self.gamma)
        return expected_accepted / step_cost

    def _update_depth(self) -> None:
        alpha = self.get_current_acceptance_rate()
        best_k = 0
        best_speedup = 1.0

        for k in range(1, self.max_k + 1):
            spd = self.expected_speedup(alpha, k)
            if spd > best_speedup * 1.03:
                best_speedup = spd
                best_k = k

        self.current_depth = best_k

    def get_recommended_depth(self) -> int:
        # If currently throttled to k=0, periodically probe with k=1 to detect regime recovery
        if self.current_depth == 0 and (self.step_count % self.probe_interval == 0):
            return 1
        return self.current_depth


def simulate_engine(policy_type: str, regimes: list[tuple[float, int]], gamma: float = 0.15, max_k: int = 4, seed: int = 42) -> dict:
    rng = random.Random(seed)
    controller = AdaptiveMTPController(window_size=16, gamma=gamma, max_k=max_k)

    total_tokens_emitted = 0
    total_time_cost = 0.0
    accepted_history = []
    depth_history = []

    for alpha_true, step_count in regimes:
        for _ in range(step_count):
            if policy_type == "BASELINE_K0":
                k = 0
            elif policy_type == "STATIC_K2":
                k = 2
            elif policy_type == "STATIC_K4":
                k = 4
            elif policy_type == "ADAPTIVE":
                k = controller.get_recommended_depth()
            else:
                raise ValueError(policy_type)

            depth_history.append(k)

            if k == 0:
                # Target emits 1 token at cost 1.0
                tokens_in_step = 1
                cost_in_step = 1.0
                controller.record_step(0, 0)
            else:
                # Draft generates k tokens, target verifies
                # Geometric acceptance: tokens accepted until first mismatch
                accepted = 0
                for _ in range(k):
                    if rng.random() < alpha_true:
                        accepted += 1
                    else:
                        break
                # Always accept at least 1 bonus token from verification step
                tokens_in_step = 1 + accepted
                cost_in_step = 1.0 + (k * gamma)
                controller.record_step(accepted, k)
                accepted_history.append(accepted / k if k > 0 else 0)

            total_tokens_emitted += tokens_in_step
            total_time_cost += cost_in_step

    throughput = total_tokens_emitted / total_time_cost if total_time_cost > 0 else 0.0
    return {
        "policy": policy_type,
        "total_tokens_emitted": total_tokens_emitted,
        "total_time_cost": round(total_time_cost, 2),
        "effective_throughput": round(throughput, 4),
        "mean_depth": round(statistics.mean(depth_history), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="BEE-L3 Adaptive MTP Profit Controller")
    parser.add_argument("--output", default="runs/research/BEE-L3-MTP-CONTROLLER-2026-08-25/raw/receipt.json")
    args = parser.parse_args()

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== BEE-L3 Adaptive MTP Profit Controller Simulation ===", flush=True)

    # Regime pattern: High Predictability (Easy tokens, alpha=0.85) <-> Low Predictability (Hard reasoning, alpha=0.15)
    regimes = [
        (0.85, 200),  # Easy phase 1
        (0.15, 200),  # Hard phase 1 (rollbacks)
        (0.85, 200),  # Easy phase 2
        (0.15, 200),  # Hard phase 2 (rollbacks)
        (0.85, 200),  # Easy phase 3
    ]

    policies = ["BASELINE_K0", "STATIC_K2", "STATIC_K4", "ADAPTIVE"]
    results = {}

    for pol in policies:
        res = simulate_engine(pol, regimes)
        results[pol] = res
        print(f"Policy [{pol:12}]: Throughput = {res['effective_throughput']:.4f} tok/cost | Mean Depth = {res['mean_depth']}")

    base_tp = results["BASELINE_K0"]["effective_throughput"]
    k4_tp = results["STATIC_K4"]["effective_throughput"]
    adaptive_tp = results["ADAPTIVE"]["effective_throughput"]

    adaptive_speedup_over_baseline = (adaptive_tp / base_tp)
    adaptive_gain_over_k4 = ((adaptive_tp - k4_tp) / k4_tp) * 100.0

    # Low predictability stress check (pure alpha=0.15)
    low_regime = [(0.15, 300)]
    low_base = simulate_engine("BASELINE_K0", low_regime)["effective_throughput"]
    low_adaptive = simulate_engine("ADAPTIVE", low_regime)["effective_throughput"]
    low_protection_pct = (low_adaptive / low_base) * 100.0

    gates = {
        "global_speedup_ge_1_25x": adaptive_speedup_over_baseline >= 1.25,
        "gain_over_static_k4_ge_15pct": adaptive_gain_over_k4 >= 15.0,
        "low_predictability_protection_ge_95pct": low_protection_pct >= 95.0,
    }

    verdict = "PROMOTED" if all(gates.values()) else "REJECTED"

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Antigravity",
        "regime_schedule": regimes,
        "policy_results": results,
        "analysis": {
            "adaptive_speedup_over_baseline": round(adaptive_speedup_over_baseline, 3),
            "adaptive_gain_over_static_k4_pct": round(adaptive_gain_over_k4, 2),
            "low_predictability_protection_pct": round(low_protection_pct, 2),
        },
        "gates": gates,
        "verdict": verdict,
    }

    out_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n==================================================", flush=True)
    print(f"  BEE-L3 CONTROLLER VERDICT: {verdict}", flush=True)
    print(f"  Speedup over Baseline:     {adaptive_speedup_over_baseline:.2f}x (Gate >=1.25x: {gates['global_speedup_ge_1_25x']})")
    print(f"  Gain over Static K=4:      +{adaptive_gain_over_k4:.1f}% (Gate >=15%: {gates['gain_over_static_k4_ge_15pct']})")
    print(f"  Low Predictability Guard:  {low_protection_pct:.1f}% (Gate >=95%: {gates['low_predictability_protection_ge_95pct']})")
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if verdict == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
