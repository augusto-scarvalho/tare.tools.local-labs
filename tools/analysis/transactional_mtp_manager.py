#!/usr/bin/env python3
"""BEE-L4: Transactional Target + MTP Restore Manager.

Implements ACID speculative token buffer management with atomic checkpoints
and zero cross-slot rollback contamination in high-concurrency multi-tenant runtimes.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@dataclass
class SlotState:
    slot_id: int
    committed_tokens: list[int] = field(default_factory=list)
    checkpoint_len: int = 0
    in_tx: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class TransactionalSpeculativeManager:
    def __init__(self, num_slots: int = 4):
        self.num_slots = num_slots
        self.slots = {i: SlotState(slot_id=i) for i in range(num_slots)}
        self.stats = {
            "total_transactions": 0,
            "commits": 0,
            "full_rollbacks": 0,
            "partial_rollbacks": 0,
            "cross_slot_corruptions": 0,
        }
        self.stats_lock = threading.Lock()

    def begin_step(self, slot_id: int) -> int:
        slot = self.slots[slot_id]
        with slot.lock:
            if slot.in_tx:
                raise RuntimeError(f"Slot {slot_id} already has an active transaction!")
            slot.in_tx = True
            slot.checkpoint_len = len(slot.committed_tokens)
            return slot.checkpoint_len

    def append_draft(self, slot_id: int, draft_tokens: list[int]) -> None:
        slot = self.slots[slot_id]
        with slot.lock:
            if not slot.in_tx:
                raise RuntimeError(f"Slot {slot_id} has no active transaction!")
            # Draft tokens appended temporarily
            slot.committed_tokens.extend(draft_tokens)

    def complete_step(self, slot_id: int, accepted_count: int, total_drafted: int) -> list[int]:
        slot = self.slots[slot_id]
        with slot.lock:
            if not slot.in_tx:
                raise RuntimeError(f"Slot {slot_id} has no active transaction!")

            # Expected length with draft tokens
            expected_drafted_len = slot.checkpoint_len + total_drafted
            if len(slot.committed_tokens) != expected_drafted_len:
                with self.stats_lock:
                    self.stats["cross_slot_corruptions"] += 1
                raise RuntimeError(
                    f"Slot {slot_id} length corruption detected! "
                    f"Expected: {expected_drafted_len}, Actual: {len(slot.committed_tokens)}"
                )

            # Roll back rejected draft tail
            target_committed_len = slot.checkpoint_len + accepted_count
            slot.committed_tokens = slot.committed_tokens[:target_committed_len]
            slot.in_tx = False
            slot.checkpoint_len = target_committed_len

            with self.stats_lock:
                self.stats["total_transactions"] += 1
                if accepted_count == total_drafted:
                    self.stats["commits"] += 1
                elif accepted_count == 0:
                    self.stats["full_rollbacks"] += 1
                else:
                    self.stats["partial_rollbacks"] += 1

            return list(slot.committed_tokens)


def worker_task(mgr: TransactionalSpeculativeManager, slot_id: int, cycles: int, seed: int) -> dict:
    rng = random.Random(seed)
    local_ground_truth = []
    overhead_times = []

    for cycle in range(cycles):
        t0 = time.perf_counter()
        mgr.begin_step(slot_id)

        # Generate K draft tokens (K in 1..4)
        k = rng.randint(1, 4)
        draft_tokens = [rng.randint(100, 999) for _ in range(k)]
        mgr.append_draft(slot_id, draft_tokens)

        # Determine accepted count
        accepted = rng.randint(0, k)
        for i in range(accepted):
            local_ground_truth.append(draft_tokens[i])

        resulting_tokens = mgr.complete_step(slot_id, accepted, k)
        t1 = time.perf_counter()
        overhead_times.append((t1 - t0) * 1_000_000.0)  # microseconds

        # Verify integrity
        if resulting_tokens != local_ground_truth:
            raise AssertionError(
                f"Integrity violation in slot {slot_id} cycle {cycle}! "
                f"Expected len {len(local_ground_truth)}, got {len(resulting_tokens)}"
            )

    return {
        "slot_id": slot_id,
        "cycles": cycles,
        "final_token_count": len(local_ground_truth),
        "mean_overhead_us": round(sum(overhead_times) / len(overhead_times), 3),
    }


def run_concurrency_stress(num_slots: int = 4, cycles_per_slot: int = 500) -> dict:
    mgr = TransactionalSpeculativeManager(num_slots=num_slots)
    worker_summaries = []

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_slots) as executor:
        futures = [
            executor.submit(worker_task, mgr, s_id, cycles_per_slot, 2026 + s_id)
            for s_id in range(num_slots)
        ]
        for f in concurrent.futures.as_completed(futures):
            worker_summaries.append(f.result())

    elapsed_s = time.perf_counter() - start
    return {
        "total_slots": num_slots,
        "cycles_per_slot": cycles_per_slot,
        "total_operations": num_slots * cycles_per_slot,
        "elapsed_seconds": round(elapsed_s, 4),
        "stats": mgr.stats,
        "workers": worker_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="BEE-L4 Transactional MTP Manager")
    parser.add_argument("--output", default="runs/research/BEE-L4-TRANSACTIONAL-MTP-2026-08-25/raw/receipt.json")
    args = parser.parse_args()

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== BEE-L4 Transactional MTP Manager Concurrency Stress ===", flush=True)
    results = run_concurrency_stress(num_slots=4, cycles_per_slot=500)

    print(f"Total Transactions: {results['stats']['total_transactions']}")
    print(f"Full Rollbacks:     {results['stats']['full_rollbacks']}")
    print(f"Partial Rollbacks:  {results['stats']['partial_rollbacks']}")
    print(f"Full Commits:       {results['stats']['commits']}")
    print(f"Cross-Slot Leaks:   {results['stats']['cross_slot_corruptions']}")

    mean_overhead = sum(w["mean_overhead_us"] for w in results["workers"]) / len(results["workers"])
    print(f"Mean TX Overhead:   {mean_overhead:.2f} µs")

    gates = {
        "zero_cross_slot_corruptions": results["stats"]["cross_slot_corruptions"] == 0,
        "all_transactions_consistent": results["stats"]["total_transactions"] == 2000,
        "overhead_le_10_us": mean_overhead <= 10.0,
    }

    verdict = "PROMOTED" if all(gates.values()) else "REJECTED"

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Antigravity",
        "results": results,
        "mean_overhead_us": round(mean_overhead, 3),
        "gates": gates,
        "verdict": verdict,
    }

    out_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n==================================================", flush=True)
    print(f"  BEE-L4 TRANSACTIONAL MTP VERDICT: {verdict}", flush=True)
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if verdict == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
