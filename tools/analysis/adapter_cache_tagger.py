#!/usr/bin/env python3
"""ADAPT-06: Adapter-Aware KV Cache & Tagging Controller.

Provides 64-bit cryptographic composite cache keys:
  Key = Hash64(model_id || adapter_id || token_block)
guaranteeing strict isolation across multi-tenant adapters in speculative / prompt caching runtimes.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import struct
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class AdapterCacheTagger:
    def __init__(self, block_size: int = 16, max_blocks: int = 256):
        self.block_size = block_size
        self.max_blocks = max_blocks
        # Maps 64-bit hash -> (block_id, adapter_id, token_ids, last_access)
        self.cache_table: dict[int, dict[str, Any]] = {}
        self.next_block_id = 1
        self.stats = {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "cross_adapter_collisions_detected": 0,
        }

    @staticmethod
    def compute_block_key(model_id: str, adapter_id: str | None, tokens: tuple[int, ...]) -> int:
        hasher = hashlib.sha256()
        hasher.update(model_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update((adapter_id or "NONE").encode("utf-8"))
        hasher.update(b"\x00")
        # Pack token integers
        for t in tokens:
            hasher.update(struct.pack("<I", t & 0xFFFFFFFF))
        digest = hasher.digest()
        # Extract 64-bit unsigned int
        return struct.unpack("<Q", digest[:8])[0]

    def lookup_block(self, model_id: str, adapter_id: str | None, tokens: tuple[int, ...]) -> tuple[bool, int]:
        self.stats["lookups"] += 1
        key = self.compute_block_key(model_id, adapter_id, tokens)

        if key in self.cache_table:
            entry = self.cache_table[key]
            # Verify integrity
            if entry["adapter_id"] != adapter_id:
                self.stats["cross_adapter_collisions_detected"] += 1
                raise RuntimeError(
                    f"CRITICAL: Hash collision or unauthorized cross-adapter access! "
                    f"Cached: {entry['adapter_id']}, Requested: {adapter_id}"
                )
            entry["last_access"] = time.monotonic()
            self.stats["hits"] += 1
            return True, entry["block_id"]

        # Cache Miss: Allocate new block
        self.stats["misses"] += 1
        if len(self.cache_table) >= self.max_blocks:
            self._evict_lru()

        block_id = self.next_block_id
        self.next_block_id += 1
        self.cache_table[key] = {
            "block_id": block_id,
            "adapter_id": adapter_id,
            "tokens": tokens,
            "last_access": time.monotonic(),
        }
        return False, block_id

    def _evict_lru(self) -> None:
        oldest_key = min(self.cache_table.keys(), key=lambda k: self.cache_table[k]["last_access"])
        del self.cache_table[oldest_key]
        self.stats["evictions"] += 1


def run_multi_tenant_simulation(num_requests: int = 100, seed: int = 42) -> dict:
    rng = random.Random(seed)
    tagger = AdapterCacheTagger(block_size=16, max_blocks=64)

    # Common system prompt prefix shared by all requests
    common_prefix = list(range(1000, 1032))  # 32 tokens = 2 blocks

    adapters = ["lokr_math_r8", "lokr_code_r8", "base_backbone"]
    request_logs = []

    for req_id in range(num_requests):
        chosen_adapter = rng.choice(adapters)
        # Random unique suffix (16 tokens = 1 block)
        task_suffix = [rng.randint(2000, 2500) for _ in range(16)]
        full_prompt = common_prefix + task_suffix

        # Break into 16-token blocks
        blocks = [tuple(full_prompt[i:i + 16]) for i in range(0, len(full_prompt), 16)]
        block_hits = []

        for b in blocks:
            hit, block_id = tagger.lookup_block("qwen3.5-0.8b", chosen_adapter, b)
            block_hits.append(hit)

        request_logs.append({
            "req_id": req_id,
            "adapter": chosen_adapter,
            "block_hits": block_hits,
            "prefix_hit": all(block_hits[:2]),  # The 2 shared blocks
        })

    # Prefix hit rate when the same adapter repeats
    prefix_hits = sum(1 for r in request_logs if r["prefix_hit"])
    intra_adapter_hit_rate = (prefix_hits / num_requests) * 100.0

    return {
        "num_requests": num_requests,
        "stats": tagger.stats,
        "prefix_hit_rate_pct": round(intra_adapter_hit_rate, 2),
        "total_active_blocks": len(tagger.cache_table),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ADAPT-06 Adapter-Aware KV Cache Tagger")
    parser.add_argument("--output", default="runs/research/ADAPT-06-ADAPTER-CACHE-TAGGING-2026-08-25/raw/receipt.json")
    args = parser.parse_args()

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== ADAPT-06 Adapter-Aware KV Cache Tagger Simulation ===", flush=True)
    sim_results = run_multi_tenant_simulation(num_requests=120)

    print(f"Total Lookups: {sim_results['stats']['lookups']}")
    print(f"Cache Hits:    {sim_results['stats']['hits']}")
    print(f"Cache Misses:  {sim_results['stats']['misses']}")
    print(f"Prefix Hit Rate: {sim_results['prefix_hit_rate_pct']}%")
    print(f"Cross-Adapter Collisions Detected: {sim_results['stats']['cross_adapter_collisions_detected']}")

    gates = {
        "zero_cross_adapter_collisions": sim_results["stats"]["cross_adapter_collisions_detected"] == 0,
        "prefix_hit_rate_ge_75pct": sim_results["prefix_hit_rate_pct"] >= 75.0,
    }

    verdict = "PROMOTED" if all(gates.values()) else "REJECTED"

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Antigravity",
        "simulation_results": sim_results,
        "gates": gates,
        "verdict": verdict,
    }

    out_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n==================================================", flush=True)
    print(f"  ADAPT-06 TAGGER VERDICT: {verdict}", flush=True)
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if verdict == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
