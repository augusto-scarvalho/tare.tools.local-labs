"""Agent-shaped workload: is the fork worth it for coding agents and research?

`SCENARIOS.md` translated the A/B into agent terms and, in doing so, made two claims that
were ARITHMETIC rather than measurement. This script measures both, because a document
that mixes measured and computed numbers without saying which is which is how this
project has already had to withdraw findings twice.

CLAIM 1 (computed): KV cache costs ~21.2 KiB/token, so ~64k context still respects the
4 GB VRAM reserve at ncmoe=8. Never measured. Phase A loads at each context size and
reads real VRAM -- no requests needed, so it is nearly free.

CLAIM 2 (arithmetic over a fixed prefill rate): an agent turn with a warm prompt cache
gains ~17%, a cold one ~35%. That treated cache hit and cache miss as if they only
differed in token count. Phase B measures them directly, and `cache_n` from the server's
timings reports exactly how many tokens were actually reused -- so a "warm" request that
silently missed cannot be reported as a hit.

Why this matters more than the headline A/B: the earlier runs sent one fixed prompt with
a unique per-repetition prefix, which DEFEATS the prompt cache by construction. Every
number so far is therefore a cache-MISS number. Real agents hit the cache most turns.

    python agent_bench.py
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from model_lifecycle.analysis.statistics import describe        # noqa: E402
from model_lifecycle.collectors.host import sample              # noqa: E402
from model_lifecycle.collectors.request import chat_stream      # noqa: E402
from model_lifecycle.servers.llama_cpp import (                 # noqa: E402
    LlamaCppAdapter, ServerProfile)

MODEL = "/home/augus/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
BASE_BIN = "/home/augus/src/slop.cpp-base/build/bin/llama-server"
FORK_BIN = "/home/augus/src/slop.cpp/build/bin/llama-server"
MASTER_BIN = "/home/augus/src/slop.cpp-main/build/bin/llama-server"

ARMS = {
    "base": (BASE_BIN, {}),
    # CORRECTED after the isolation run. The first version of this file ran prefetch with
    # pinning OFF, on the assumption that the two switches were independent and that the
    # cheap one was worth measuring on its own. They are NOT independent: measured over
    # four balanced rounds, prefetch WITHOUT pinning is -26% on prefill -- worse than
    # plain upstream -- because cudaMemcpyAsync out of pageable memory is not actually
    # asynchronous. The driver stages it through an internal pinned buffer, so the copy
    # serialises and the overlap never happens.
    #
    # So there is no cheap arm. This is the only configuration of the fork worth
    # measuring, and its price is 12.3 GB of non-pageable host memory.
    "fork": (FORK_BIN, {"GGML_SCHED_PREFETCH_EXPERTS": "3",
                        "GGML_CUDA_REGISTER_HOST": "1"}),
    # Today's upstream, 266 commits past the base. The env is inert here (no such getenv
    # in stock llama.cpp) and is passed only to keep the command lines identical.
    "master": (MASTER_BIN, {"GGML_SCHED_PREFETCH_EXPERTS": "3",
                            "GGML_CUDA_REGISTER_HOST": "1"}),
}

NCMOE = 8          # the config that fits this desktop; established by the earlier sweep


def _filler(approx_tokens: int) -> str:
    """Prose of roughly the requested token count. Prose, not `"x" * n`: a repeated
    character tokenises at a wildly different ratio and the prompt would not be the size
    it claims. ~4 chars/token is the usual English ratio; the real count is read back
    from the server's `prompt_n` rather than trusted."""
    unit = ("The scheduler assigns each operation to the backend that owns its weights, "
            "so a tensor living in system memory pulls its computation onto the host "
            "unless the graph explicitly uploads it first. ")
    return unit * max(1, int(approx_tokens * 4 / len(unit)))


def phase_a() -> list[dict]:
    """VRAM against context size. Load, measure, stop -- no requests, so it costs a
    minute and settles a table that was pure arithmetic."""
    print("\n=== PHASE A: does context cost what the arithmetic said? ===", flush=True)
    rows = []
    for ctx in (8192, 32768, 65536, 131072):
        adapter = LlamaCppAdapter(server_bin=BASE_BIN)
        profile = ServerProfile(model_path=MODEL, port=8080, n_cpu_moe=NCMOE,
                                ctx_size=ctx, cache_type_k="q8_0", cache_type_v="q8_0")
        h = adapter.start(profile)
        try:
            ok = adapter.wait_until_healthy(h, timeout_s=300)
            s = sample()
            free_gb = s.vram_free_mb / 1024
            rows.append({"ctx": ctx, "healthy": ok, "vram_free_mb": s.vram_free_mb,
                         "vram_used_mb": s.vram_used_mb})
            print(f"  ctx {ctx:>7}: {'OK ' if ok else 'FAIL'} "
                  f"vram_free={s.vram_free_mb:>6} MB ({free_gb:.2f} GB)"
                  f"{'   <- inside the 4 GB reserve' if free_gb < 4 else ''}", flush=True)
        finally:
            adapter.stop(h)
            adapter.force_stop(h)
            time.sleep(5)
    return rows


def phase_b(ctx: int = 32768, prompt_tokens: int = 20000, reps: int = 3) -> list[dict]:
    """Cold cache versus warm cache, both arms.

    COLD: every request carries a different prefix, so the server must prefill all of it.
    WARM: one priming request, then follow-ups that only APPEND -- which is what an agent
    turn actually looks like. `cache_n` proves which happened instead of assuming it.
    """
    print(f"\n=== PHASE B: agent turns at ctx={ctx}, ~{prompt_tokens} token prompt ===",
          flush=True)
    body = _filler(prompt_tokens)
    rows = []
    for arm, (binary, env) in ARMS.items():
        adapter = LlamaCppAdapter(server_bin=binary, env=env)
        profile = ServerProfile(model_path=MODEL, port=8080, n_cpu_moe=NCMOE,
                                ctx_size=ctx, cache_type_k="q8_0", cache_type_v="q8_0")
        h = adapter.start(profile)
        try:
            if not adapter.wait_until_healthy(h, timeout_s=600):
                print(f"  {arm}: server never healthy"); continue

            # max_tokens=8 on purpose: this phase measures READING, and generation was
            # already measured to be unaffected. Short answers keep the run to minutes.
            cold, warm, cold_cached, warm_cached = [], [], [], []

            for i in range(reps):
                r = chat_stream(h.base_url, f"[doc {i}] {body}\n\nSummarise in one word.",
                                max_tokens=8)
                if r.prompt_tps:
                    cold.append(r.prompt_tps)
                    cold_cached.append(r.cache_n or 0)
                    print(f"  {arm:<9} cold  n={r.prompt_n:<6} cached={r.cache_n:<6} "
                          f"{r.prompt_tps:>8.1f} t/s", flush=True)

            # Prime once, then append -- the shape of a real agent turn.
            chat_stream(h.base_url, f"{body}\n\nSummarise in one word.", max_tokens=8)
            for i in range(reps):
                r = chat_stream(h.base_url,
                                f"{body}\n\nSummarise in one word.\n\nFollow-up {i}: "
                                f"name one bottleneck.", max_tokens=8)
                if r.prompt_tps:
                    warm.append(r.prompt_tps)
                    warm_cached.append(r.cache_n or 0)
                    print(f"  {arm:<9} warm  n={r.prompt_n:<6} cached={r.cache_n:<6} "
                          f"{r.prompt_tps:>8.1f} t/s", flush=True)

            rows.append({"arm": arm, "ctx": ctx,
                         "cold_tps": describe(cold).as_dict() if cold else None,
                         "warm_tps": describe(warm).as_dict() if warm else None,
                         "cold_cached_mean": sum(cold_cached)/len(cold_cached) if cold_cached else None,
                         "warm_cached_mean": sum(warm_cached)/len(warm_cached) if warm_cached else None})
        finally:
            adapter.stop(h)
            adapter.force_stop(h)
            time.sleep(10)
    return rows


def main() -> int:
    out = pathlib.Path(__file__).parent / "runs"
    out.mkdir(exist_ok=True)
    a = phase_a()
    b = phase_b()
    (out / "agent_bench.json").write_text(
        json.dumps({"phase_a": a, "phase_b": b}, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("PHASE A -- context cost, MEASURED")
    print("=" * 70)
    base = next((r for r in a if r["ctx"] == 8192), None)
    for r in a:
        delta = (r["vram_used_mb"] - base["vram_used_mb"]) if base else 0
        print(f"  ctx {r['ctx']:>7}: vram_used={r['vram_used_mb']:>6} MB "
              f"(+{delta:>5} MB vs 8k)  free={r['vram_free_mb']/1024:.2f} GB")

    print("\n" + "=" * 70)
    print("PHASE B -- prefill, cold cache vs warm cache")
    print("=" * 70)
    by_arm = {r["arm"]: r for r in b}
    for mode in ("cold", "warm"):
        print(f"\n  -- {mode} cache --")
        for arm in ARMS:
            r = by_arm.get(arm)
            d = r and r[f"{mode}_tps"]
            if not d:
                print(f"    {arm:<9} no data"); continue
            print(f"    {arm:<9} {d['mean']:>8.1f} t/s  (cv {d['cv']:.2f}, "
                  f"mean cached tokens {r[f'{mode}_cached_mean']:.0f})")
        b = by_arm.get("base", {}).get(f"{mode}_tps")
        for arm in ARMS:
            if arm == "base" or not b:
                continue
            other = by_arm.get(arm, {}).get(f"{mode}_tps")
            if not other:
                continue
            d = other["mean"] - b["mean"]
            print(f"    {arm + ' vs base':<18} {d:>+8.1f} t/s  ({d / b['mean'] * 100:+.1f}%)")
    print("\nA warm request whose `cached` count is ~0 did NOT hit the cache; treat its "
          "row as a second cold measurement rather than as evidence about warm turns.")
    return 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        f = _filler(1000)
        assert 3200 <= len(f) <= 4800, f"filler is {len(f)} chars, wanted ~4000"
        assert f.count("scheduler") > 10, "filler must be prose, not one repeated token"
        print(f"filler: {len(f)} chars for ~1000 tokens; self-check OK")
        raise SystemExit(0)
    raise SystemExit(main())
