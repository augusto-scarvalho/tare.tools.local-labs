"""L18 orthogonal screen — six knobs in 18 runs instead of 486.

WHY NOT THE SWEEPS. `prefill_sweep.py` is one-factor-at-a-time. OFAT has two problems
that matter here:

  * it cannot see interactions, and the main one is almost certainly real -- the prefetch
    hides uploads behind compute, so a bigger micro-batch gives it more to hide behind.
    Measuring `ubatch` at fixed prefetch and prefetch at fixed `ubatch` cannot show that;
  * it wastes runs. Six factors at the levels below is 2 x 3^5 = **486** full-factorial
    configurations. At ~2 minutes each that is 16 hours per replicate.

An L18 orthogonal array estimates all six MAIN effects in **18 runs**, because every
level of every factor is tested against a balanced mix of every other factor's levels.
That balance is the whole trick: the average over the runs where A=1 has the same
distribution of B, C, D... as the average over A=2, so their difference is attributable
to A.

WHAT IT CANNOT DO, stated up front because a screening design oversold is a trap. L18
confounds two-factor interactions into the residual. It says WHICH knobs matter and
roughly how much; it does not give a confidence interval on any single comparison, and
it cannot resolve an interaction. So this is stage one of two:

    stage 1 (this)   L18 screen -> rank the knobs, find the promising corner
    stage 2          ab_isolate.py on the two or three settings that survived, which
                     pairs, interleaves, flips order, and reports CI95

Reporting a Taguchi marginal mean as if it were a measured difference would repeat the
exact error this project already withdrew three times today: a number stated with more
confidence than its design supports.

RESPONSES. Prefill t/s, generation t/s and minimum free VRAM, analysed separately. One
composite score would hide that the knobs pull in different directions -- the prefetch
buys prefill and costs VRAM, and averaging those is how a configuration that kills the
desktop wins a benchmark.

Also reported per factor: the larger-is-better S/N ratio, -10*log10(mean(1/y^2)). It
penalises settings that are fast on average but occasionally bad, which is the property
this platform cares about and the mean alone hides.

    python taguchi_screen.py --replicates 2
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from model_lifecycle.collectors.host import sample              # noqa: E402
from model_lifecycle.collectors.request import chat_stream      # noqa: E402
from model_lifecycle.servers.llama_cpp import (                 # noqa: E402
    LlamaCppAdapter, ServerProfile)

MODEL = "/home/augus/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
# The full stack: 34 commits ahead of upstream, only 16 behind, and every feature below
# is a switch on THIS ONE binary. That is what makes an orthogonal array possible at all
# -- comparing branches instead would change dozens of things per "level".
STACK_BIN = "/home/augus/src/llama.cpp-stack/build/bin/llama-server"

# Standard L18 (2^1 x 3^7). Columns 1..8; we use 1-6 and leave 7-8 empty as an error
# estimate -- their apparent "effect" is pure noise and calibrates how big a real one
# must be.
L18 = [
    (1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 2, 2, 2, 2, 2, 2), (1, 1, 3, 3, 3, 3, 3, 3),
    (1, 2, 1, 1, 2, 2, 3, 3), (1, 2, 2, 2, 3, 3, 1, 1), (1, 2, 3, 3, 1, 1, 2, 2),
    (1, 3, 1, 2, 1, 3, 2, 3), (1, 3, 2, 3, 2, 1, 3, 1), (1, 3, 3, 1, 3, 2, 1, 2),
    (2, 1, 1, 3, 3, 2, 2, 1), (2, 1, 2, 1, 1, 3, 3, 2), (2, 1, 3, 2, 2, 1, 1, 3),
    (2, 2, 1, 2, 3, 1, 3, 2), (2, 2, 2, 3, 1, 2, 1, 3), (2, 2, 3, 1, 2, 3, 2, 1),
    (2, 3, 1, 3, 2, 3, 1, 2), (2, 3, 2, 1, 3, 1, 2, 3), (2, 3, 3, 2, 1, 2, 3, 1),
]

# factor -> (column index, level values). Column 0 is the only 2-level column.
FACTORS = {
    "pin":     (0, ["0", "1"]),                 # GGML_CUDA_REGISTER_HOST
    "prefetch": (1, ["0", "3", "8"]),           # GGML_SCHED_PREFETCH_EXPERTS
    "ubatch":  (2, [512, 1024, 2048]),
    "ncmoe":   (3, [8, 24, 40]),
    "kv":      (4, ["q8_0", "turbo3", "turbo4"]),
    "cache":   (5, [0, 4, 16]),                 # --moe-cache-slots
}

PROMPT_TOKENS = 16000


def _filler(n: int) -> str:
    unit = ("The scheduler assigns each operation to the backend that owns its weights, "
            "so a tensor living in system memory pulls its computation onto the host "
            "unless the graph explicitly uploads it first. ")
    return unit * max(1, int(n * 4 / len(unit)))


def _settings(row: tuple) -> dict:
    return {name: levels[row[col] - 1] for name, (col, levels) in FACTORS.items()}


def run_one(s: dict, tag: str) -> dict | None:
    env = {"GGML_CUDA_REGISTER_HOST": s["pin"],
           "GGML_SCHED_PREFETCH_EXPERTS": s["prefetch"]}
    extra = ("--moe-cache-slots", str(s["cache"])) if s["cache"] else ()
    adapter = LlamaCppAdapter(server_bin=STACK_BIN, env=env)
    profile = ServerProfile(model_path=MODEL, port=8080, n_cpu_moe=s["ncmoe"],
                            ctx_size=32768, batch=2048, ubatch=s["ubatch"],
                            cache_type_k=s["kv"], cache_type_v=s["kv"],
                            extra_args=extra)
    h = adapter.start(profile)
    try:
        if not adapter.wait_until_healthy(h, timeout_s=900):
            print(f"  {tag} UNHEALTHY {s}", flush=True)
            return None
        body = _filler(PROMPT_TOKENS)
        # One prefill measurement and one generation measurement, deliberately separate:
        # they respond to different knobs and a combined number would average them.
        pf = chat_stream(h.base_url, f"[{tag}] {body}\n\nOne word:", max_tokens=8)
        gen = chat_stream(h.base_url, f"[{tag}-g] Write a short paragraph about caching.",
                          max_tokens=400)
        st = sample()
        if not pf.prompt_tps or not gen.generation_tps:
            return None
        return {**s, "prefill_tps": pf.prompt_tps, "gen_tps": gen.generation_tps,
                "vram_free_mb": st.vram_free_mb, "prompt_n": pf.prompt_n}
    finally:
        adapter.stop(h)
        adapter.force_stop(h)
        time.sleep(8)


def sn_larger_is_better(vals: list[float]) -> float:
    """Taguchi S/N, larger-is-better. Penalises a setting that is fast on average but
    occasionally terrible -- the mean alone would call that a winner."""
    vals = [v for v in vals if v and v > 0]
    if not vals:
        return float("nan")
    return -10.0 * math.log10(sum(1.0 / (v * v) for v in vals) / len(vals))


def analyse(rows: list[dict]) -> None:
    for response in ("prefill_tps", "gen_tps", "vram_free_mb"):
        print("\n" + "=" * 70)
        print(f"MAIN EFFECTS on {response}   (marginal means over the L18)")
        print("=" * 70)
        ranked = []
        for name, (_col, levels) in FACTORS.items():
            cells = []
            for lv in levels:
                vals = [r[response] for r in rows if r.get(name) == lv and r.get(response)]
                cells.append((lv, sum(vals) / len(vals) if vals else float("nan"),
                              sn_larger_is_better(vals) if response != "vram_free_mb"
                              else float("nan"), len(vals)))
            finite = [m for _, m, _, _ in cells if m == m]
            spread = (max(finite) - min(finite)) if len(finite) > 1 else 0.0
            ranked.append((spread, name, cells))
        # Rank by spread: in a Taguchi screen the factor whose levels move the response
        # most is the one worth confirming, and the unused columns 7-8 set the noise floor.
        for spread, name, cells in sorted(ranked, reverse=True):
            body = "   ".join(f"{lv}:{m:.1f}" for lv, m, _, _ in cells)
            print(f"  {name:<9} spread={spread:>8.1f}   {body}")
            if response != "vram_free_mb":
                sn = "   ".join(f"{lv}:{s:.1f}" for lv, _, s, _ in cells)
                print(f"  {'':<9} S/N dB              {sn}")

    print("\n" + "-" * 70)
    print("A spread is a RANKING signal, not a measurement. Confirm the top one or two")
    print("with ab_isolate.py, which pairs, interleaves, flips order and reports CI95.")
    print("L18 confounds two-factor interactions into the residual by construction.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=2)
    args = ap.parse_args()
    rows: list[dict] = []
    for rep in range(args.replicates):
        # Reverse on odd replicates: run order correlates with machine warmth, and a
        # fixed order would alias that drift onto whichever factor moves along the array.
        order = list(enumerate(L18)) if rep % 2 == 0 else list(reversed(list(enumerate(L18))))
        for i, row in order:
            s = _settings(row)
            tag = f"r{rep}-{i + 1:02d}"
            print(f"[{time.strftime('%H:%M:%S')}] {tag} {s}", flush=True)
            got = run_one(s, tag)
            if got:
                rows.append({**got, "rep": rep, "run": i + 1})
                print(f"  -> prefill={got['prefill_tps']:.1f} gen={got['gen_tps']:.1f} "
                      f"vram_free={got['vram_free_mb']}", flush=True)
    out = pathlib.Path(__file__).parent / "runs"
    out.mkdir(exist_ok=True)
    (out / "taguchi_l18.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    analyse(rows)
    return 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        # Orthogonality is the ONLY thing that makes marginal means interpretable, so it
        # is checked rather than assumed: every level equally frequent in each column,
        # and every pair of columns balanced across level combinations.
        assert len(L18) == 18
        for col in range(8):
            counts: dict[int, int] = {}
            for row in L18:
                counts[row[col]] = counts.get(row[col], 0) + 1
            want = 9 if col == 0 else 6
            assert all(c == want for c in counts.values()), (col, counts)
        for a in range(1, 8):
            for b in range(a + 1, 8):
                pairs: dict[tuple, int] = {}
                for row in L18:
                    pairs[(row[a], row[b])] = pairs.get((row[a], row[b]), 0) + 1
                assert len(pairs) == 9 and all(v == 2 for v in pairs.values()), (a, b, pairs)
        assert abs(sn_larger_is_better([100, 100, 100]) - 40.0) < 0.01
        # A setting that averages the same but has one terrible run must score LOWER.
        assert sn_larger_is_better([100, 100, 100]) > sn_larger_is_better([145, 145, 10])
        seen = {tuple(sorted(_settings(r).items())) for r in L18}
        assert len(seen) == 18, "every run must be a distinct configuration"
        print(f"L18 orthogonal, {len(seen)} distinct configs, S/N penalises instability")
        print("taguchi_screen self-check OK")
        raise SystemExit(0)
    raise SystemExit(main())
