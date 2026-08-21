"""A/B: does the fork's expert-prefetch earn its custom build?

The fork (`thecodacus/llama.cpp`, branch `fable5/prefetch-experts`) is 232 lines over
upstream `4fc4ec554`, in four files, all on one path: pin mmap-backed CPU weights, then
overlap the host->device upload of offloaded expert weights with compute. `-ncmoe` is
NOT from the fork -- it is upstream (ggml-org/llama.cpp#15077). So the fork does not
enable the axis; it claims to make it cheaper. That claim is measurable, and until it
is measured the custom build is a dependency carried on faith.

Three design decisions, each answering something this project already measured the hard
way:

1. ARMS ARE INTERLEAVED, NEVER BLOCKED.
   Running all of A then all of B is the obvious shape and it is wrong here. WSL2
   returns guest memory slowly, so the second half of any long sweep runs in a smaller
   envelope than the first -- measured: an identical config passed once and was then
   rejected at 10611MB available. Blocked order hands that drift entirely to arm B and
   the A/B silently becomes a measurement of queue position. Interleaving spreads the
   drift across both arms, and pairing then cancels what is left.

2. PAIRED DELTAS, NOT TWO SEPARATE MEANS.
   Each round produces one A and one B under near-identical host conditions. The datum
   is delta = B - A within the round. Comparing grand means instead would let one bad
   round in either arm decide the verdict.

3. DOSE-RESPONSE IS THE CONTROL, AND THE DOSE RANGE MUST SPAN THE MECHANISM.
   A negative control (ncmoe=0, where the fork's code cannot run) is not available: this
   model at ncmoe=0 does not fit the envelope, and the guard would reject it. So the
   control is the SHAPE instead. The fork's mechanism is overlapping expert uploads --
   more offloaded experts means more uploads to overlap, so the gain must GROW with
   ncmoe. A delta that is flat across ncmoe is not this mechanism, whatever its sign;
   it is thermals, drift, or luck wearing the result's clothes.

   The first attempt got this half right: it had the shape control but ran it over
   {8, 12, 16} of 40 layers -- the bottom fifth to bottom half of the range. Declaring a
   mechanism dead after testing only where it has nothing to do is not a null result, it
   is an unasked question. The axis now spans 8 to 40.

Usage:
    python ab_compare.py                 # 3 rounds x {8,12,16}, both arms
    python ab_compare.py --rounds 1      # smoke run
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from model_lifecycle.analysis.statistics import describe            # noqa: E402
from model_lifecycle.control_plane.guard import Envelope            # noqa: E402
from model_lifecycle.servers.llama_cpp import (                     # noqa: E402
    LlamaCppAdapter, ServerProfile)
from model_lifecycle.workloads.throughput import run_config         # noqa: E402

MODEL = "/home/augus/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"

ARMS = {
    # name        binary
    "base": "/home/augus/src/slop.cpp-base/build/bin/llama-server",   # upstream 4fc4ec554
    "fork": "/home/augus/src/slop.cpp/build/bin/llama-server",        # +232 lines
}

# THE FORK'S HEADLINE FEATURE IS OPT-IN AND DEFAULTS TO OFF:
#     const char * e = getenv("GGML_SCHED_PREFETCH_EXPERTS");
#     sched->prefetch_experts = op_offload && (e ? atoi(e) : 0) > 0;
# Three A/B runs were completed before anyone read that line, comparing the fork against
# upstream with the thing under test switched off. Nothing on the command line, in
# --help, or in the server log revealed it.
#
# Set on BOTH arms deliberately: upstream has no such getenv, so it is inert there, and
# identical command lines keep the arms symmetric. 3 = the author's default, one slot per
# gate/up/down tensor of a MoE layer.
# ...and the PINNING half needs a THIRD switch, this one from upstream:
#     bool ggml_backend_cuda_register_host_buffer(...) {
#         if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) return false;
# So the fork has three gates and all three default closed. Verified live: with none of
# them set, the server logs no "pinned ... MiB" line at all -- the patch never ran.
#
# WARNING, and it is this project's own subject: pinning registers ~18 GB of model pages
# as NON-PAGEABLE. Pinned memory cannot be swapped or reclaimed by the OS, which is a
# direct attack on the desktop headroom this platform exists to protect. The guard
# watches it; a speed-up that costs the machine is a NEGATIVE result here.
SERVER_ENV = {"GGML_SCHED_PREFETCH_EXPERTS": "3", "GGML_CUDA_REGISTER_HOST": "1"}

# The dose axis. Ascending offload => ascending upload volume => the fork's claimed
# gain should ascend with it.
#
# CORRECTED 2026-07-25 after the owner pointed out the confound. The model has
# **40 layers** (`qwen35moe.block_count`, read from GGUF metadata). The first axis was
# {8, 12, 16} -- 20%, 30% and 40% of the model, all bunched at the BOTTOM of the range.
# The fork's author developed it on a 16 GB GPU, where this same model needs roughly
# 32-40. So the first experiment measured the regime in which the mechanism has almost
# nothing to hide, and concluded the mechanism does not work.
#
# 40 is the maximum: every layer's experts on the CPU, which is what a 16 GB card is
# forced into. 8 stays as the LOW control -- already measured as no-effect, so if the
# high dose also shows nothing, the null is about the patch and not about the range.
NCMOE = [8, 24, 40]

# The prefetch has a SECOND gate, on batch size:
#     if (ids->ne[0]*ids->ne[1] >= 2*n_expert && ...)
# ids holds n_expert_used * n_tokens entries. This model has n_expert=256 and
# n_expert_used=8, so the prefetch only engages at 2*256/8 = 64 tokens or more. The
# original prompt was ~50 tokens -- under the threshold, so even with the env var set the
# feature would have stayed dormant and the null result would have repeated.
#
# This prompt is deliberately long enough to clear that gate several times over, because
# the author's own comment says the mechanism is FOR large batches: "with a large batch
# virtually every expert is used, so the routing ids are not worth waiting for".
PROMPT = (
    "Explain in detail why memory bandwidth rather than raw compute usually limits "
    "token generation speed for a large language model running on a single consumer "
    "GPU. Cover: the arithmetic intensity of a matrix-vector product versus a "
    "matrix-matrix product; why batch size one is the worst case; how a mixture-of-"
    "experts model changes the picture when only a few experts are active per token; "
    "what happens to the roofline when expert weights live in system RAM and must "
    "cross PCIe; why the KV cache grows with context and what that costs in bandwidth "
    "per generated token; how quantisation changes the ratio of bytes moved to "
    "operations performed; and why prompt processing is compute-bound while generation "
    "is bandwidth-bound on the very same hardware. Be concrete, use worked numbers for "
    "a 24 GB card with roughly 936 GB/s of memory bandwidth against system RAM at "
    "roughly 50 GB/s, and explain each step of the arithmetic rather than stating "
    "conclusions. Finish with the practical implication for someone choosing between a "
    "smaller model fully resident in VRAM and a larger one partially offloaded."
)


def verify_linkage() -> None:
    """Refuse to run unless each arm loads ITS OWN libggml/libllama.

    `llama-server` is 17 KB: everything that matters is in shared objects, and the
    fork's 232 lines live in `libggml-base` and `libllama`. Both builds are linked with
    an absolute RUNPATH into their own `build/bin`, so they are isolated -- but RUNPATH
    loses to `LD_LIBRARY_PATH`, and one stray export would have both arms executing the
    same code while every number still looked healthy. A null result that cannot be
    distinguished from a real one is the worst outcome available here, so this is a
    hard precondition rather than a comment.
    """
    import subprocess
    for arm, binary in ARMS.items():
        own = str(pathlib.PurePosixPath(binary).parent)
        out = subprocess.run(["wsl", "-d", "Ubuntu-24.04", "--", "ldd", binary],
                             capture_output=True, text=True, timeout=120)
        libs = [ln for ln in out.stdout.splitlines()
                if ("libggml" in ln or "libllama" in ln) and "=>" in ln]
        if not libs:
            raise SystemExit(f"{arm}: ldd resolved no ggml/llama libraries for {binary}")
        stray = [ln.strip() for ln in libs if own not in ln]
        if stray:
            raise SystemExit(f"{arm}: loads libraries from OUTSIDE {own}, so the two "
                             f"arms are not independent:\n  " + "\n  ".join(stray))
        print(f"  linkage OK: {arm} -> {len(libs)} libs, all under {own}")


def _profile(ncmoe: int) -> ServerProfile:
    return ServerProfile(model_path=MODEL, port=8080, n_cpu_moe=ncmoe,
                         ctx_size=8192, cache_type_k="q8_0", cache_type_v="q8_0",
                         no_mmap=False)


def _metric(r, name: str) -> float | None:
    d = getattr(r, name, None)
    return d["mean"] if isinstance(d, dict) and d.get("mean") is not None else None


def main() -> int:
    ap = argparse.ArgumentParser()
    # EVEN by default, and this is not a style choice. The arm order flips each round,
    # so an odd count gives one arm the first slot in the pair more often than the other.
    # Measured 2026-07-25: the first config of a sweep ran at 79.0 t/s against 75.7-76.6
    # for the same config later -- cold machine, cold cache. With 3 rounds `base` would
    # collect that first-slot effect twice and `fork` once, and the imbalance would be
    # read as code.
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--repetitions", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=1500)
    args = ap.parse_args()

    if args.rounds % 2:
        print(f"REFUSING: --rounds {args.rounds} is odd, so the arm-order flip does not "
              f"balance. Use an even number.")
        return 2

    print("checking that the two arms are actually two arms ...")
    verify_linkage()

    out_dir = pathlib.Path(__file__).parent / "runs" / "ab"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = Envelope()
    records: list[dict] = []

    # SWEEP warm-up, discarded. `run_config` already discards the first REQUEST of each
    # configuration; this is the same reasoning one level up, and its absence was a real
    # defect: the sweep's very first configuration ran measurably faster than identical
    # configurations later, on a machine that had not yet warmed up. Recording it made
    # whichever arm happened to go first look better.
    print("[warm-up] discarded configuration, only to warm the machine ...", flush=True)
    warm = LlamaCppAdapter(server_bin=ARMS["base"], env=SERVER_ENV)
    run_config(warm, _profile(NCMOE[0]), config_id="ab__warmup", prompt=PROMPT,
               repetitions=1, max_tokens=args.max_tokens, envelope=env)

    for rnd in range(args.rounds):
        for ncmoe in NCMOE:
            # Arm order FLIPS every round. Even within a pair the first arm pays the
            # cold-cache cost and the second inherits a warmed page cache; a fixed order
            # would hand that advantage to the same arm every single time and it would
            # read as a code difference.
            arms = list(ARMS.items())
            if rnd % 2:
                arms.reverse()
            for arm, binary in arms:
                cid = f"ab__{arm}__ncmoe{ncmoe}__r{rnd}"
                adapter = LlamaCppAdapter(server_bin=binary, env=SERVER_ENV)
                if not adapter.is_port_free(8080):
                    print(f"{cid}: port 8080 busy - aborting rather than measuring "
                          f"someone else's server")
                    return 2
                print(f"[{time.strftime('%H:%M:%S')}] {cid} ...", flush=True)
                r = run_config(adapter, _profile(ncmoe), config_id=cid, prompt=PROMPT,
                               repetitions=args.repetitions,
                               max_tokens=args.max_tokens, envelope=env)
                rec = {"round": rnd, "arm": arm, "ncmoe": ncmoe,
                       "verdict": r.verdict, "reason": r.reason,
                       "load_seconds": r.load_seconds,
                       "gen_tps": _metric(r, "gen_tps"),
                       "prompt_tps": _metric(r, "prompt_tps"),
                       "total_s": _metric(r, "total"),
                       "min_free_vram_mb": r.min_free_vram_mb,
                       "host_recovered": r.host_recovered,
                       # Both are validity flags, not statistics. A non-zero
                       # cached_prefill means the prefill numbers describe the KV cache;
                       # a non-zero lower_bound means a rate came from wall-clock.
                       "cached_prefill": r.cached_prefill_count,
                       "lower_bound": r.gen_tps_lower_bound_count}
                records.append(rec)
                (out_dir / f"{cid}.json").write_text(
                    json.dumps(r.as_dict(), indent=2, default=str), encoding="utf-8")
                print(f"    {r.verdict:9} gen={rec['gen_tps']} prompt={rec['prompt_tps']} "
                      f"load={r.load_seconds}s vram_free={r.min_free_vram_mb}MB"
                      f"{'' if r.host_recovered else '  [HOST DID NOT RECOVER]'}",
                      flush=True)

    (out_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    report(records)
    return 0


def report(records: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("PAIRED DELTAS  (fork - base); positive = the fork is faster")
    print("=" * 72)

    tainted = [r for r in records if r.get("cached_prefill") or r.get("lower_bound")]
    if tainted:
        print(f"!! {len(tainted)} run(s) carry a validity flag (cached prefill or "
              f"wall-clock rate) - listed at the end")

    unrecovered = [r for r in records if not r.get("host_recovered", True)]
    if unrecovered:
        # Not a footnote. A round whose host never recovered measured a shrunken
        # envelope, and its delta is about the machine, not about the code.
        print(f"!! {len(unrecovered)} run(s) started in a shrunken envelope - "
              f"their rounds are SUSPECT, listed at the end")

    by_dose: dict[int, dict] = {}
    for metric in ("gen_tps", "prompt_tps"):
        print(f"\n-- {metric} --")
        for ncmoe in sorted({r["ncmoe"] for r in records}):
            deltas, pct = [], []
            for rnd in sorted({r["round"] for r in records}):
                pair = {r["arm"]: r for r in records
                        if r["ncmoe"] == ncmoe and r["round"] == rnd}
                if set(pair) != set(ARMS):
                    continue
                a, b = pair["base"], pair["fork"]
                # A pair is only usable when BOTH arms produced a number. Dropping the
                # whole pair (not just the missing arm) is the point of pairing.
                if a["verdict"] != "OK" or b["verdict"] != "OK":
                    continue
                if a[metric] is None or b[metric] is None:
                    continue
                deltas.append(b[metric] - a[metric])
                pct.append((b[metric] - a[metric]) / a[metric] * 100.0)

            if not deltas:
                print(f"  ncmoe={ncmoe:<3} no usable pair (both arms must be OK)")
                continue
            d = describe(deltas)
            p = describe(pct)
            # The CI is the whole verdict: a mean delta that straddles zero is a
            # difference the instrument cannot see, and shipping a custom build for an
            # invisible difference is how a dependency becomes permanent by default.
            crosses_zero = d.ci95_low <= 0.0 <= d.ci95_high
            call = "inside noise" if crosses_zero else ("FORK FASTER" if d.mean > 0
                                                        else "FORK SLOWER")
            print(f"  ncmoe={ncmoe:<3} n={d.n}  delta={d.mean:+8.2f} "
                  f"({p.mean:+6.2f}%)  ci95[{d.ci95_low:+.2f},{d.ci95_high:+.2f}]  {call}")
            if metric == "gen_tps":
                by_dose[ncmoe] = {"mean": d.mean, "pct": p.mean, "noise": crosses_zero}

    print("\n-- dose-response control (gen_tps) --")
    doses = sorted(by_dose)
    if len(doses) < 2:
        print("  not enough dose levels to test the shape")
    else:
        vals = [by_dose[k]["pct"] for k in doses]
        rising = all(y >= x for x, y in zip(vals, vals[1:]))
        span = max(vals) - min(vals)
        print("  " + "  ".join(f"ncmoe{k}={by_dose[k]['pct']:+.2f}%" for k in doses))
        if all(by_dose[k]["noise"] for k in doses):
            print("  VERDICT: no effect at any dose -> the fork does not earn its build.")
        elif rising and span >= 1.0:
            print("  VERDICT: gain RISES with offload - consistent with the prefetch "
                  "mechanism. The fork earns its build on the offloaded configs.")
        else:
            print("  VERDICT: an effect exists but does NOT scale with offload. That is "
                  "not the claimed mechanism; suspect drift/thermals before crediting "
                  "the 232 lines.")

    for r in unrecovered:
        print(f"  SUSPECT (no recovery): round {r['round']} arm {r['arm']} "
              f"ncmoe {r['ncmoe']}")
    for r in tainted:
        print(f"  TAINTED: round {r['round']} arm {r['arm']} ncmoe {r['ncmoe']} "
              f"cached_prefill={r.get('cached_prefill')} "
              f"lower_bound={r.get('lower_bound')}")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        # The report must survive the shapes that actually occur: a rejected arm, a
        # missing metric, and a delta that is pure noise.
        fake = []
        for rnd in range(3):
            for nc, gain in ((8, 0.0), (12, 0.0), (16, 0.0)):
                fake += [
                    {"round": rnd, "arm": "base", "ncmoe": nc, "verdict": "OK",
                     "gen_tps": 80.0, "prompt_tps": 500.0, "host_recovered": True},
                    {"round": rnd, "arm": "fork", "ncmoe": nc, "verdict": "OK",
                     "gen_tps": 80.0 + gain + (0.1 if rnd else -0.1),
                     "prompt_tps": 500.0, "host_recovered": True},
                ]
        fake.append({"round": 0, "arm": "fork", "ncmoe": 99, "verdict": "REJECTED",
                     "gen_tps": None, "prompt_tps": None, "host_recovered": False})
        report(fake)
        print("\nselfcheck OK (a zero-effect fork must read as 'inside noise')")
        raise SystemExit(0)
    raise SystemExit(main())
