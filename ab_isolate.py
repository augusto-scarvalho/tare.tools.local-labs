"""Compare llama.cpp builds and switches, one question per arm-set.

Started as "does the fork's pinning earn its 12.3 GB". It now carries three arm-sets,
selected with `--arms`, because the same paired machinery answers three questions and
each needs a different thing subtracted:

    --arms switches   base(4fc4ec554) / prefetch-only / prefetch+pinning
                      ANSWERED: prefetch alone is -26% on prefill (async out of pageable
                      memory is not async); prefetch+pinning is +58%. The switches are
                      NOT independent and there is no cheap middle option.

    --arms versions   base(4fc4ec554) / upstream master today / fork
                      Is the fork still worth carrying? Upstream is 266 commits ahead and
                      the fork has had no commit since 2026-07-08.

    --arms rebased    upstream master today / fork / fork REBASED onto master
                      The patch replayed onto current upstream. It applied with zero
                      conflicts and an identical diffstat, so in 266 commits upstream
                      never touched a line the fork changes.

Design decisions and what each defends against are documented at their definitions:
interleaved arms, paired deltas, dose-response over a range that spans the mechanism,
linkage verification, a discarded warm-up configuration, and a refusal to run an odd
number of rounds.

    python ab_isolate.py --arms rebased --rounds 4
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from model_lifecycle.analysis.statistics import describe            # noqa: E402
from model_lifecycle.analysis.robust import (                      # noqa: E402
    bootstrap_ci, sign_test_p, min_rounds_for, hodges_lehmann, mad, cliffs_delta)
from model_lifecycle.control_plane.guard import Envelope            # noqa: E402
from model_lifecycle.models import ab_models                        # noqa: E402
from model_lifecycle.servers.llama_cpp import (                     # noqa: E402
    LlamaCppAdapter, ServerProfile)
from model_lifecycle.workloads.throughput import run_config         # noqa: E402

# key -> (gguf, block_count, n_expert, n_expert_used), from the shared registry
# (model_lifecycle.models). Offload is expressed as a FRACTION of layers, never a fixed
# count, so "model" and "dose" do not become the same variable.
#
# A second geometry is not a nicety here. Three separate experiments agreed that the
# prefetch is a 22-36% tax -- and all three were the same GGUF. The first independent
# geometry reversed the sign. Any arm-set in this file that is run on one model answers a
# question about that model. The geometry details (why the Nemotron at 88 blocks / 512
# experts / 22 used is the transfer-bound B1 subject; why its path is a runtime-resolved
# glob) now live at their definition in model_lifecycle/models.py.
MODELS = ab_models()
MODEL = MODELS["qwen36-35b"][0]      # replaced by main() from --model

BASE_BIN   = "/home/augus/src/llama.cpp-base/build/bin/llama-server"     # 4fc4ec554
FORK_BIN   = "/home/augus/src/llama.cpp/build/bin/llama-server"          # 5e7f6271c
MASTER_BIN = "/home/augus/src/llama.cpp-master/build/bin/llama-server"   # upstream today
REBASE_BIN = "/home/augus/src/llama.cpp-rebase/build/bin/llama-server"   # fork rebased on it
STACK_BIN  = "/home/augus/src/llama.cpp-stack/build/bin/llama-server"    # all 8 branches
LOCAL_BIN  = "/home/augus/src/llama.cpp-local/build/bin/llama-server"    # ours

BOTH_SWITCHES = {"GGML_SCHED_PREFETCH_EXPERTS": "3", "GGML_CUDA_REGISTER_HOST": "1"}

# name -> (binary, env). Three arm-sets, because there are three different questions
# and mixing them into one run would answer none of them cleanly. Each set names its own
# `base`, since what you subtract depends on what you are asking.
ARM_SETS = {
    # Question 1: what does each SWITCH cost and buy? A ladder -- each rung adds exactly
    # one switch, so the delta between rungs is that switch's price. ANSWERED: prefetch
    # alone is -26% on prefill (pageable async is not async); prefetch+pinning is +58%.
    "switches": {
        "base":     (BASE_BIN, {}),
        "prefetch": (FORK_BIN, {"GGML_SCHED_PREFETCH_EXPERTS": "3"}),
        "both":     (FORK_BIN, dict(BOTH_SWITCHES)),
    },
    # Question 2: is the fork still worth carrying? Upstream master is 266 commits ahead
    # of the fork's branch point and the fork has had no new commit since 2026-07-08. If
    # today's stock llama.cpp matches the fork, the custom build goes away regardless of
    # what its 232 lines do -- which is the decision the owner actually asked for.
    #
    # `base` stays in the set as the continuity anchor: every earlier number in this
    # project is measured against it, so keeping it makes the new run comparable to the
    # old ones instead of starting a fresh, unrelatable series.
    "versions": {
        "base":   (BASE_BIN, {}),
        "master": (MASTER_BIN, dict(BOTH_SWITCHES)),   # inert there, kept for symmetry
        "fork":   (FORK_BIN, dict(BOTH_SWITCHES)),
    },
    # Question 3: the fork's patch replayed onto TODAY's upstream. The rebase applied
    # with zero conflicts and an identical diffstat -- in 266 commits upstream never
    # touched a line the fork changes -- so this arm is the fork's mechanism plus three
    # weeks of upstream work, which is the configuration a user would actually want.
    #
    # `base` here is MASTER, not 4fc4ec554: the question is what the patch adds to
    # current upstream, so current upstream is the thing to subtract.
    "rebased": {
        "base":    (MASTER_BIN, dict(BOTH_SWITCHES)),   # today's upstream, env inert
        "fork":    (FORK_BIN, dict(BOTH_SWITCHES)),     # the patch on 3-week-old upstream
        "rebased": (REBASE_BIN, dict(BOTH_SWITCHES)),   # the patch on today's upstream
    },
    # Question 0, and it should have been first. A NULL A/B: two arms that are the same
    # binary with the same environment, where the true delta is zero by construction.
    #
    # Everything this project has published is a delta compared against nothing. The
    # harness research playbook states the rule plainly -- "before promoting any delta,
    # especially sub-10% gains, run the noise-floor probe; a delta below the floor is not
    # evidence" -- and the deltas currently in dispute are 1.8% (the L18's prefetch),
    # -10.4% (the local build's residual no-mmap cost) and 0.8% (prefetch slots). All
    # three live in exactly the band where the floor decides whether they exist.
    #
    # A null A/B measures more than round-to-round scatter: it also catches SYSTEMATIC
    # order effects, because arm `a` and arm `b` occupy the first and second slot of the
    # pair equally often. If the null comes back non-zero, the interleaving is not
    # balancing what it was built to balance, and every paired number here inherits that
    # bias. That failure mode is invisible to any amount of replication of a real A/B.
    "null": {
        "base": (MASTER_BIN, dict(BOTH_SWITCHES)),
        "same": (MASTER_BIN, dict(BOTH_SWITCHES)),
    },
    # Question 7 (B1): does pinning help GENERATION when generation is TRANSFER-BOUND?
    #
    # This project published "generation is unaffected by anything" from arms whose
    # generation was not transfer-bound: the 35B at ncmoe=24 keeps most weight near the
    # card and generates at 44 t/s. Nemotron at ncmoe=99 fetches 22 expert tensors per
    # token across PCIe and generates at 0.64 t/s. That is the regime where turning a
    # bounce-buffered copy into a direct DMA should matter most, and it is exactly the
    # regime never measured.
    #
    # The prefetch is deliberately absent from both arms: its batch gate needs
    # ids >= 2*n_expert, which for 512/22 is 47 tokens, and generation supplies ONE. It
    # cannot engage here, so including it would vary a switch that does nothing.
    "genpin": {
        "base": (LOCAL_BIN, {}),                                 # no pinning
        "pin":  (LOCAL_BIN, {"GGML_CUDA_REGISTER_HOST": "1"}),   # pinning only
    },
    # Question 6: the fork's OTHER half. Everything measured so far is prefill; generation
    # has never moved -- `ncmoe` explains 99.8% of its variance and every other factor sits
    # at the noise floor. `turbo-mma-decode` is the fork author's own attempt at exactly
    # that half: a fused GQA tensor-core flash-attention decode path.
    #
    # No merge is needed to test it. The stack build already carries all eight branches and
    # this one ships a documented kill-switch, so the A/B is one binary with one env var --
    # the cleanest design available, and strictly better than merging 28 commits to find
    # out whether a feature that toggles at runtime is worth having.
    #
    # NOTE the switch is DEFAULT ON (`return !(s && s[0] == '0')`), so `base` here is the
    # DISABLED arm. Every stack measurement this project has taken already had this path
    # active while the rebase and master arms did not even contain the code.
    "decode": {
        "base":  (STACK_BIN, {"GGML_CUDA_REGISTER_HOST": "1",
                              "GGML_TURBO_MMA_FUSED": "0"}),
        "turbo": (STACK_BIN, {"GGML_CUDA_REGISTER_HOST": "1"}),
    },
    # Question 5, and it turns out to be THE question. Every "+58%" this project published
    # compared `pinning + prefetch` against `neither`. Nobody ever measured pinning ALONE,
    # so the fork's two mechanisms have never been separated on the same build.
    #
    # The 2x2 above forced the issue: on one binary, adding the prefetch to an ALREADY
    # PINNED baseline measured -21.9% (n=6, sign p=0.031). `llama-model-loader.cpp` calls
    # `register_host()` unconditionally whenever mmap is in use and the backend exposes
    # the proc address -- it does NOT depend on the prefetch switch -- so that baseline
    # really was pinned, and the loss is real rather than a re-measurement of the known
    # -26% on pageable memory.
    #
    # Three arms, one build, one factor added at a time. If pinning alone lands near the
    # ~1000 t/s that `--no-mmap` reaches, then the fork's entire value is the
    # cudaHostRegister call, its headline overlap is a 22% tax, and the +58% was one
    # number hiding two effects pointing in opposite directions.
    "pinning": {
        "base":  (LOCAL_BIN, {}),                                   # neither
        "pin":   (LOCAL_BIN, {"GGML_CUDA_REGISTER_HOST": "1"}),     # pinning only
        "pinpf": (LOCAL_BIN, dict(BOTH_SWITCHES)),                  # pinning + prefetch
    },
    # Question 4: the L18 screen and the paired A/B disagree about the SAME switch. The
    # A/B says prefetch+pinning is +58% on prefill; the L18's marginal means put prefetch
    # OFF fastest (1719.8 off / 1481.1 at 3 / 1666.4 at 8) and non-monotonic, which is a
    # confounding signature rather than a mechanism.
    #
    # What differs between the two runs is the BUILD: the L18 ran on the stack, the +58%
    # came from the `prefetch-experts` branch. The first guess was that the stack's expert
    # CACHE competes with the prefetch for the same bottleneck -- but `--moe-cache-slots`
    # without `--moe-cache-profile` is documented as disabled (`common.h`: "empty =
    # disabled") and `llama-moe-trace`, which produces the profile, is not even built. The
    # L18's `cache` factor was inert BY CONSTRUCTION, which is also why it scored 0.1% of
    # variance -- the same default-closed trap that made three complete A/B runs measure a
    # fork whose three feature gates were all closed.
    #
    # The real difference is in the prefill path itself. The stack carries
    # `b69dc9499 route MUL_MAT_ID with skip-capable ids away from mmq/mmf` and the dual
    # hot/cold MoE graph wiring -- changes to the exact operation the prefetch accelerates,
    # active whether or not the cache is enabled.
    #
    # So the 2x2 is BUILD x PREFETCH, the pair actually confounded:
    #     rebase+off   rebase+on     <- must reproduce the +58%
    #     stack +off   stack +on     <- must reproduce the L18's ordering
    # The interaction contrast is the whole answer and is PRINTED rather than eyeballed
    # from four cells: this project already called a result "partly right" from one cell
    # of a table whose other cells said otherwise.
    "stack": {
        "base":     (REBASE_BIN, {"GGML_CUDA_REGISTER_HOST": "1"}),
        "prefetch": (REBASE_BIN, dict(BOTH_SWITCHES)),
        "stack":    (STACK_BIN,  {"GGML_CUDA_REGISTER_HOST": "1"}),
        "stackpf":  (STACK_BIN,  dict(BOTH_SWITCHES)),
    },
}

# arm-set -> the four cells of a 2x2, as (a00, a10, a01, a11): factor A varies 00->10,
# factor B varies 00->01. Only the interaction needs naming; the main effects are already
# in the paired table above.
FACTORIAL_2X2 = {
    "stack": {"a": "prefetch", "b": "build",
              "cells": ("base", "prefetch", "stack", "stackpf")},
}

# Arms are (binary, env) or (binary, env, extra_args). Normalised once, here, so every
# consumer can assume three fields instead of guessing.
for _set in ARM_SETS.values():
    for _name, _spec in list(_set.items()):
        if len(_spec) == 2:
            _set[_name] = (_spec[0], _spec[1], ())

ARMS = ARM_SETS["switches"]      # replaced by main() from --arms

# The three gates, all default-closed, found by reading the patch after three A/B runs
# had already compared upstream against a fork that never executed:
#   1. GGML_SCHED_PREFETCH_EXPERTS > 0        (fork)     -> enables the prefetch at all
#   2. ids->ne[0]*ids->ne[1] >= 2*n_expert    (fork)     -> needs >=64 tokens here
#   3. GGML_CUDA_REGISTER_HOST                (upstream) -> enables the pinning
#
# Gates 1 and 3 are the two switches this script separates; gate 2 is handled by PROMPT.
# There is deliberately no module-level "enable everything" constant: each arm owns its
# own env in ARMS, so there is exactly one place that decides what an arm is.

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
    for arm, (binary, _env, _extra) in ARMS.items():
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


def _profile(ncmoe: int, extra: tuple[str, ...] = ()) -> ServerProfile:
    return ServerProfile(model_path=MODEL, port=8080, n_cpu_moe=ncmoe,
                         ctx_size=8192, cache_type_k="q8_0", cache_type_v="q8_0",
                         no_mmap=False, extra_args=extra)


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
    # 6, not 4. At n=4 the exact sign test bottoms out at p=0.125, so every "CI excludes
    # zero" this project published at 4 rounds was a parametric claim wearing a
    # distribution-free coat. 6 is the first count that can reach p<0.05.
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--repetitions", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--arms", choices=sorted(ARM_SETS), default="switches")
    ap.add_argument("--model", choices=sorted(MODELS), default="qwen36-35b")
    # Explicit file override, keeping --model's GEOMETRY. The design is arch-keyed on
    # purpose (offload is a fraction of layers; quant varies separately), so a quant that
    # is not the arch default is run by naming it here -- e.g. Nemotron IQ1_S, the only
    # quant that fits resident+pinned inside the RAM reserve, where the arch default Q3
    # (61.7 GB) cannot.
    ap.add_argument("--gguf", help="override the model file; keep --model's geometry")
    # A re-run at a different quant/dose must NOT overwrite an earlier run's raw records
    # (immutability is the one promise here). The output dir is ab-{arms}-{model}; --tag
    # suffixes it, so e.g. the Nemotron IQ1_S genpin run writes ab-genpin-nemotron-120b-iq1s
    # and leaves the earlier ab-genpin-nemotron-120b (the rejected Q3) intact.
    ap.add_argument("--tag", default="", help="suffix for the output dir, keeps re-runs separate")
    # One dose, when the question is about an INTERACTION rather than a dose-response.
    # The 2x2 in --arms stack needs four arms per cell; running it at three doses would
    # triple a run whose answer does not depend on the dose.
    ap.add_argument("--ncmoe", type=int, action="append",
                    help="override the dose axis (repeatable)")
    args = ap.parse_args()

    if args.rounds % 2:
        print(f"REFUSING: --rounds {args.rounds} is odd, so the arm-order flip does not "
              f"balance. Use an even number.")
        return 2

    global MODEL, NCMOE
    gguf, blocks, n_expert, n_used = MODELS[args.model]
    if args.gguf:
        gguf = args.gguf                      # explicit quant; geometry stays from --model
    elif "*" in gguf:
        # HF cache paths carry a snapshot hash and a quant directory. Resolving by
        # basename rather than globbing a guessed layout: a glob written from memory
        # matched nothing once and 24 configurations were reported as "did not fit"
        # without a single load being attempted.
        from residency_sweep import resolve
        found = resolve(gguf)
        if not found:
            print(f"REFUSING: could not resolve {gguf}")
            return 2
        gguf = found
    MODEL = gguf
    if args.ncmoe:
        NCMOE = list(args.ncmoe)
    else:
        # 60% of layers, the same dose fraction every earlier run used on the 40-layer
        # model. A fixed integer would mean a different offload fraction per model.
        NCMOE = [max(1, round(blocks * 0.6))]
    print(f"model={args.model} ({blocks}L, {n_expert}/{n_used} experts, gate at "
          f"{max(1, -(-2 * n_expert // n_used))} tok)  ncmoe={NCMOE}")

    global ARMS
    ARMS = ARM_SETS[args.arms]
    print(f"arm set '{args.arms}': " + ", ".join(ARMS))

    print("checking that the arms are actually different arms ...")
    verify_linkage()

    # Namespaced by arm-set. Previously every run wrote runs/ab/records.json, so the
    # `rebased` run silently destroyed the `versions` run's raw data. Raw records are
    # the one thing this platform promises to keep -- scores are recomputable, runs
    # are not.
    _suffix = f"-{args.tag}" if args.tag else ""
    out_dir = pathlib.Path(__file__).parent / "runs" / f"ab-{args.arms}-{args.model}{_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = Envelope()
    records: list[dict] = []

    # SWEEP warm-up, discarded. `run_config` already discards the first REQUEST of each
    # configuration; this is the same reasoning one level up, and its absence was a real
    # defect: the sweep's very first configuration ran measurably faster than identical
    # configurations later, on a machine that had not yet warmed up. Recording it made
    # whichever arm happened to go first look better.
    print("[warm-up] discarded configuration, only to warm the machine ...", flush=True)
    warm = LlamaCppAdapter(server_bin=ARMS["base"][0], env={})
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
            for arm, (binary, arm_env, arm_extra) in arms:
                cid = f"ab__{arm}__ncmoe{ncmoe}__r{rnd}"
                adapter = LlamaCppAdapter(server_bin=binary, env=arm_env)
                if not adapter.is_port_free(8080):
                    print(f"{cid}: port 8080 busy - aborting rather than measuring "
                          f"someone else's server")
                    return 2
                print(f"[{time.strftime('%H:%M:%S')}] {cid} ...", flush=True)
                r = run_config(adapter, _profile(ncmoe, arm_extra), config_id=cid,
                               prompt=PROMPT, repetitions=args.repetitions,
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
    print(f"raw records -> {out_dir / 'records.json'}")
    report(records)
    return 0


def report(records: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("PAIRED DELTAS vs base; positive = the fork arm is faster")
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
    # Each non-base arm is compared against base INDEPENDENTLY, so the table reads as a
    # ladder: what prefetch alone buys, and what adding pinning buys on top of it. That
    # second number is the one with 12.3 GB of locked memory attached to it.
    arms_under_test = [a for a in ARMS if a != "base"]
    for arm_under_test in arms_under_test:
      for metric in ("gen_tps", "prompt_tps"):
        print(f"\n-- {arm_under_test} vs base :: {metric} --")
        for ncmoe in sorted({r["ncmoe"] for r in records}):
            deltas, pct = [], []
            for rnd in sorted({r["round"] for r in records}):
                pair = {r["arm"]: r for r in records
                        if r["ncmoe"] == ncmoe and r["round"] == rnd}
                if not {"base", arm_under_test} <= set(pair):
                    continue
                a, b = pair["base"], pair[arm_under_test]
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
            # The normal-approximation CI above is doing real work at n=4. Three checks
            # that do not assume normality, printed beside it so the claim can be read at
            # its true strength rather than at its most flattering.
            blo, bhi = bootstrap_ci(deltas)
            sp = sign_test_p(deltas)
            need = min_rounds_for(abs(p.mean), max(d.cv * 100.0, 0.1))
            floor = "  [n too small for p<0.05 without assuming a distribution]"                 if sp > 0.05 else ""
            print(f"  {'':<10} boot95[{blo:+.2f},{bhi:+.2f}]  sign-test p={sp:.3f}"
                  f"{floor}")

            # Robust location and spread BESIDE the mean, and an effect size that does not
            # grow with n. These estimators were written for exactly the round that
            # happened on 2026-07-25 -- one configuration measured in a shrunken envelope
            # after a stray server took 10 GB of the card -- and then were not called by
            # any report for the rest of the day. A mean and an sd carry that round at full
            # weight; the Hodges-Lehmann estimate and the MAD barely move.
            #
            # When HL and the mean disagree by more than the MAD, one round is steering the
            # headline and the table says so instead of leaving it to be noticed.
            hl, sp_mad = hodges_lehmann(deltas), mad(deltas)
            cd = cliffs_delta([r[metric] for r in records
                               if r["ncmoe"] == ncmoe and r["arm"] == arm_under_test
                               and r["verdict"] == "OK" and r[metric] is not None],
                              [r[metric] for r in records
                               if r["ncmoe"] == ncmoe and r["arm"] == "base"
                               and r["verdict"] == "OK" and r[metric] is not None])
            skew = abs(hl - d.mean) > sp_mad and sp_mad > 0
            print(f"  {'':<10} HL={hl:+.2f} MAD={sp_mad:.2f} (sd={d.stdev:.2f})  "
                  f"cliff={cd:+.2f}"
                  + ("   <- MEAN AND MEDIAN-LIKE DISAGREE: one round is steering it"
                     if skew else ""))
            if crosses_zero:
                # A round count is only useful advice when it is achievable. Resolving a
                # near-zero effect needs unbounded n, and printing "542514483 rounds" is
                # arithmetically true and operationally noise.
                if abs(p.mean) < 0.5:
                    print(f"  {'':<10} effect is ~0; no round count resolves it. The "
                          f"useful statement is the bound: |delta| < "
                          f"{max(abs(d.ci95_low), abs(d.ci95_high)):.2f}")
                elif need <= 200:
                    print(f"  {'':<10} to resolve {abs(p.mean):.1f}% at this noise: "
                          f"~{need} rounds")
                else:
                    print(f"  {'':<10} to resolve {abs(p.mean):.1f}% would need >200 "
                          f"rounds -- not worth chasing on this host")
            by_dose.setdefault(arm_under_test, {}).setdefault(metric, {})[ncmoe] = {
                "pct": p.mean, "noise": crosses_zero}

    # One verdict per ARM and per METRIC.
    #
    # The previous version read gen_tps ONLY and then announced whether "the fork earns
    # its build". Run live 2026-07-25 it printed "gain RISES with offload - the fork
    # earns its build" on the strength of a ~1% generation delta, while the actual
    # justification -- +57 to +68% on PREFILL -- was never consulted. Right answer,
    # wrong evidence, which is the same defect this project withdrew twice earlier the
    # same day.
    #
    # A single headline cannot be honest here: the two metrics disagree by two orders of
    # magnitude. So there is no single headline.
    for arm in arms_under_test:
        print(f"\n-- verdict :: {arm} vs base --")
        for metric in ("prompt_tps", "gen_tps"):
            cells = by_dose.get(arm, {}).get(metric, {})
            doses = sorted(cells)
            if len(doses) < 2:
                print(f"  {metric:<11} not enough dose levels")
                continue
            vals = [cells[k]["pct"] for k in doses]
            print(f"  {metric:<11} " + "  ".join(
                f"ncmoe{k}={cells[k]['pct']:+.2f}%" for k in doses))
            mean = sum(vals) / len(vals)
            if all(cells[k]["noise"] for k in doses):
                print(f"  {'':<11} -> no effect at any dose")
            elif all(v > 0 for v in vals):
                print(f"  {'':<11} -> FASTER at every dose (mean {mean:+.1f}%)")
            elif all(v < 0 for v in vals):
                print(f"  {'':<11} -> SLOWER at every dose (mean {mean:+.1f}%)")
            else:
                print(f"  {'':<11} -> MIXED: the sign changes across doses. Treat as "
                      f"noise unless the magnitudes are large.")

    # The interaction, when the arm-set is a 2x2. A factorial exists precisely to measure
    # the thing no screening array can see, and printing four cells without the contrast
    # leaves the reader to compute it by eye -- which is how this project once declared a
    # result "partly right" from a single cell while the rest of the table disagreed.
    #
    # Paired at the ROUND level: both differences come from the same round, so round-to-
    # round drift cancels out of the interaction exactly as it does out of a main effect.
    fac = next((f for n, f in FACTORIAL_2X2.items()
                if set(f["cells"]) == set(ARMS)), None)
    if fac:
        a00, a10, a01, a11 = fac["cells"]
        print("\n" + "=" * 72)
        print(f"INTERACTION  {fac['a']} x {fac['b']}  "
              f"(does {fac['a']}'s effect depend on {fac['b']}?)")
        print("=" * 72)
        for metric in ("prompt_tps", "gen_tps"):
            for ncmoe in sorted({r["ncmoe"] for r in records}):
                lo, hi, inter = [], [], []
                for rnd in sorted({r["round"] for r in records}):
                    pair = {r["arm"]: r for r in records
                            if r["ncmoe"] == ncmoe and r["round"] == rnd
                            and r["verdict"] == "OK" and r[metric] is not None}
                    if not set(fac["cells"]) <= set(pair):
                        continue
                    v = {k: pair[k][metric] for k in fac["cells"]}
                    e_lo = (v[a10] - v[a00]) / v[a00] * 100.0    # A's effect at B=0
                    e_hi = (v[a11] - v[a01]) / v[a01] * 100.0    # A's effect at B=1
                    lo.append(e_lo)
                    hi.append(e_hi)
                    inter.append(e_hi - e_lo)
                if not inter:
                    print(f"  {metric:<11} ncmoe={ncmoe:<3} no complete 2x2 in any round")
                    continue
                d_lo, d_hi, d_in = describe(lo), describe(hi), describe(inter)
                blo, bhi = bootstrap_ci(inter)
                sp = sign_test_p(inter)
                print(f"  {metric:<11} ncmoe={ncmoe:<3} n={d_in.n}")
                print(f"  {'':<11} {fac['a']} effect at {fac['b']}=0: "
                      f"{d_lo.mean:+7.2f}%")
                print(f"  {'':<11} {fac['a']} effect at {fac['b']}=1: "
                      f"{d_hi.mean:+7.2f}%")
                print(f"  {'':<11} INTERACTION {d_in.mean:+7.2f} pp  "
                      f"boot95[{blo:+.2f},{bhi:+.2f}]  sign p={sp:.3f}")
                if blo <= 0 <= bhi:
                    print(f"  {'':<11} -> no interaction the instrument can see. The two "
                          f"runs disagreeing is then\n  {'':<11}    NOT explained by "
                          f"{fac['b']}, and the disagreement stands open.")
                else:
                    print(f"  {'':<11} -> REAL interaction: {fac['a']}'s effect depends "
                          f"on {fac['b']}. Neither run was wrong;\n  {'':<11}    each "
                          f"measured a different cell and reported it as the effect.")

    # A null arm-set -- every arm the same binary with the same environment -- has no
    # effect to report. What it produces is the FLOOR: the smallest delta this rig can
    # tell apart from zero. Printed on its own because the useful number is the WIDTH of
    # the interval, not the point estimate, and because a non-zero null is not a small
    # result: it means the interleaving is not balancing what it exists to balance, and
    # every paired number this project has published inherits the bias.
    if len({(b, tuple(sorted(e.items())), x) for b, e, x in ARMS.values()}) == 1:
        print("\n" + "=" * 72)
        print("NOISE FLOOR (null A/B: identical arms, true delta = 0 by construction)")
        print("=" * 72)
        for metric in ("prompt_tps", "gen_tps"):
            for ncmoe in sorted({r["ncmoe"] for r in records}):
                pct = []
                for rnd in sorted({r["round"] for r in records}):
                    pair = {r["arm"]: r for r in records
                            if r["ncmoe"] == ncmoe and r["round"] == rnd}
                    if len(pair) < 2 or any(v["verdict"] != "OK" for v in pair.values()):
                        continue
                    a, b = (pair[k][metric] for k in ARMS)
                    if a and b:
                        pct.append((b - a) / a * 100.0)
                if not pct:
                    continue
                d = describe(pct)
                blo, bhi = bootstrap_ci(pct)
                floor = max(abs(blo), abs(bhi))
                verdict = ("OK: consistent with zero" if d.ci95_low <= 0 <= d.ci95_high
                           else "!! NON-ZERO NULL -- the pairing is biased, not noisy")
                print(f"  {metric:<11} ncmoe={ncmoe:<3} n={d.n}  null delta="
                      f"{d.mean:+6.2f}%  boot95[{blo:+.2f},{bhi:+.2f}]")
                print(f"  {'':<11} -> floor {floor:.2f}%: a measured effect smaller than "
                      f"this is NOT evidence.  {verdict}")

    for r in unrecovered:
        print(f"  SUSPECT (no recovery): round {r['round']} arm {r['arm']} "
              f"ncmoe {r['ncmoe']}")
    for r in tainted:
        print(f"  TAINTED: round {r['round']} arm {r['arm']} ncmoe {r['ncmoe']} "
              f"cached_prefill={r.get('cached_prefill')} "
              f"lower_bound={r.get('lower_bound')}")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        # Every arm in every set is a normalised 3-tuple, and every set names a `base`:
        # the report subtracts `base` and would otherwise produce an empty table.
        for name, aset in ARM_SETS.items():
            assert "base" in aset, f"{name} has no base arm"
            for arm, spec in aset.items():
                assert len(spec) == 3, f"{name}.{arm} not normalised: {spec}"
                assert isinstance(spec[2], tuple), f"{name}.{arm} extra_args not a tuple"

        # --arms stack must be a real 2x2: both factors present at both levels, and the
        # binary held constant. A 2x2 that changes the build between cells cannot
        # attribute anything, which is the defect it exists to correct.
        # The null set must be exactly that: identical arms. A null whose arms differ
        # measures something, and would report that something as the floor -- which is
        # worse than having no floor at all, because it looks like one.
        # B1's arms must differ ONLY in pinning. Adding the prefetch would vary a switch
        # that cannot engage during generation anyway (its gate needs 47 tokens for this
        # geometry, and generation supplies one), turning a clean two-arm test into a
        # confounded one for no information.
        gp = ARM_SETS["genpin"]
        assert len(gp) == 2, "genpin is a two-arm test"
        assert len({b for b, _, _ in gp.values()}) == 1, "genpin must hold the build fixed"
        assert all("GGML_SCHED_PREFETCH_EXPERTS" not in e for _b, e, _x in gp.values()), \
            "genpin must not vary the prefetch: it cannot engage during generation"
        assert ("GGML_CUDA_REGISTER_HOST" in gp["pin"][1]
                and "GGML_CUDA_REGISTER_HOST" not in gp["base"][1]), \
            "genpin's only difference must be pinning"

        nl = ARM_SETS["null"]
        assert len(nl) == 2, "a null A/B is two arms"
        assert len({(b, tuple(sorted(e.items())), x) for b, e, x in nl.values()}) == 1, \
            f"null arms are not identical: {nl}"

        # --arms stack must be a real 2x2: both factors present at both levels. A 2x2
        # with a missing or duplicated cell cannot attribute anything, and the
        # interaction contrast would silently average over whatever is there.
        st = ARM_SETS["stack"]
        pf = {a: st[a][1].get("GGML_SCHED_PREFETCH_EXPERTS", "0") != "0" for a in st}
        bd = {a: st[a][0] for a in st}
        assert len(set(bd.values())) == 2, f"build must take 2 levels: {set(bd.values())}"
        assert len({(pf[a], bd[a]) for a in st}) == 4, \
            f"stack is not a 2x2: {[(a, pf[a], bd[a]) for a in st]}"
        # The cells tuple must be ordered (a00, a10, a01, a11) or the interaction is
        # computed from the wrong differences and comes out mirrored.
        c = FACTORIAL_2X2["stack"]["cells"]
        assert set(c) == set(st), "cells must name every arm exactly once"
        assert not pf[c[0]] and pf[c[1]] and not pf[c[2]] and pf[c[3]], \
            "cells must be ordered (a00, a10, a01, a11) by the FIRST factor"
        assert bd[c[0]] == bd[c[1]] and bd[c[2]] == bd[c[3]] and bd[c[0]] != bd[c[2]], \
            "cells must be ordered (a00, a10, a01, a11) by the SECOND factor"

        # The report must survive the shapes that actually occur: a rejected arm, a
        # missing metric, and a delta that is pure noise.
        fake = []
        for rnd in range(3):
            for nc, gain in ((8, 0.0), (12, 0.0), (16, 0.0)):
                fake += [
                    {"round": rnd, "arm": "base", "ncmoe": nc, "verdict": "OK",
                     "gen_tps": 80.0, "prompt_tps": 500.0, "host_recovered": True},
                    {"round": rnd, "arm": "prefetch", "ncmoe": nc, "verdict": "OK",
                     "gen_tps": 80.0 + gain + (0.1 if rnd else -0.1),
                     "prompt_tps": 500.0, "host_recovered": True},
                    {"round": rnd, "arm": "both", "ncmoe": nc, "verdict": "OK",
                     "gen_tps": 80.0 + gain + (0.1 if rnd else -0.1),
                     "prompt_tps": 500.0, "host_recovered": True},
                ]
        fake.append({"round": 0, "arm": "both", "ncmoe": 99, "verdict": "REJECTED",
                     "gen_tps": None, "prompt_tps": None, "host_recovered": False})
        report(fake)

        # The interaction contrast, against data whose answer is known by construction:
        # prefetch is +50% on the rebase build and -10% on the stack. The design exists
        # to detect exactly that, so a report that misses it is worse than no report.
        import io
        from contextlib import redirect_stdout
        ARMS = ARM_SETS["stack"]
        cells, inter_fake = FACTORIAL_2X2["stack"]["cells"], []
        for rnd in range(6):
            j = 0.3 if rnd % 2 else -0.3          # a little round-to-round drift
            for arm, tps in zip(cells, (500.0, 750.0, 520.0, 468.0)):
                inter_fake.append({"round": rnd, "arm": arm, "ncmoe": 24, "verdict": "OK",
                                   "gen_tps": 80.0 + j, "prompt_tps": tps + j,
                                   "host_recovered": True})
        buf = io.StringIO()
        with redirect_stdout(buf):
            report(inter_fake)
        out = buf.getvalue()
        assert "REAL interaction" in out, out[-1500:]
        # +50% at build=0, -10% at build=1 -> the contrast is about -60 pp.
        assert "INTERACTION  -60" in out or "INTERACTION  -59" in out, \
            [l for l in out.splitlines() if "INTERACTION" in l]

        # And the mirror case: no interaction must NOT be reported as one.
        flat = []
        for rnd in range(6):
            for arm, tps in zip(cells, (500.0, 750.0, 520.0, 780.0)):
                flat.append({"round": rnd, "arm": arm, "ncmoe": 24, "verdict": "OK",
                             "gen_tps": 80.0, "prompt_tps": tps, "host_recovered": True})
        buf = io.StringIO()
        with redirect_stdout(buf):
            report(flat)
        assert "no interaction the instrument can see" in buf.getvalue()

        print("\nselfcheck OK (a zero-effect fork must read as 'inside noise'; "
              "a +50%/-10% split must read as a REAL interaction)")
        raise SystemExit(0)
    raise SystemExit(main())
