"""Prefill rate against one parameter at a time — four open questions, one instrument.

All four are the same shape (how does prefill respond to X?), so they are one script with
`--axis` rather than four near-identical files. Each axis is a question nobody in the
upstream thread has numbers for; see UPSTREAM-LANDSCAPE.md for who is stuck on what.

    --axis ubatch     512 / 1024 / 2048
        `thecodacus` measured at pp2048/ub2048; every number in this project so far used
        llama.cpp's default ub=512. The mechanism hides uploads behind compute, so a
        bigger micro-batch means more compute to hide behind. Our +58% may be UNDERSTATED
        and the curve is unpublished.

    --axis mmap       --no-mmap vs pinned mmap
        The exact question PR #21067 is stuck on. Its body says `--no-mmap` is required
        "otherwise the operations are implicitly serialized"; the fork instead pins the
        mmap'd pages. Neither side has posted a head-to-head. If pinned-mmap matches
        --no-mmap, the caveat can be deleted rather than auto-enabled with a warning.

    --axis slots      GGML_SCHED_PREFETCH_EXPERTS = 2 / 3 / 4 / 6 / 8
        The default of 3 is justified by a code comment ("gate/up/down of one MoE layer"),
        not by a measurement. The hard cap is 8. Each slot costs one max-sized expert
        tensor of VRAM, so this is a speed-versus-VRAM curve.

    --axis promptlen  64 / 256 / 1024 / 4096 / 16384 tokens
        The prefetch engages only at `ids >= 2*n_expert` -- 64 tokens for this model.
        Somewhere above that the gain turns on. The threshold is asserted in code and
        never swept, and it decides which real workloads benefit at all.

    --axis gate       prompt lengths straddling THIS model's own gate threshold
        The same question asked so the answer can be attributed. `promptlen` sweeps one
        model, where "the gain turns on near 64 tokens" is indistinguishable from "the
        gain turns on near the length where prefill stops being dominated by overhead".
        Three models with n_expert/n_expert_used of 256/8, 128/8 and 32/4 put the
        threshold at 64, 32 and 16 tokens while the overhead length stays put, so the two
        explanations predict different curves. Run it per model with --model.

Measures PREFILL only, with a generation budget just large enough that the server still
reports its `timings` block. `max_tokens=8` was the original choice and it silently
destroyed two thirds of two model runs: a thinking model spends a tiny budget entirely on
reasoning, the stream ends with `finish_reason=length`, no `timings` chunk arrives, and
`prompt_ms` — the only source of a prefill rate — is never seen. `request.py` warns about
this in its own docstring ("keep it generous, or record the floor") and the warning was
ignored to save wall-clock.

    python prefill_sweep.py --axis ubatch --rounds 2
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from model_lifecycle.analysis.statistics import describe        # noqa: E402
from model_lifecycle.collectors.host import sample              # noqa: E402
from model_lifecycle.collectors.request import chat_stream      # noqa: E402
from model_lifecycle.servers.llama_cpp import (                 # noqa: E402
    LlamaCppAdapter, ServerProfile)

MASTER_BIN = "/home/augus/src/llama.cpp-master/build/bin/llama-server"
REBASE_BIN = "/home/augus/src/llama.cpp-rebase/build/bin/llama-server"
LOCAL_BIN  = "/home/augus/src/llama.cpp-local/build/bin/llama-server"   # our skip-when-pinned

# key -> (gguf, block_count, n_expert, n_expert_used). The last two set this model's
# batch gate: the prefetch engages at `ids >= 2*n_expert`, and ids holds
# n_expert_used*n_tokens entries, so the threshold in TOKENS is 2*n_expert/n_expert_used
# -- 64, 32 and 16 here. Three different thresholds against one unchanged host is what
# makes the gate attributable; one model cannot separate the gate from prefill overhead.
MODELS = {
    "qwen36-35b": ("/home/augus/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                   40, 256, 8),
    "qwen3-30b":  ("/home/augus/models/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf",
                   48, 128, 8),
    "gpt-oss-20b": ("/home/augus/models/gpt-oss-20b/gpt-oss-20b-Q4_K_M.gguf",
                    24, 32, 4),
}
MODEL_KEY = "qwen36-35b"      # replaced by main() from --model

ON = {"GGML_SCHED_PREFETCH_EXPERTS": "3", "GGML_CUDA_REGISTER_HOST": "1"}

# Offload dose as a FRACTION of the model, not a layer count. 24 of 40 was the mid-dose
# for the 40-layer model; the same 24 on a 24-layer model is full offload and on a
# 48-layer model is half, so a fixed integer would have made "model" and "dose" the same
# variable and no comparison across models would mean anything.
NCMOE_FRACTION = 0.6


def _model() -> tuple[str, int, int, int]:
    return MODELS[MODEL_KEY]


def _ncmoe() -> int:
    return max(1, round(_model()[1] * NCMOE_FRACTION))


def gate_tokens(n_expert: int, n_expert_used: int) -> int:
    """Prompt length at which the fork's batch gate opens, from the source condition
    `ids->ne[0]*ids->ne[1] >= 2*n_expert` where ids holds n_expert_used*n_tokens."""
    return max(1, math.ceil(2 * n_expert / n_expert_used))


_UNIT = ("The scheduler assigns each operation to the backend that owns its weights, "
         "so a tensor living in system memory pulls its computation onto the host "
         "unless the graph explicitly uploads it first. ")


def _filler(approx_tokens: int) -> str:
    """Prose of roughly the requested token count. Prose, not a repeated character: a
    repeated token compresses differently and the prompt would not be the size it claims.
    The real count is read back from the server's `prompt_n`, never trusted from here.

    Granularity is WORDS, not sentences. The sentence-sized unit is ~30 tokens, so every
    request below 30 tokens produced the same prompt and the `gate` axis -- whose whole
    point is the two points either side of a 16-token threshold -- would have measured one
    length twice and called it a curve.
    """
    words = _UNIT.split()
    need = max(1, int(approx_tokens / 1.3))      # ~1.3 tokens/word for this prose
    return " ".join(words[i % len(words)] for i in range(need))


# "arm|level" -> (binary, env, profile kwargs, prompt tokens)
#
# The label is TWO fields with a separator, not prose. It used to be free text and the
# report paired arms by string suffix; "fork   mmap+pin" ends with no suffix any master
# label shares, so the single most important row of the mmap axis silently printed "no
# master counterpart" and the comparison the axis exists for was never made. An explicit
# level is one character of syntax and removes the whole class of failure.
def _axis_variants(axis: str) -> list[tuple]:
    if axis == "ubatch":
        out = []
        for ub in (512, 1024, 2048):
            out.append((f"master|ub{ub}", MASTER_BIN, dict(ON), {"ubatch": ub, "batch": 2048}, 16000))
            out.append((f"fork|ub{ub}", REBASE_BIN, dict(ON), {"ubatch": ub, "batch": 2048}, 16000))
        return out
    if axis == "mmap":
        # `local` is the arm this axis now exists for: LOCAL-FORK.md predicts, in writing
        # and before the run, that skipping the prefetch when the buffer is already pinned
        # returns the no-mmap level to master's ~1007 t/s while leaving mmap untouched.
        out = []
        for level, kw in (("mmap", {"no_mmap": False}), ("no-mmap", {"no_mmap": True})):
            out.append((f"master|{level}", MASTER_BIN, dict(ON), dict(kw), 16000))
            out.append((f"fork|{level}",   REBASE_BIN, dict(ON), dict(kw), 16000))
            out.append((f"local|{level}",  LOCAL_BIN,  dict(ON), dict(kw), 16000))
        return out
    if axis == "slots":
        out = [("master|na", MASTER_BIN, dict(ON), {}, 16000)]
        for n in (2, 3, 4, 6, 8):
            out.append((f"fork|slots={n}", REBASE_BIN,
                        {"GGML_SCHED_PREFETCH_EXPERTS": str(n),
                         "GGML_CUDA_REGISTER_HOST": "1"}, {}, 16000))
        return out
    if axis == "promptlen":
        out = []
        for n in (64, 256, 1024, 4096, 16000):
            out.append((f"master|{n}tok", MASTER_BIN, dict(ON), {}, n))
            out.append((f"fork|{n}tok", REBASE_BIN, dict(ON), {}, n))
        return out
    if axis == "gate":
        # Lengths straddling THIS model's threshold, plus fixed anchors far above it so a
        # flat curve can be told apart from a broken run. The two points either side of
        # the gate are the measurement; everything else is context for reading it.
        _, _, n_expert, n_used = _model()
        g = gate_tokens(n_expert, n_used)
        lens = sorted({max(4, g // 4), max(6, g // 2), g - 1, g, g + 1, g * 2, g * 4,
                       256, 1024, 4096})
        out = []
        for n in lens:
            out.append((f"master|{n}tok", MASTER_BIN, dict(ON), {}, n))
            out.append((f"fork|{n}tok", REBASE_BIN, dict(ON), {}, n))
        return out
    raise SystemExit(f"unknown axis {axis}")


def _server_groups(variants: list[tuple]) -> list[dict]:
    """Collapse variants that differ ONLY in prompt length into one server lifetime.

    Prompt length is a property of the REQUEST, not of the server, so restarting between
    lengths bought nothing and cost a model load each time -- 20 loads per round on the
    `gate` axis against 2. It also removed the only part of the comparison that was
    genuinely paired: with one server per length, every length carried its own load-time
    variance. Axes whose levels are server flags (mmap, ubatch, slots) group into
    singletons and behave exactly as before.
    """
    groups: list[dict] = []
    for label, binary, env, kw, ptok in variants:
        key = (binary, tuple(sorted(env.items())), tuple(sorted(kw.items())))
        for g in groups:
            if g["key"] == key:
                g["items"].append((label, ptok))
                break
        else:
            groups.append({"key": key, "binary": binary, "env": env, "kw": kw,
                           "items": [(label, ptok)]})
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True,
                    choices=("ubatch", "mmap", "slots", "promptlen", "gate"))
    ap.add_argument("--model", choices=sorted(MODELS), default="qwen36-35b")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    # 256, not 8. The budget does not need to be big enough for a good ANSWER -- prefill
    # is measured from the server's own timings and is finished before the first token --
    # it needs to be big enough that the request COMPLETES, because a stream cut short at
    # finish_reason=length never delivers the timings block.
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    global MODEL_KEY
    MODEL_KEY = args.model
    gguf, blocks, n_expert, n_used = _model()

    variants = _axis_variants(args.axis)
    print(f"axis={args.axis}  model={args.model} ({blocks}L, {n_expert}/{n_used} experts, "
          f"gate at {gate_tokens(n_expert, n_used)} tok)  ncmoe={_ncmoe()}  "
          f"variants={len(variants)}  rounds={args.rounds}", flush=True)

    groups = _server_groups(variants)
    print(f"  {len(variants)} variants collapse to {len(groups)} server starts per round",
          flush=True)

    records: list[dict] = []
    for rnd in range(args.rounds):
        # Reverse on odd rounds for the same reason the A/B does: the first variant of a
        # sweep pays the cold-cache cost, and a fixed order hands that penalty to the same
        # label every time, where it reads as a property of the parameter.
        order = list(groups) if rnd % 2 == 0 else list(reversed(groups))
        for grp in order:
            adapter = LlamaCppAdapter(server_bin=grp["binary"], env=grp["env"])
            profile = ServerProfile(model_path=gguf, port=8080, n_cpu_moe=_ncmoe(),
                                    ctx_size=32768, cache_type_k="q8_0",
                                    cache_type_v="q8_0", **grp["kw"])
            h = adapter.start(profile)
            try:
                if not adapter.wait_until_healthy(h, timeout_s=900):
                    # PRINT the drained stderr. The adapter has always collected it into
                    # stderr_tail and this script has always thrown it away, so a server
                    # that died on its own command line reported "SERVER NEVER HEALTHY"
                    # and nothing else -- six consecutive failures with no diagnosis and
                    # a live machine sitting idle. A failure that does not say why is a
                    # defect in the instrument, not information about the subject.
                    print(f"  r{rnd} SERVER NEVER HEALTHY. argv:", flush=True)
                    print("    " + " ".join(adapter.argv(profile)), flush=True)
                    for ln in h.stderr_tail[-12:]:
                        print(f"    | {ln}", flush=True)
                    for label, _ in grp["items"]:
                        records.append({"round": rnd, "label": label,
                                        "error": "unhealthy",
                                        "stderr_tail": h.stderr_tail[-12:]})
                    continue
                for label, ptok in grp["items"]:
                    body = _filler(ptok)
                    rates, ns, why, fracs = [], [], [], []
                    for i in range(args.reps):
                        # Distinct prefix per request: the KV cache would otherwise serve
                        # repetitions 2+ and the rate would describe the cache, not
                        # prefill. It matters more now that several prompt lengths share
                        # one server -- the shorter prompt is a literal prefix of the
                        # longer one, so without a distinct first token the longer
                        # request would be served from the shorter one's cache.
                        # cache_prompt=False: the chat template is a common prefix on every
                        # request, so the server reuses it and the measurement stops being
                        # about the length that was asked for. A unique tag at the start of
                        # the USER content cannot prevent that -- the template sits before
                        # it. On gpt-oss the template is ~84 tokens and was being served
                        # from cache while a handful of tokens got processed and timed.
                        r = chat_stream(h.base_url,
                                        f"[{label}-r{rnd}-{i}] {body}\n\nOne word:",
                                        max_tokens=args.max_tokens, cache_prompt=False)
                        # PROPORTIONAL, not binary. `not r.cache_n` rejected any reuse at
                        # all, and llama.cpp reuses the shared chat-template prefix -- 6 to
                        # 12 tokens -- on every request after the first. On a 3372-token
                        # prompt that is 0.4% and irrelevant; on a 38-token prompt it is
                        # 32% and would corrupt the rate. The binary guard threw away every
                        # short level of two model runs while the long ones sailed through,
                        # which is why the failure looked length-dependent and was misread
                        # twice as starvation.
                        #
                        # SETTLED: `cache_n` is counted BESIDE `prompt_n`, not inside it.
                        # The first version divided by prompt_n alone and printed
                        # "cached 222%" -- impossible if prompt_n were the total, and the
                        # answer to a question this file had explicitly written down as
                        # unknown. The gpt-oss chat template is ~84 tokens, all served from
                        # cache, leaving a much smaller processed remainder. Recording the
                        # fraction is what made an out-of-range value visible; asserting a
                        # semantics would have hidden it.
                        cached_frac = ((r.cache_n or 0)
                                       / max(1, (r.cache_n or 0) + (r.prompt_n or 0)))
                        if r.prompt_tps and cached_frac <= 0.02:
                            rates.append(r.prompt_tps)
                            ns.append(r.prompt_n)
                            fracs.append(cached_frac)
                        else:
                            # The STRUCTURAL cause first, and `r.error` only as context.
                            # The first version of this line read `r.error or ...`, and
                            # `r.error` is set on any response with no content -- so every
                            # rejected sample printed the same "starved" text whatever the
                            # actual reason was, and the reason could not be recovered from
                            # the log at all. That is the defect this instrumentation was
                            # added to fix, reproduced inside the fix.
                            cause = (f"cached {cached_frac:.0%} of the prompt "
                                     f"(cache_n={r.cache_n}, prompt_n={r.prompt_n})"
                                     if cached_frac > 0.02 else
                                     f"no prompt_tps (prompt_n={r.prompt_n}, "
                                     f"prompt_ms={r.prompt_ms})")
                            why.append(f"{cause}" + (f" | {r.error}" if r.error else ""))
                    if rates:
                        d = describe(rates)
                        s = sample()
                        records.append({"round": rnd, "label": label, "axis": args.axis,
                                        "model": args.model, "ncmoe": _ncmoe(),
                                        "prompt_n": ns[0], "prefill_tps": d.mean,
                                        "cv": d.cv, "vram_free_mb": s.vram_free_mb,
                                        # Kept so "was any of this served from cache?"
                                        # stays answerable from the record rather than
                                        # from a threshold decided once in code.
                                        "cached_frac_max": max(fracs) if fracs else 0.0})
                        print(f"  r{rnd} {label:<18} n={ns[0]:<6} {d.mean:>8.1f} t/s "
                              f"cv={d.cv:.3f}  vram_free={s.vram_free_mb}MB", flush=True)
                    else:
                        print(f"  r{rnd} {label:<18} NO USABLE SAMPLE: "
                              f"{'; '.join(why[:3])}", flush=True)
                        records.append({"round": rnd, "label": label, "axis": args.axis,
                                        "model": args.model, "error": why[:3]})
            finally:
                adapter.stop(h)
                adapter.force_stop(h)
                time.sleep(8)

    # Namespaced by model as well as axis. `sweep_mmap.json` was already one filename for
    # two different questions once (the ab/ records.json overwrite); a second model would
    # have made the same mistake with the same file.
    out = pathlib.Path(__file__).parent / "runs"
    out.mkdir(exist_ok=True)
    (out / f"sweep_{args.axis}_{args.model}.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")
    report(records, args.axis)
    return 0


def report(records: list[dict], axis: str) -> None:
    print("\n" + "=" * 68)
    print(f"PREFILL vs {axis}")
    print("=" * 68)
    labels = []
    for r in records:
        if r.get("prefill_tps") and r["label"] not in labels:
            labels.append(r["label"])

    stats, prompt_ns = {}, {}
    for lab in labels:
        vals = [r["prefill_tps"] for r in records if r["label"] == lab and r.get("prefill_tps")]
        if vals:
            stats[lab] = describe(vals)
            n = next(r["prompt_n"] for r in records if r["label"] == lab and r.get("prompt_n"))
            prompt_ns[lab] = n
            print(f"  {lab.replace('|', ' '):<20} n={n:<6} {stats[lab].mean:>8.1f} t/s  "
                  f"(rounds={stats[lab].n}, cv={stats[lab].cv:.3f})")

    # Pair by explicit LEVEL, so the table answers "what does this arm buy AT this
    # setting" rather than "which row is biggest" -- the second question is answered by
    # the level, not by the arm. Every non-master arm is paired, not just one named
    # "fork": the local build is an arm too, and the previous suffix-matching version
    # would have dropped it exactly as it dropped `fork mmap+pin`.
    for arm in [a for a in dict.fromkeys(l.split("|")[0] for l in labels) if a != "master"]:
        print(f"\n  -- {arm} vs master, per level --")
        for lab in labels:
            a_name, _, level = lab.partition("|")
            if a_name != arm:
                continue
            mate = f"master|{level}"
            if mate not in stats:
                print(f"  {level:<14} no master counterpart at this level")
                continue
            a, b = stats[mate].mean, stats[lab].mean
            print(f"  {level:<14} master {a:>8.1f} -> {arm} {b:>8.1f} t/s   "
                  f"{(b - a) / a * 100:+6.1f}%")

    # A `gate` axis asks for lengths a chat template may not be able to deliver: the
    # template and the per-request prefix have a floor of their own, so two requested
    # levels can arrive at the server as the same prompt_n. Silently, and the resulting
    # flat pair reads as "no effect at the threshold" -- which is the conclusion the axis
    # exists to test. Say so instead.
    if axis == "gate":
        seen: dict[int, list[str]] = {}
        for lab, n in prompt_ns.items():
            if lab.startswith("master|"):
                seen.setdefault(n, []).append(lab.split("|", 1)[1])
        collided = {n: v for n, v in seen.items() if len(v) > 1}
        if collided:
            print("\n  !! requested levels that arrived as the SAME prompt_n -- the "
                  "template floor, not a\n     measurement. These levels do not resolve:")
            for n, v in sorted(collided.items()):
                print(f"       prompt_n={n}: {', '.join(v)}")
        floor = min(prompt_ns.values()) if prompt_ns else 0
        print(f"  shortest prompt the instrument could actually send: {floor} tokens")

    print("\n  Rates are means over rounds, not paired CIs: this is a SHAPE sweep meant "
          "to locate where\n  the interesting levels are. Confirm any level that matters "
          "with ab_isolate.py, which pairs\n  and reports CI95.")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        for ax in ("ubatch", "mmap", "slots", "promptlen", "gate"):
            v = _axis_variants(ax)
            assert v and all(len(x) == 5 for x in v), ax
            assert any(l.startswith("master|") for l, *_ in v), f"{ax} has no baseline"

        # The gate threshold is the arithmetic the whole `gate` axis rests on.
        assert gate_tokens(256, 8) == 64 and gate_tokens(128, 8) == 32
        assert gate_tokens(32, 4) == 16
        assert _ncmoe() == 24, _ncmoe()                    # 60% of 40 layers

        # EVERY non-master arm must find its master counterpart. This is the regression
        # the suffix-matching version had: `fork mmap+pin` paired with nothing, so the
        # mmap axis printed a comparison table that omitted its own headline row.
        for ax in ("ubatch", "mmap", "promptlen", "gate"):
            levels = {}
            for lab, *_ in _axis_variants(ax):
                arm, _, lv = lab.partition("|")
                levels.setdefault(lv, set()).add(arm)
            for lv, arms in levels.items():
                assert "master" in arms, f"{ax} level {lv} has no master arm: {arms}"

        # Word granularity: two requests one token apart must not produce one prompt.
        assert _filler(15) != _filler(17), "gate axis needs sub-sentence granularity"
        assert 3200 <= len(_filler(1000)) <= 4800, len(_filler(1000))

        # Grouping must collapse prompt lengths and must NOT collapse server flags --
        # merging two different `--no-mmap` settings into one server would silently
        # measure one of them twice.
        assert len(_server_groups(_axis_variants("gate"))) == 2, "gate: one server per arm"
        assert len(_server_groups(_axis_variants("mmap"))) == 6, "mmap: no collapsing"
        assert len(_server_groups(_axis_variants("ubatch"))) == 6
        assert sum(len(g["items"]) for g in _server_groups(_axis_variants("gate"))) == \
            len(_axis_variants("gate")), "grouping must not drop a variant"

        fake = [{"round": r, "label": lab, "axis": "mmap", "prompt_n": 16000,
                 "prefill_tps": tps, "cv": 0.01, "vram_free_mb": 13000}
                for r in (0, 1)
                for lab, tps in (("master|mmap", 487.5), ("fork|mmap", 770.6),
                                 ("local|mmap", 770.0),
                                 ("master|no-mmap", 1007.5), ("fork|no-mmap", 776.6),
                                 ("local|no-mmap", 1005.0))]
        report(fake, "mmap")
        print("\nprefill_sweep self-check OK")
        raise SystemExit(0)
    raise SystemExit(main())
