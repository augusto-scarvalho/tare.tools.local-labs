"""Find the fastest configuration that does NOT violate the host envelope.

Two responses in tension and one hard constraint. The 122B loads at `ncmoe` 80/99 with
zero swap — and leaves Windows 6.9 GiB against a declared floor of 16, while **14.7 GiB of
VRAM sits idle**. That is not a tuning subtlety, it is a configuration nobody should ship:
a third of the card unused while the workstation starves.

The decision rule is **lexicographic**, per the harness research playbook, and deliberately
not a weighted score:

    1. HARD CONSTRAINTS, first and absolutely — it loads, swap is zero, the Windows
       pagefile did not grow, and Windows keeps at least the declared floor.
    2. Among the feasible, and only among them, maximise throughput.

A weighted objective would let a fast configuration buy its way out of a violated envelope
by being fast enough. It cannot: *"escaping resources does not buy a violated hard
constraint."* The whole project's first line says a configuration that wins on tokens/s and
takes the desktop down is a NEGATIVE result, and this file is where that stops being a
slogan.

The full frontier is printed, not just the winner, because the trade is the interesting
part and the floor is a policy the owner may want to move knowingly.

    python optimize_config.py --model qwen35-122b --ncmoe 45 --ncmoe 55 --ncmoe 65
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from model_lifecycle.analysis.statistics import describe            # noqa: E402
from model_lifecycle.collectors.host import sample                  # noqa: E402
from model_lifecycle.collectors.request import chat_stream          # noqa: E402
from model_lifecycle.servers.llama_cpp import (                     # noqa: E402
    LlamaCppAdapter, ServerProfile)
from residency_sweep import (                                       # noqa: E402
    MODELS, resolve, windows_free_gb, windows_pagefile_gb, swap_used_gb)

LOCAL_BIN = "/home/augus/src/llama.cpp-local/build/bin/llama-server"

# Pinning ON, prefetch OFF. This is not a default, it is the configuration this project
# measured across three expert geometries: pinning is worth +105% to +123% prefill and the
# prefetch costs 8.9% to 22.3% in every one. Optimising with the prefetch enabled would be
# optimising a configuration we have already shown to be dominated.
ENV = {"GGML_CUDA_REGISTER_HOST": "1"}

_UNIT = ("The scheduler assigns each operation to the backend that owns its weights, so a "
         "tensor living in system memory pulls its computation onto the host unless the "
         "graph explicitly uploads it first. ")


def _prompt(approx_tokens: int) -> str:
    words = _UNIT.split()
    need = max(1, int(approx_tokens / 1.3))
    return " ".join(words[i % len(words)] for i in range(need))


def probe(model_key: str, ncmoe: int, ctx: int, *, reps: int, settle_s: float,
          no_mmap: bool = False) -> dict:
    """One configuration, measured.

    `no_mmap` is a FACTOR here, not a detail, and the measurement that made it one:
    dropping `ncmoe` from 99 to 45 on the 122B moved Windows free RAM by **zero** —
    6.9 GiB at every level. With mmap the whole file is mapped and its touched pages stay
    resident; our own `cudaHostRegister` then LOCKS them, so they cannot be evicted. Host
    RAM tracks the FILE, not the offload split, and `ncmoe` is not the knob that frees the
    workstation. `--no-mmap` allocates only the buffers actually needed, so GPU-resident
    tensors stop costing host RAM — and it is also the path our gate makes fast.

    Sweeping ncmoe alone would have spent hours confirming a knob that does not move the
    constraint.
    """
    gguf = resolve(MODELS[model_key][0])
    rec: dict = {"model": model_key, "ncmoe": ncmoe, "ctx": ctx, "no_mmap": no_mmap,
                 "loaded": False, "attempted": bool(gguf)}
    if not gguf:
        rec["error"] = "gguf not found"
        return rec

    pf_before, _ = windows_pagefile_gb()
    adapter = LlamaCppAdapter(server_bin=LOCAL_BIN, env=dict(ENV))
    profile = ServerProfile(model_path=gguf, port=8080, n_cpu_moe=ncmoe, ctx_size=ctx,
                            cache_type_k="q8_0", cache_type_v="q8_0", no_mmap=no_mmap)
    h = adapter.start(profile)
    try:
        if not adapter.wait_until_healthy(h, timeout_s=2400):
            rec["stderr_tail"] = h.stderr_tail[-8:]
            return rec
        rec["loaded"] = True
        rec["load_seconds"] = h.load_seconds

        pre, gen = [], []
        body = _prompt(4000)
        for i in range(reps):
            r = chat_stream(h.base_url, f"[opt-{ncmoe}-{i}] {body}\n\nExplain briefly:",
                            max_tokens=256, cache_prompt=False)
            if r.prompt_tps:
                pre.append(r.prompt_tps)
            if r.generation_tps and not r.generation_tps_is_lower_bound:
                gen.append(r.generation_tps)
        s = sample()
        rec.update({
            "prefill_tps": describe(pre).mean if pre else None,
            "gen_tps": describe(gen).mean if gen else None,
            "vram_free_mb": s.vram_free_mb,
            "windows_free_gb": windows_free_gb(),
            "swap_used_gb": swap_used_gb(),
            "pagefile_delta_gb": (None if pf_before is None
                                  else (windows_pagefile_gb()[0] or 0) - pf_before),
        })
        return rec
    finally:
        adapter.stop(h)
        adapter.force_stop(h)
        time.sleep(settle_s)


def fill_vram_first(model_key: str, ladder: list[int], ctx: int, *, reps: int,
                    settle_s: float, no_mmap: bool, vram_floor_mb: int,
                    win_floor_gb: float) -> list[dict]:
    """Descend `ncmoe` until VRAM hits its declared floor. The last good rung is the answer.

    This is a POLICY, not a search over a space of equals: **fill the fast memory first,
    then offload the remainder.** A byte in VRAM is a byte that never crosses PCIe, so a
    configuration that offloads to RAM while leaving 14.7 GiB of card idle is not one
    option among several — it is waste, at every point of the space.

    So the question is not "which ncmoe is best" but "how low can ncmoe go before VRAM
    reaches the reserve", and that is a boundary, found by descending until the floor is
    touched. It costs a handful of loads instead of a sweep, and what it returns is the
    rule rather than a sample of it.

    Descends because more offload means LESS on the card: ncmoe=99 puts every expert layer
    on the CPU, ncmoe=0 puts none there.
    """
    out: list[dict] = []
    for ncmoe in sorted(ladder, reverse=True):
        r = probe(model_key, ncmoe, ctx, reps=reps, settle_s=settle_s, no_mmap=no_mmap)
        out.append(r)
        tag = "no-mmap" if no_mmap else "mmap   "
        if not r["loaded"]:
            print(f"  ncmoe={ncmoe:<3} {tag} did not load -- floor crossed, stopping",
                  flush=True)
            break
        vram = r.get("vram_free_mb") or 0
        win = r.get("windows_free_gb")
        print(f"  ncmoe={ncmoe:<3} {tag} vram_free={vram}MB win_free={win} "
              f"prefill={r.get('prefill_tps')}", flush=True)
        if vram < vram_floor_mb:
            print(f"  ncmoe={ncmoe:<3} {tag} VRAM below the {vram_floor_mb} MB reserve "
                  f"-- stopping; the previous rung is the pick", flush=True)
            break

        # ABORT on a host-floor breach. The first version said "continuing to see whether
        # more VRAM recovers it" -- a guess, and a wrong one. Measured on the Nemotron
        # under mmap, descending made the host WORSE at every step: Windows free went
        # 6.6 -> 2.7 -> 1.4 GiB, and at 1.4 the pagefile grew by 1.4 GiB. Putting more on
        # the card allocates more device buffers and more CUDA host-side staging, on top
        # of a file that stays mapped either way.
        #
        # The descent walked the workstation into paging because this branch chose to keep
        # going on a hypothesis. It stops now, and the reason is a measurement.
        if win is not None and win < win_floor_gb:
            print(f"  ncmoe={ncmoe:<3} {tag} STOPPING: Windows at {win:.1f} GiB, below the "
                  f"{win_floor_gb:.0f} GiB floor.\n            Descending further has been "
                  f"measured to make this worse, not better.", flush=True)
            break

        # And stop if the SSD started paying, whatever the free-RAM reading says.
        if (r.get("pagefile_delta_gb") or 0) > 0.05:
            print(f"  ncmoe={ncmoe:<3} {tag} STOPPING: the Windows pagefile grew by "
                  f"{r['pagefile_delta_gb']:.2f} GiB.\n            That is the SSD wear "
                  f"this envelope exists to prevent.", flush=True)
            break
    return out


def feasible(r: dict, floor_gb: float) -> tuple[bool, str]:
    """Hard constraints, checked in order, with the FIRST violation named.

    Returning the reason matters: 'infeasible' with no cause sends the next reader tuning
    the wrong knob."""
    if not r.get("loaded"):
        return False, "did not load"
    if (r.get("swap_used_gb") or 0) > 0.05:
        return False, f"used {r['swap_used_gb']:.2f} GiB of swap"
    if (r.get("pagefile_delta_gb") or 0) > 0.05:
        return False, f"grew the pagefile by {r['pagefile_delta_gb']:.2f} GiB"
    w = r.get("windows_free_gb")
    if w is None:
        return False, "Windows free RAM unknown"
    if w < floor_gb:
        return False, f"left Windows {w:.1f} GiB, floor is {floor_gb:.0f}"
    return True, ""


def report(records: list[dict], floor_gb: float) -> None:
    print("\n" + "=" * 84)
    print(f"CONFIGURATION FRONTIER  (hard floor: Windows keeps >= {floor_gb:.0f} GiB, "
          f"zero swap, pagefile untouched)")
    print("=" * 84)
    print(f"  {'ncmoe':>5} {'mmap':>7} {'prefill':>9} {'gen':>8} {'VRAM free':>10} "
          f"{'Win free':>9}  verdict")
    ok = []
    for r in sorted(records, key=lambda x: (x.get("no_mmap", False), x["ncmoe"])):
        good, why = feasible(r, floor_gb)
        if good:
            ok.append(r)
        pre = f"{r['prefill_tps']:8.1f}" if r.get("prefill_tps") else "       -"
        gen = f"{r['gen_tps']:7.1f}" if r.get("gen_tps") else "      -"
        vram = f"{r['vram_free_mb']:8}" if r.get("vram_free_mb") is not None else "       -"
        win = f"{r['windows_free_gb']:7.1f}" if r.get("windows_free_gb") else "      -"
        mm = "no-mmap" if r.get("no_mmap") else "mmap"
        print(f"  {r['ncmoe']:>5} {mm:>7} {pre} {gen} {vram} MB {win} G  "
              f"{'FEASIBLE' if good else 'rejected: ' + why}")

    # Is ncmoe even a lever on host RAM? Measured on the 122B it was not: Windows sat at
    # 6.9 GiB from ncmoe 45 through 99. Say so explicitly rather than leaving the next
    # reader to sweep it again.
    for mm in (False, True):
        grp = [r for r in records if r.get("no_mmap") == mm and r.get("windows_free_gb")]
        if len(grp) >= 2:
            spread = max(r["windows_free_gb"] for r in grp) - min(
                r["windows_free_gb"] for r in grp)
            if spread < 1.0:
                print(f"\n  ncmoe moved Windows free RAM by only {spread:.1f} GiB across "
                      f"{len(grp)} levels ({'no-mmap' if mm else 'mmap'}).\n  It is not the "
                      f"knob that frees the host here.")

    if not ok:
        print(f"\n  NO configuration satisfies the envelope. This is a real answer: at this "
              f"quant\n  and context the model does not belong on this host. Lower the "
              f"context, use a\n  smaller quant, or move the floor DELIBERATELY -- not by "
              f"averaging it away.")
        return

    # The policy, stated against the data: fill VRAM to the reserve, then offload. The
    # feasible configuration with the LEAST idle VRAM is the one that honours it.
    tightest_vram = min(ok, key=lambda r: r.get("vram_free_mb") or 10**9)
    print(f"\n  VRAM-FILL POLICY: the feasible rung that puts the most on the card is "
          f"ncmoe={tightest_vram['ncmoe']}\n  ({'no-mmap' if tightest_vram.get('no_mmap') else 'mmap'}), "
          f"leaving {tightest_vram.get('vram_free_mb')} MB of VRAM in reserve.")

    # Lexicographic: among the feasible ONLY, maximise throughput. Prefill first, because
    # this project measured agentic workloads to be prefill-bound (~800k prefilled against
    # ~20k generated per SWE-bench instance).
    best = max(ok, key=lambda r: (r.get("prefill_tps") or 0, r.get("gen_tps") or 0))
    print(f"\n  PICK: ncmoe={best['ncmoe']} ctx={best['ctx']} -- "
          f"{best.get('prefill_tps', 0):.1f} t/s prefill, {best.get('gen_tps', 0):.1f} t/s "
          f"generation,\n        leaving Windows {best['windows_free_gb']:.1f} GiB and "
          f"{best['vram_free_mb']} MB of VRAM unused.")

    idle = [r for r in ok if (r.get("vram_free_mb") or 0) > 4000]
    if idle:
        print(f"\n  {len(idle)} feasible configuration(s) leave >4 GB of VRAM idle. VRAM "
              f"held in reserve\n  is RAM taken from Windows for no return -- lower ncmoe "
              f"moves weights onto the card\n  and gives the workstation its memory back.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--ncmoe", type=int, action="append", required=True)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--min-windows-gb", type=float, default=16.0,
                    help="the declared envelope floor; a policy, not a tuning parameter")
    ap.add_argument("--settle", type=float, default=120.0)
    ap.add_argument("--mmap", choices=("both", "on", "off"), default="both",
                    help="mmap is a FACTOR: ncmoe was measured NOT to move host RAM")
    ap.add_argument("--min-vram-mb", type=int, default=4096,
                    help="the declared VRAM reserve; descent stops when it is reached")
    args = ap.parse_args()

    modes = {"both": [False, True], "on": [False], "off": [True]}[args.mmap]
    print(f"model={args.model} ctx={args.ctx} floors: Windows {args.min_windows_gb} GiB / "
          f"VRAM {args.min_vram_mb} MB  ncmoe ladder={sorted(args.ncmoe, reverse=True)} "
          f"mmap={args.mmap}", flush=True)
    records = []
    for no_mmap in modes:
        # Fill VRAM to the reserve, THEN offload the remainder. Descending stops at the
        # floor instead of walking the whole ladder, so this costs a few loads rather
        # than a sweep.
        records += fill_vram_first(args.model, args.ncmoe, args.ctx, reps=args.reps,
                                   settle_s=args.settle, no_mmap=no_mmap,
                                   vram_floor_mb=args.min_vram_mb,
                                   win_floor_gb=args.min_windows_gb)

    out = pathlib.Path(__file__).parent / "runs"
    out.mkdir(exist_ok=True)
    (out / f"optimize_{args.model}_ctx{args.ctx}.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")
    report(records, args.min_windows_gb)
    return 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        floor = 16.0
        base = {"loaded": True, "swap_used_gb": 0.0, "pagefile_delta_gb": 0.0,
                "windows_free_gb": 20.0, "ncmoe": 40, "ctx": 32768,
                "prefill_tps": 300.0, "gen_tps": 30.0, "vram_free_mb": 1000}
        assert feasible(base, floor)[0]

        # Each hard constraint must reject on its own, and NAME itself.
        for key, val, needle in (("loaded", False, "did not load"),
                                 ("swap_used_gb", 1.0, "swap"),
                                 ("pagefile_delta_gb", 0.5, "pagefile"),
                                 ("windows_free_gb", 7.0, "floor is 16")):
            bad = dict(base); bad[key] = val
            good, why = feasible(bad, floor)
            assert not good and needle in why, (key, why)

        # THE property that makes this lexicographic: a much faster configuration that
        # violates the envelope must never be picked over a slower one that respects it.
        import io
        from contextlib import redirect_stdout
        fast_but_bad = dict(base, ncmoe=99, prefill_tps=9999.0, windows_free_gb=6.9)
        slow_but_ok = dict(base, ncmoe=45, prefill_tps=250.0, windows_free_gb=18.0)
        buf = io.StringIO()
        with redirect_stdout(buf):
            report([fast_but_bad, slow_but_ok], floor)
        out = buf.getvalue()
        assert "PICK: ncmoe=45" in out, out
        assert "9999" not in out.split("PICK")[1], "the violating config must not be picked"

        # No feasible option is a RESULT, phrased as one.
        buf = io.StringIO()
        with redirect_stdout(buf):
            report([fast_but_bad], floor)
        assert "NO configuration satisfies the envelope" in buf.getvalue()

        # Idle VRAM beside a starved host is the finding that motivated the file.
        buf = io.StringIO()
        with redirect_stdout(buf):
            report([dict(slow_but_ok, vram_free_mb=14707)], floor)
        assert "idle" in buf.getvalue()

        # THE policy: among feasible rungs, the one that puts the most on the card wins
        # the VRAM-fill statement -- a byte in VRAM never crosses PCIe, so idle card
        # beside offloaded weight is waste at every point of the space, not a trade.
        buf = io.StringIO()
        with redirect_stdout(buf):
            report([dict(slow_but_ok, ncmoe=60, vram_free_mb=14707),
                    dict(slow_but_ok, ncmoe=30, vram_free_mb=4300)], floor)
        out = buf.getvalue()
        assert "VRAM-FILL POLICY" in out and "ncmoe=30" in out.split("VRAM-FILL")[1], out

        print("optimize_config self-check OK")
        raise SystemExit(0)
    raise SystemExit(main())
