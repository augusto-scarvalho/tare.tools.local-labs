"""Fully-resident envelope: what fits in VRAM+RAM with NO swap, and what Windows keeps.

The question this answers is not "how fast" but "what can be run at all without the
workstation paying for it". Three numbers per configuration, and the third is the one that
has never been measured here:

    VRAM free      - from nvidia-smi, the usual
    WSL RAM free   - the VM's own view
    WINDOWS free   - what the workstation actually has left

The third matters because the first two cannot see it. WSL2 takes pages from the host and,
with `autoMemoryReclaim=gradual`, gives them back slowly; `free` inside the distro reports
the VM's allocation, not the host's remainder. A configuration that looks comfortable from
inside can leave Windows starved.

NO SWAP is the point, not a detail. With `swap=0` in .wslconfig a configuration that does
not fit is OOM-killed inside the VM instead of quietly thrashing an SSD for hours. That
converts "it seemed to work" into a binary, contained, observable outcome -- and the
`memory=` cap is what contains it, so the failure never reaches the workstation.

    python residency_sweep.py --model qwen35-122b --selfcheck
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from model_lifecycle.collectors.host import sample                 # noqa: E402
from model_lifecycle.servers.llama_cpp import (                    # noqa: E402
    LlamaCppAdapter, ServerProfile)

LOCAL_BIN = "/home/augus/src/llama.cpp-local/build/bin/llama-server"

# key -> (gguf, block_count, approx GiB on disk). The big two are Q3_K_M in three shards;
# llama.cpp takes the first shard's path and finds the rest.
MODELS = {
    "qwen35-122b": ("/home/augus/models/llama-cache/models--unsloth--Qwen3.5-122B-A10B-GGUF/"
                    "snapshots/*/Qwen3.5-122B-A10B-Q3_K_M-00001-of-00003.gguf", 0, 53),
    # nemotron-120b DISCARDED 2026-07-31: no quant fits the envelope (measured -- IQ1_S loads
    # only at ncmoe=50 with 594 MB VRAM / 2.0 GB Windows free, both inside the reserves). See
    # models.py for the full note; files deleted.
    # laguna-s discarded 2026-08-01 (pinning needs mmap needs file<=26GB; Q2_K_XL is 39.7 GB).
    "qwen36-35b": ("/home/augus/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", 40, 20),
}

# Descending: the first context that loads is the ceiling. Cheaper than a binary search
# because a miss costs a failed load and a hit costs the same load you wanted anyway.
CONTEXTS = [131072, 65536, 32768, 16384, 8192, 4096]


def windows_pagefile_gb() -> tuple[float | None, float | None]:
    """(current usage, peak usage) of the Windows pagefile, in GiB.

    THE number this sweep exists to drive to zero. The owner's constraint is SSD wear, and
    wear comes from the sum of WSL and Windows exceeding physical RAM -- not from the
    pagefile existing. Keeping a small pagefile and PROVING it is never touched gives the
    same outcome as deleting it, with a safety net instead of an application crash when
    something spikes. Measured before and after every configuration so the proof is per
    configuration rather than a single reassuring reading at the end.
    """
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "$p=Get-CimInstance Win32_PageFileUsage; "
             "if($p){\"$($p.CurrentUsage) $($p.PeakUsage)\"}else{'0 0'}"],
            capture_output=True, text=True, timeout=60)
        cur, peak = out.stdout.split()
        return int(cur) / 1024.0, int(peak) / 1024.0
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, None


def windows_free_gb() -> float | None:
    """Free physical RAM as WINDOWS sees it. Not `free` inside the distro: that reports the
    VM's own allocation and is blind to what the host has left, which is the entire
    question here."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            capture_output=True, text=True, timeout=60)
        return int(out.stdout.strip()) / 1024.0 / 1024.0
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def wsl_mem_gb() -> tuple[float | None, float | None]:
    """(total, available) inside the distro, in GiB."""
    try:
        out = subprocess.run(["wsl", "-d", "Ubuntu-24.04", "--", "cat", "/proc/meminfo"],
                             capture_output=True, text=True, timeout=60)
        vals = {}
        for line in out.stdout.splitlines():
            k, _, rest = line.partition(":")
            if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                vals[k] = int(rest.strip().split()[0]) / 1024.0 / 1024.0
        return vals.get("MemTotal"), vals.get("MemAvailable")
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, None


def swap_used_gb() -> float | None:
    """Swap actually consumed. The whole design assumes zero; if this is non-zero the run
    is measuring a swapping configuration and its numbers describe the SSD."""
    try:
        out = subprocess.run(["wsl", "-d", "Ubuntu-24.04", "--", "cat", "/proc/meminfo"],
                             capture_output=True, text=True, timeout=60)
        tot = free = None
        for line in out.stdout.splitlines():
            if line.startswith("SwapTotal:"):
                tot = int(line.split()[1]) / 1024.0 / 1024.0
            elif line.startswith("SwapFree:"):
                free = int(line.split()[1]) / 1024.0 / 1024.0
        if tot is None or free is None:
            return None
        return tot - free
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def resolve(pattern: str) -> str | None:
    """Locate a GGUF in the HF cache by FILENAME, searching the tree.

    The first version globbed `snapshots/*/<file>` and silently matched nothing: the real
    layout carries an extra quant directory, `snapshots/<hash>/Q3_K_M/<file>`. Every
    configuration for both large models was then reported as "did not fit" when not one of
    them had been attempted -- 24 rows of a conclusion drawn from a path bug.

    `find` by basename is layout-independent, which a glob written from a guess is not.
    """
    if "*" not in pattern:
        return pattern
    root, _, name = pattern.rpartition("/")
    root = root.split("*")[0].rstrip("/")
    try:
        out = subprocess.run(
            ["wsl", "-d", "Ubuntu-24.04", "--", "bash", "-lc",
             f"find {root} -name {name!r} -print -quit 2>/dev/null"],
            capture_output=True, text=True, timeout=120)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def probe(model_key: str, ncmoe: int, ctx: int, *, settle_s: float,
          gguf_override: str | None = None) -> dict:
    """One load attempt. Returns a record whether it succeeded or not: a configuration that
    does NOT fit is the primary result of this sweep, not an error to be swallowed."""
    gguf_pattern, _blocks, _gib = MODELS[model_key]
    gguf = gguf_override or resolve(gguf_pattern)
    rec = {"model": model_key, "ncmoe": ncmoe, "ctx": ctx, "loaded": False}
    if not gguf:
        # NOT the same as "did not fit", and the distinction is the whole point of the
        # sweep. Flagged so the report can refuse to draw an envelope conclusion from a
        # file it never opened.
        rec["error"] = "gguf not found"
        rec["attempted"] = False
        return rec
    rec["attempted"] = True
    rec["gguf"] = gguf

    before_win = windows_free_gb()
    pf_before, _ = windows_pagefile_gb()
    adapter = LlamaCppAdapter(server_bin=LOCAL_BIN,
                              env={"GGML_CUDA_REGISTER_HOST": "1"})
    profile = ServerProfile(model_path=gguf, port=8080, n_cpu_moe=ncmoe, ctx_size=ctx,
                            cache_type_k="q8_0", cache_type_v="q8_0")
    h = adapter.start(profile)
    try:
        ok = adapter.wait_until_healthy(h, timeout_s=1800)   # 53 GB loads are slow
        rec["loaded"] = ok
        rec["load_seconds"] = h.load_seconds
        if not ok:
            rec["stderr_tail"] = h.stderr_tail[-8:]
            return rec
        s = sample()
        tot, avail = wsl_mem_gb()
        rec.update({
            "vram_free_mb": s.vram_free_mb,
            # The GUARD's own metric (Windows AvailableMBytes, incl. reclaimable standby),
            # which is looser than windows_free below. This is what the A/B actually enforces.
            "windows_available_mb_guard": s.ram_available_mb,
            "wsl_ram_total_gb": tot,
            "wsl_ram_avail_gb": avail,
            "windows_free_gb_before": before_win,
            "windows_free_gb_after": windows_free_gb(),
            # Non-zero swap invalidates the run's premise. Recorded, never assumed away.
            "swap_used_gb": swap_used_gb(),
            "pagefile_gb_before": pf_before,
            "pagefile_gb_after": windows_pagefile_gb()[0],
        })
        rec["pagefile_delta_gb"] = (
            None if rec["pagefile_gb_after"] is None or pf_before is None
            else rec["pagefile_gb_after"] - pf_before)
        return rec
    finally:
        adapter.stop(h)
        adapter.force_stop(h)
        # `autoMemoryReclaim=gradual` returns pages slowly -- measured at ~14.7 GB in 45 s.
        # Without this wait the NEXT configuration starts inside a shrunken envelope and
        # its failure is about the previous run, not about itself.
        time.sleep(settle_s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--gguf", help="override the model file; keep --model's geometry "
                                   "(e.g. Nemotron IQ1_S, not the arch-default Q3)")
    ap.add_argument("--ctx", type=int, help="probe only this context, not the full "
                                            "CONTEXTS sweep (match the A/B's ctx)")
    ap.add_argument("--ncmoe", type=int, action="append", required=True,
                    help="offload levels to try, repeatable")
    ap.add_argument("--settle", type=float, default=90.0,
                    help="seconds to wait for WSL to return memory between loads")
    ap.add_argument("--min-windows-gb", type=float, default=16.0,
                    help="abort the sweep once a configuration leaves Windows below this")
    args = ap.parse_args()

    print(f"model={args.model}  ncmoe={args.ncmoe}  contexts={CONTEXTS}", flush=True)
    tot, avail = wsl_mem_gb()
    print(f"WSL: total {tot:.1f} GiB, available {avail:.1f} GiB | "
          f"Windows free {windows_free_gb():.1f} GiB | swap used "
          f"{swap_used_gb():.2f} GiB", flush=True)

    records: list[dict] = []
    for ncmoe in args.ncmoe:
        # STOP the whole sweep once the host is in trouble. Without this the sweep walked
        # the Nemotron down to ncmoe=45, leaving Windows 1.4 GiB free and growing the
        # pagefile by 1.4 GiB -- it found the limit by crossing it. Loading is not the only
        # failure that matters; a configuration that loads while the workstation pages has
        # already lost by this project's own first line.
        prev = [r for r in records if r.get("loaded")]
        if prev:
            last = prev[-1]
            w = last.get("windows_free_gb_after")
            if w is not None and w < args.min_windows_gb:
                print(f"  STOPPING SWEEP: last configuration left Windows {w:.1f} GiB, "
                      f"below the {args.min_windows_gb:.0f} GiB floor.", flush=True)
                break
            if (last.get("pagefile_delta_gb") or 0) > 0.05:
                print(f"  STOPPING SWEEP: last configuration grew the pagefile by "
                      f"{last['pagefile_delta_gb']:.2f} GiB.", flush=True)
                break
        # Descending contexts: stop at the first that loads. That IS the ceiling for this
        # offload level, and every larger context has already been shown not to fit.
        for ctx in ([args.ctx] if args.ctx else CONTEXTS):
            r = probe(args.model, ncmoe, ctx, settle_s=args.settle,
                      gguf_override=args.gguf)
            records.append(r)
            if r["loaded"]:
                print(f"  ncmoe={ncmoe:<3} ctx={ctx:<7} LOADED  "
                      f"vram_free={r['vram_free_mb']}MB  "
                      f"wsl_avail={r['wsl_ram_avail_gb']:.1f}G  "
                      f"WINDOWS_FREE={r['windows_free_gb_after']:.1f}G  "
                      f"swap={r['swap_used_gb']:.2f}G", flush=True)
                break
            elif r.get("attempted") is False:
                print(f"  ncmoe={ncmoe:<3} ctx={ctx:<7} NOT ATTEMPTED: "
                      f"{r.get('error')}", flush=True)
                break                      # the file is missing; more contexts is noise
            else:
                print(f"  ncmoe={ncmoe:<3} ctx={ctx:<7} did not fit", flush=True)

    out = pathlib.Path(__file__).parent / "runs"
    out.mkdir(exist_ok=True)
    (out / f"residency_{args.model}.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")
    report(records)
    return 0


def report(records: list[dict]) -> None:
    print("\n" + "=" * 74)
    print("FULLY-RESIDENT ENVELOPE  (loaded = fits in VRAM+RAM; swap must read 0.00)")
    print("=" * 74)
    # "Never attempted" must never be read as "did not fit". A missing file produced 24
    # rows of "did not fit" once, and an envelope conclusion drawn from a path bug is
    # worse than no conclusion.
    missing = [r for r in records if r.get("attempted") is False]
    if missing:
        print(f"  !! {len(missing)} configuration(s) were NOT ATTEMPTED "
              f"({missing[0].get('error')}).\n     Nothing about the envelope follows "
              f"from them.")
    fits = [r for r in records if r.get("loaded")]
    if not fits:
        tried = [r for r in records if r.get("attempted")]
        if tried:
            print(f"  nothing loaded in {len(tried)} genuine attempt(s).")
        else:
            print("  NO CONFIGURATION WAS EVER ATTEMPTED - this is a harness failure, "
                  "not a result.")
        return
    print(f"  {'ncmoe':>5} {'max ctx':>9} {'VRAM free':>10} {'WSL avail':>10} "
          f"{'WINDOWS free':>13} {'swap':>7} {'pagefile':>9}")
    for r in fits:
        pfd = r.get("pagefile_delta_gb")
        pfs = f"{pfd:+6.2f} G" if pfd is not None else "     n/a"
        print(f"  {r['ncmoe']:>5} {r['ctx']:>9} {r['vram_free_mb']:>8} MB "
              f"{r['wsl_ram_avail_gb']:>8.1f} G {r['windows_free_gb_after']:>11.1f} G "
              f"{r['swap_used_gb']:>6.2f} G {pfs}")
    swapping = [r for r in fits if (r.get("swap_used_gb") or 0) > 0.05]
    if swapping:
        print(f"\n  !! {len(swapping)} configuration(s) used swap. Their numbers describe "
              f"the SSD,\n     not residency, and the premise of this sweep does not hold "
              f"for them.")
    # The owner's actual constraint is SSD wear, and the pagefile is where it would show.
    # A configuration that grew it did NOT run fully resident, however healthy the rest of
    # its numbers look.
    paged = [r for r in fits if (r.get("pagefile_delta_gb") or 0) > 0.05]
    if paged:
        print(f"\n  !! {len(paged)} configuration(s) GREW the Windows pagefile:")
        for r in paged:
            print(f"       ncmoe={r['ncmoe']} ctx={r['ctx']}: "
                  f"+{r['pagefile_delta_gb']:.2f} GiB written")
        print("     Those are not fully resident. The SSD paid for them.")
    else:
        print("\n  Windows pagefile did not grow in any loaded configuration: the small "
              "pagefile\n  is insurance that is never drawn on, which is the outcome that "
              "was asked for.")
    worst = min(fits, key=lambda r: r.get("windows_free_gb_after") or 999)
    print(f"\n  Tightest for the workstation: ncmoe={worst['ncmoe']} ctx={worst['ctx']} "
          f"leaves Windows {worst['windows_free_gb_after']:.1f} GiB")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        assert CONTEXTS == sorted(CONTEXTS, reverse=True), \
            "contexts must descend: the first that loads is the ceiling"
        import io
        from contextlib import redirect_stdout

        fake = [
            {"model": "m", "ncmoe": 20, "ctx": 65536, "loaded": True, "vram_free_mb": 1800,
             "wsl_ram_avail_gb": 3.2, "windows_free_gb_after": 22.4, "swap_used_gb": 0.0,
             "pagefile_delta_gb": 0.0},
            {"model": "m", "ncmoe": 30, "ctx": 131072, "loaded": True, "vram_free_mb": 9100,
             "wsl_ram_avail_gb": 1.1, "windows_free_gb_after": 18.9, "swap_used_gb": 1.7,
             "pagefile_delta_gb": 2.4},
            {"model": "m", "ncmoe": 10, "ctx": 4096, "loaded": False, "attempted": True},
            {"model": "m", "ncmoe": 99, "ctx": 4096, "loaded": False,
             "attempted": False, "error": "gguf not found"},
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            report(fake)
        out = buf.getvalue()
        print(out)
        # Both failure modes must be named, not averaged into a healthy-looking table.
        assert "used swap" in out
        assert "GREW the Windows pagefile" in out
        assert "Tightest for the workstation" in out
        assert "NOT ATTEMPTED" in out, "a missing file must never read as 'did not fit'"

        # And an all-missing run must be called a harness failure, not an empty envelope.
        buf3 = io.StringIO()
        with redirect_stdout(buf3):
            report([{"model": "m", "ncmoe": 99, "ctx": 4096, "loaded": False,
                     "attempted": False, "error": "gguf not found"}])
        assert "NO CONFIGURATION WAS EVER ATTEMPTED" in buf3.getvalue()

        # And the clean case must state the positive result explicitly, because "no
        # warning printed" is indistinguishable from "the check did not run".
        clean = [r for r in fake if r.get("loaded") and not r.get("pagefile_delta_gb")]
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            report(clean)
        assert "did not grow in any loaded configuration" in buf2.getvalue()
        print("\nresidency_sweep self-check OK")
        raise SystemExit(0)
    raise SystemExit(main())
