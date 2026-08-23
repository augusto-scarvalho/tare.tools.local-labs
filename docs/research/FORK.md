# FORK — the consolidated `lifecycle` build and its runtime levers (updated 2026-08-04)

> **Ownership boundary:** [`slop.cpp`](https://github.com/augusto-scarvalho/slop.cpp)
> owns engine implementation, flags, build instructions, and the current
> qualification harness. This document is the historical RTX 3090 experiment
> report: it owns methods, receipts, measured regimes, and promotion decisions.
> Paths, branches, and SHAs below describe the recorded 2026-08-04 tuple unless
> explicitly marked current.

Our own llama.cpp fork: **one binary** that is the pristine upstream baseline by default and gains
every validated-in-some-regime lever the campaign found, each **toggled at runtime** (env var or CLI
flag) — no recompile, and with nothing set the behaviour is byte-identical to upstream `720d7fa40`.

- **Where:** `/home/augus/src/slop.cpp-main`, branch **`lifecycle`**.
- **Base:** upstream **`720d7fa40`** (NOT fresh master — `f5919bf45` regressed `draft-mtp` token-exactness;
  see LANDSCAPE §1c). `GGML_CUDA_GRAPHS=ON`, `FA_ALL_QUANTS=OFF`, `sm_86`.
- **Binary:** `build/bin/llama-server` (+ `build/bin/llama-moe-trace` for the expert-cache profiler).
- **`ab_isolate.py`:** `LIFECYCLE_BIN`.

## The levers — all default OFF, all runtime-toggleable

| Lever | Turn ON with | Default | What it does | Measured verdict / best regime |
|---|---|---|---|---|
| **§B2b KV-host-pin** | env `GGML_KV_PIN_HOST=1` (+ `--no-kv-offload`) | off | pins the KV host buffer → direct-DMA the per-token host→GPU KV copy | **+up to 17%** in the `--no-kv-offload` / long-context (128k, VRAM-starved) regime; ours, novel |
| **Prefetch experts** | flag `--prefetch-experts N` (or env `GGML_SCHED_PREFETCH_EXPERTS=N`, N>0) | off | overlaps offloaded-expert H2D uploads on a 2nd CUDA stream; **skips the staging hop when weights already live in a pinned host buffer** (`--no-mmap`), which was a −22.9% tax there | **+58% prefill** / **−22% decode** — a **prefill / small-card** lever, a decode tax at our ncmoe=6 (§B3). Now visible in `--help` + logged at WARN so it's falsifiable |
| **Pin CPU weights** | flag `--prefetch-pin` (or env `GGML_CUDA_REGISTER_HOST=1`); `--no-prefetch-pin` to force off | off | page-locks mmap'd CPU expert weights for faster H2D | null at the decode optimum; helps **forced-heavy-offload** |
| **MoE expert cache** | flags `--moe-cache-slots N --moe-cache-profile <csv>` | off | keeps the N hottest routed experts resident in VRAM (hot/cold `mul_mat_id`) → hits skip per-token H2D | **null/redundant** vs static `--n-cpu-moe` on Qwen3's load-balanced routing (§E5); for **concentrated-routing** models |
| **GDN chunk-parallel prefill** | env `GGML_CUDA_GDN_CHUNKED=1` (+ `GGML_CUDA_GDN_CHUNKED_MIN_TOKENS`, `GGML_CUDA_GDN_TC`) | off | chunk-parallel + TF32 rewrite of the Gated-DeltaNet prefill scan; engages only at n_tokens ≥ 1024 | **CLOSED lever** (M4b): correct 46/46 @ 2e-7, quality-neutral, but shape-bound — ~parity on the deploy MoE (H=32), −2–4% on H≥48, no concurrent win. Kept opt-in for no-regression; see `GDN_M4_RESUME.md` |

The profile CSV for the expert cache is produced by the bundled profiler:
`MOE_TRACE_OUT=trace.csv build/bin/llama-moe-trace -m MODEL -fa on --n-cpu-moe 6 -p "..." -n 400`.

Upstream levers already in the base (no toggle needed): **placement** (`--n-cpu-moe`), **CUDA graphs**
(on by default), **MTP** (`--spec-type draft-mtp`). These are the actual decode winners; the four above
are the non-upstream additions, each kept because it wins in *some* regime.

## Deploy default (nothing toggled) — unchanged from the validated config
```
llama-server -m qwen36-35b-a3b-mtp -fa on --n-cpu-moe 8 --ctx-size 8192 \
  --cache-type-k q8_0 --cache-type-v q8_0 --spec-type draft-mtp --spec-draft-n-max 4
```
~116 t/s, `draft-mtp` token-exact. See DEPLOY.md.

## Not included: TurboQuant / turbo-MMA decode
Attempted, abandoned by decision. It is not a cherry-pick but **multi-backend surgery** (CPU+CUDA+
Metal+Vulkan: a new `GGML_OP_TURBO_WHT`, 4k-line rotation tables, per-backend quant kernels), it conflicts
on quantization-kernel files (hand-merge = silent-corruption risk), and it only benefits the `turbo2/3/4`
KV format — which we never use and whose decode path measured **null** on our q8_0 (§6). It remains
available in the `stack` build (`--cache-type-k turbo4_0`) if that KV format is ever wanted.

## Blessing (run after any change to the fork)
`SLOP_MODEL=/path/to/mtp-model.gguf bash /home/augus/src/slop.cpp/tools/scripts_sh/bless_fork.sh -- --n-cpu-moe 8`
— G1 §B2b engages · G2 base-vs-`draft-mtp` greedy output identity (#23335) · G3 coherence + `-nkvo`
(#20140). Each ported lever was cherry-picked then re-checked against G2 so the **default path stays
byte-identical to `720d7fa40`** (MTP exact). Rebuild: `cmake --build build --target llama-server -j 20`.

**Historical re-bless 2026-08-04: 3/3 PASS** after folding two lines onto `lifecycle`: the GDN chunk-parallel
kernel (c8761b40c) and the prefetch-skip-pinned improvements (skip-when-pinned + `--prefetch-experts`/
`--prefetch-pin` CLI flags + WARN logging, cherry-picked from the `prefetch-skip-pinned` branch). G2 stayed
`IDENTICAL=True` and `test-backend-ops` was 13349/13349 + GDN 46/46 — both folds are inert on the default
path (prefetch gated behind `prefetch_experts`, GDN behind its env gate). **Consolidation note:** the fork
repo now holds every campaign line as a local branch (`lifecycle` deploy + `turbo-stack`,
`prefetch-skip-pinned`, `fable5-prefetch-experts`); the Turbo/expert-cache lineage that was a fragile
detached HEAD in the `stack` tree is now the `turbo-stack` branch. Nothing deleted.

**Harness relocation recheck 2026-08-23: 3/3 PASS** against binary build
`10159` (`068764d92`) and the Qwen3.6 35B-A3B MTP Q4_K_M model at
`--n-cpu-moe 8`. This rechecked the historical tuple; it did not qualify a
newer binary. See the exact command, hashes, and observations in
[`runs/fork/SLOP-BOUNDARY-QUALIFICATION-2026-08-23/RESULT.md`](../../runs/fork/SLOP-BOUNDARY-QUALIFICATION-2026-08-23/RESULT.md).
