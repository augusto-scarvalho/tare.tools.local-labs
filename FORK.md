# FORK — the consolidated `lifecycle` build and its runtime levers (2026-08-02)

Our own llama.cpp fork: **one binary** that is the pristine upstream baseline by default and gains
every validated-in-some-regime lever the campaign found, each **toggled at runtime** (env var or CLI
flag) — no recompile, and with nothing set the behaviour is byte-identical to upstream `720d7fa40`.

- **Where:** `/home/augus/src/llama.cpp-master`, branch **`lifecycle`**.
- **Base:** upstream **`720d7fa40`** (NOT fresh master — `f5919bf45` regressed `draft-mtp` token-exactness;
  see LANDSCAPE §1c). `GGML_CUDA_GRAPHS=ON`, `FA_ALL_QUANTS=OFF`, `sm_86`.
- **Binary:** `build/bin/llama-server` (+ `build/bin/llama-moe-trace` for the expert-cache profiler).
- **`ab_isolate.py`:** `LIFECYCLE_BIN`.

## The levers — all default OFF, all runtime-toggleable

| Lever | Turn ON with | Default | What it does | Measured verdict / best regime |
|---|---|---|---|---|
| **§B2b KV-host-pin** | env `GGML_KV_PIN_HOST=1` (+ `--no-kv-offload`) | off | pins the KV host buffer → direct-DMA the per-token host→GPU KV copy | **+up to 17%** in the `--no-kv-offload` / long-context (128k, VRAM-starved) regime; ours, novel |
| **Prefetch experts** | env `GGML_SCHED_PREFETCH_EXPERTS=N` (N>0) | off | overlaps offloaded-expert H2D uploads on a 2nd CUDA stream | **+58% prefill** / **−22% decode** — a **prefill / small-card** lever, a decode tax at our ncmoe=6 (§B3) |
| **Pin CPU weights** | env `GGML_CUDA_REGISTER_HOST=1` | off | page-locks mmap'd CPU expert weights for faster H2D | null at the decode optimum; helps **forced-heavy-offload** |
| **MoE expert cache** | flags `--moe-cache-slots N --moe-cache-profile <csv>` | off | keeps the N hottest routed experts resident in VRAM (hot/cold `mul_mat_id`) → hits skip per-token H2D | **null/redundant** vs static `--n-cpu-moe` on Qwen3's load-balanced routing (§E5); for **concentrated-routing** models |

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
`bash bless_fork.sh` — G1 §B2b engages · G2 `draft-mtp` token-identity (#23335) · G3 coherence + `-nkvo`
(#20140). Each ported lever was cherry-picked then re-checked against G2 so the **default path stays
byte-identical to `720d7fa40`** (MTP exact). Rebuild: `cmake --build build --target llama-server -j 20`.
