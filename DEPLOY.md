# DEPLOY — the consolidated best config and why (2026-08-02; re-validated + extended through 2026-08-04)

The one-page answer to "how do I run this fast, and what did the campaign settle." Every number
here is committed evidence; see `STATUS.md` for the full derivation of each and `runs/` for the raw
records. This is the durable handoff — read it first after a context reset.

---

## TL;DR — the recommended serve config (MoE, the deploy model)

```bash
/home/augus/src/llama.cpp-master/build/bin/llama-server \
  -m /home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -fa on \
  --n-cpu-moe 8 \
  --ctx-size 8192 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  --batch-size 2048 --ubatch-size 2048 \
  --host 0.0.0.0 --port 8080
```
> **Flag names (2026-08-04):** on the deploy binary (`lifecycle` @ `068764d92`, newer than base `720d7fa40`)
> the batch flags are `--batch-size` / `--ubatch-size`; the short `--batch` / `--ubatch` are no longer
> accepted (`error: invalid argument: --batch`). The MTP flags (`--spec-type draft-mtp`,
> `--spec-draft-n-max`) are unchanged.
>
> **Validated end-to-end 2026-08-04** (this exact config, post-reboot, clean run): decode **127–130 t/s**
> (meets/beats the ~116 campaign figure below), **draft acceptance 83.4%** (196/235, mean accepted len
> 4.32), model→`/health` ~11 s (warm page-cache), prompt-cache reuse TTFT 270→83 ms. VRAM 21014/24576 MiB
> (~3.5 GB free at 8k ctx + q8 KV — a hair under the 4 GB reserve on a single post-run sample; watch it
> under sustained load).
>
> **Long-context (mode 2) re-validated 2026-08-04** (server path, ~124.5k-token prompt, `-c 131072` q4 KV):
> cold prefill **137.8 s @ ub512 → 67.9 s @ ub2048 = 2.03×** (the doubling holds exactly); decode at depth
> **61.8→67.6 t/s**; warm prompt-cache reuse TTFT **0.24 s (273×)**; needle at 124.5k answered correctly.
> **No real prefill regression** — the earlier "~1.8×" was a measurement confound (resolved same day): the
> §PF "38 s @ 128k" was a *linear extrapolation* from a ~41k-token throughput number, ignoring the mild
> throughput falloff with length; the real measured 128k TTFT is ~68 s and always was. Apples-to-apples
> with the SAME §PF tool (`prefill_probe.py`, ~38k prompt) this binary gives **2838 t/s @ ub2048 vs §PF's
> 3441 (~1.2×, within length + run-to-run variance; a direct clock-lock A/B was ~0% here — 2387 t/s locked
> @1800 vs 2311 unlocked @1905, within noise — this MoE prefill is transfer-bound, not clock-bound), doubling
> reproduces (+100.8%)**. Prefill is
> transfer/GPU-side, not CPU-bound (CPU ~6% during it); PCIe is Gen4 x16 full; fork/binary, MTP, XMP, clock,
> TDR, power, and Game-Boost/HVCI all cleared as causes of even the residual.
> **Caveat: ub2048 at 128k leaves only ~1.6 GB VRAM free** — under the 4 GB reserve; use ub1024 if you need margin.
> **`--ubatch-size 2048`** (§PF) roughly **doubles prefill** (2× confirmed both tools, +100.8%; real 128k
> TTFT ~68 s — the 08-02 "38 s" was a linear extrapolation from ~41k) — free; the default
> 512 leaves half the prefill speed on the table. For multi-turn on a shared long context, keep
> `cache_prompt` on: a follow-up query REUSES the prefix KV → **~15× faster follow-up TTFT** (prefill the
> doc once, then sub-second). (Fable prefetch is NULL for prefill at ncmoe=8 — only heavy offload; §PF.)
>
> **Serving mode (§CC):** single/low-concurrency (N≤2) → `-np 1` + **MTP ON** (lowest latency, ~116 t/s).
> Multi-user throughput → `-np 8` + **MTP OFF** → **~217 t/s aggregate (2.5×)**, ~30 t/s/user, +0.6 GB
> VRAM. **MTP flips at N≈4** — it HALVES aggregate throughput at N=8 (and costs VRAM), so drop `--spec-type`
> for high concurrency. Concurrency is nearly free on VRAM (q4 KV per slot is tiny).
>
> **Drafter = `draft-mtp` alone — do NOT stack n-gram (S3, rigorously re-verified 2026-08-04).** Measured vs a
> **no-spec floor of ~87 t/s** (6 reps, 95% CI, clock-stable): `draft-mtp` = **132–151 t/s (+53% to +73%)** across
> code-gen / edit / pure-copy — MTP is a real ~1.7× win, and it wins in every regime. `ngram-simple` alone is
> net-NEGATIVE on real code (GEN −5%, EDIT −44%) and only pays on ~verbatim copy (+24%). Stacking
> `draft-mtp,ngram-simple` is **always worse than mtp-alone** (EDIT drops to −26% below floor) because ngram has
> higher priority (upstream `docs/speculative.md`: "draftless decoding has higher precedence") and preempts MTP's
> better draft. **Exactness note:** `ngram-simple` is greedy-exact; `draft-mtp` deterministically diverges from
> greedy (quality-neutral on HumanEval+, but not bit-exact — the FORK.md "token-exact" is fork==base, not
> spec==greedy). Regression gate: `ops/spec-drafter-bench.sh` (no-spec floor + CI + 3 regimes).
>
> **GEMM path = leave it default (MMQ int8-TC) — do NOT force cuBLAS (S2, closed + double-checked 2026-08-04).**
> llama.cpp already runs the INT8-Tensor-Core fused-dequant GEMM (MMQ, `s8.s8.s32`) by default on the 3090 for
> Q4_K at every batch. For the deploy MoE it beats the dequant→FP16→cuBLAS path by **~+284%** (ncmoe=8) — and
> forcing cuBLAS for MoE isn't just slow, it's **BROKEN**: it overflows to NaN / corrupts output on sm_86
> (upstream #19659) and breaks CUDA graphs. `GGML_CUDA_FORCE_CUBLAS` only *legitimately* helps large-ubatch
> **dense-27B** prefill (~+5-11%), and there it costs VRAM (FP16 dequant buffers) + isn't quality-tested → not
> worth it on our VRAM-tight box. Physics (why there's no win to build): on GA102 int8 is only ~2× the
> fp16/fp16-acc rate cuBLAS uses, quant overhead eats it, and ub2048 prefill is compute-bound (Marlin converges
> to fp16 there). Gate: `ops/mmq-vs-cublas-bench.sh` (isolated arms + cooldown — back-to-back cells heat-soak the
> GPU and inflate variance). **Pin-watch:** upstream #26141 (`smpbo<48KiB` guard) regresses 3090 prefill to ~40
> t/s (#26285); we're pinned pre-that (720d7fa40) — re-check before any pin bump.

**Delivers ~116 t/s decode inside the safe envelope, quality-neutral (pass@1 unchanged; equivalent
output, not byte-identical to non-spec — §Q).** This single
command already banks three independent wins — placement, CUDA graphs, and MTP — stacked.

Knobs:
- **`--n-cpu-moe 8`**, not 6. ncmoe=6 is the raw decode optimum (~102 t/s) but MTP's draft context
  (~1.15 GB VRAM) pushes it under the 4 GB VRAM reserve. ncmoe=8 seats the draft and MTP still holds
  ~116 t/s (it decouples decode from placement — see §E4).
- **Do NOT set `GGML_CUDA_DISABLE_GRAPHS`.** Graphs are ON by default and worth +27% (§B4).
- **KV `q8_0`** at ≤8k ctx. **Long context is now MEASURED (CONTEXT_PLAN): the MoE runs at `-c 131072`
  with `q4_0` KV at ncmoe=8 — 100% multi-hop accuracy, ~60 t/s, within the envelope (a 16× jump over 8k
  at zero quality cost).** q4 KV is lossless here (hybrid arch) and doubles headroom; native 262k fits
  physically. §B2b / KV-in-RAM turned out UNNECESSARY for the MoE. For 128k+ just set `-c` and use `q4_0`.
- **No pinning** (`GGML_CUDA_REGISTER_HOST`) — null at this placement (§E1); only helps forced heavy
  offload.
- **MTP shines on structured/tool-heavy output** (higher accept → up to +54% on code); on free-form
  reasoning it is ~+27%. **Quality-neutral** (pass@1 unchanged, §Q) so free to leave on — but note it is
  NOT byte-identical to non-spec decode over long generations (batched-verify numerics flip greedy ties).

---

## Why — the decode-lever stack (all committed)

Baseline was **27.6 t/s** (the whole early campaign ran at ncmoe=40, max offload — the worst
placement). The deployed ~116 t/s is that baseline with three levers stacked:

| Lever | Gain | Note |
|---|---|---|
| **Placement** — sweep `--n-cpu-moe` 40→6/8 (§E1) | **+268%** (27.6→~102 t/s) | the foundation; free, stock |
| **CUDA graphs** (§B4) | **+27%** (79→~100 t/s) | **ON by default** — already inside the 102 |
| **MTP speculative decode** (§E4) | **+27% text / +54% code** (MoE) | exact, 80.5% accept; stacks → ~116 t/s |

Levers that did **not** help on this box:
- **ik_llama.cpp** engine swap (§E2): decode **tie** at ncmoe=6; RAM-unsafe at heavy offload. Revisit at 128 GB.
- **KTransformers** (§E3): **not built** — its headline is AMX CPU kernels, and this CPU (i7-13700K)
  has no AMX/AVX512 (AMX is Xeon-only). Its AVX2 fallback is the same class that tied in §E2. Revisit
  only with an AMX server CPU.

Dense variant (Qwen3.6-27B, a Gated Delta Net hybrid): MTP gives **+49% text / +83% code** — the
biggest uplift, because a dense forward pass is costliest per token. `-ngl 65`; binds on the 16 GB
RAM reserve, not VRAM.

**Unifying thesis:** the differentiated asset of this machine is the **GPU (RTX 3090, 24 GB,
936 GB/s)**, not the consumer CPU. Every **GPU-side** lever won (placement, graphs, MTP); every
**CPU-side** engine tied (ik, KTransformers-AVX2). Optimize the GPU path.

---

## The safety envelope (what binds, and where)

`guard.py` enforces two reserves over sustained samples; a config that breaches either is REJECTED:

| Reserve | Value | Binds for… |
|---|---|---|
| **VRAM free** | **4 GB** | MoE at ncmoe=6 + MTP (draft ctx ~1.15 GB) → use ncmoe=8 |
| **Windows RAM free** | **16 GB** | dense 27B (17 GB mmap + draft + iGPU RAM); ik at heavy offload |

VRAM is the binding constraint for the MoE; system RAM for the dense and for compute-on-CPU engines.

---

## Machine baseline (environment prerequisites a fresh context won't know)

- **CPU** Intel i7-13700K (Raptor Lake): AVX2 + AVX-VNNI, **no AMX, no AVX512** (20 threads in WSL).
- **GPU** RTX 3090, 24 GB, Ampere sm_86 (Volta+, so CUDA graphs are arch-eligible).
- **RAM** 64 GB DDR5, **XMP active @ 5600 MHz** (measured ~+13% on CPU-bound offloaded decode vs JEDEC).
- **Display** on the 3090 (so ~0.76 GB VRAM held by DWM); **11 desktop apps moved to the Intel UHD 770
  iGPU** to free VRAM (`ops/gpu_prefs_*`, `GpuPreference=1`). iGPU renders from system RAM → adds RAM
  pressure, relevant to the 16 GB reserve.
- **WSL** Ubuntu-24.04, `.wslconfig`: `memory=44GB`, **`swap=16GB`** (added this session — turned the
  ik ncmoe=40 hard-crash into a measurable REJECTED; near-zero SSD wear), `processors=20`,
  `autoMemoryReclaim=gradual`. Backup at `~/.wslconfig.bak-*`.
- **THE fork / deploy binary** (2026-08-02): `/home/augus/src/llama.cpp-master`, branch **`lifecycle`**
  = upstream **`720d7fa40`** + **four runtime-gated non-upstream levers, all OFF by default** (§B2b
  KV-host-pin, Fable prefetch, CPU-weight pinning, MoE expert cache) → the one binary is baseline *and*
  fork. Default (nothing toggled) is byte-identical to `720d7fa40`. **BLESSED** (`bless_fork.sh`):
  draft-mtp token-exact, §B2b engages, coherent. Pinned to `720d7fa40`, **not** fresh master (`f5919bf45`
  regressed draft-mtp exactness). Turbo/TurboQuant excluded (multi-backend, unused KV format, null).
  `GGML_CUDA_GRAPHS=ON`, `FA_ALL_QUANTS=OFF`, `sm_86`. **Full lever manual + toggles: `FORK.md`.**
- **Other builds** (historical / rival): `llama.cpp-{local,base,rebase,stack}`, ik at
  `/home/augus/src/ik_llama.cpp`. Models in WSL at `/home/augus/models/`.
- **Gotcha** simple `$VAR` in `wsl.exe -- bash -lc '...'` via the Bash tool expands EMPTY — use
  literal paths or a script file (see the `verify_mtp.py` pattern).

---

## Campaign status — done vs remaining optionals

**DONE & committed** (§E1 placement · §E2 ik · §E4 MTP MoE+dense · §B4 CUDA graphs · §B1 pinning · plus
the three optionals below): the big, safe decode levers are settled.

**Optionals — ALL THREE DONE (2026-08-02):**
- **§B3** — GPU-idle% instrumentation: **DONE.** Harness now reports serving-window %busy/idle. Idle
  tracks the placement penalty (39%→63% as ncmoe 6→24); the 3090 sits ~40% idle even at the optimum
  (batch-1 A3B is bandwidth-bound). Reconciles the prefetch tax vs the 3060's win. STATUS §B3.
- **§B5** — `--pin-hot-experts`: **DONE — N/A here.** Flag unmerged/experimental (#25932 closed, #26414
  open); mechanism is anti-disk-eviction for a MoE exceeding RAM. Precondition measured absent (0
  steady-state major faults at ncmoe=40) — experts never spill. No build warranted. STATUS §B5.
- **§B2** — pinned KV in RAM: **DONE — precondition confirmed, patch works.** `--no-kv-offload` is a
  −70%→−77% transfer-bound regime scaling with context; an env-gated patch (`GGML_KV_PIN_HOST`,
  `patches/b2b-kv-host-pin.patch`) pins the KV host buffer and recovers +2.5%→+16.8% (rising with depth,
  lower bound). Held in reserve for the **128k long-context / VRAM-starved** case; KV-on-GPU still wins
  at 8k. STATUS §B2.

**Since settled (2026-08-03/04) — supersedes the "still open" list that used to live here:**
- **Build consolidation: DONE.** The fork (`llama.cpp-master` / `lifecycle`) now holds every campaign line
  as a local branch; the prefetch-skip CLI flags + the GDN chunked kernel are folded in, all gated OFF by
  default, re-blessed 3/3 (byte-identical to `720d7fa40` on the default path). See `FORK.md`.
- **Quality axis: DONE** (STATUS §Q). The budget-starvation bug was fixed and pass@1 measured — **no lever
  costs quality**; q4 KV is lossless, MTP and the GDN kernel are quality-neutral.
- **Long context: DONE** (`CONTEXT_PLAN.md`) — 128k usable at ~60 t/s, q4 KV lossless; §B2b unneeded for the MoE.
- **GDN chunked kernel: DONE & CLOSED** (`GDN_TF32_PLAN.md`) — blessed as an opt-in, no-regression option;
  TF32 is the GA102 ceiling, so it does not surface E2E at deploy scale.
- **GPU stability: SETTLED** — clock-lock retired for a validated GPU undervolt (~1860 MHz @ 850 mV);
  see `ops/gpu-stability/setup_gpu_protections.ps1`.

**Genuinely still open (low priority):**
- **Lever C** (GDN 3-kernel split → occupancy) — future research; wants A100-class HW where TF32 is 8×.
  Not worth pursuing on the 3090 (`GDN_NEXT_LEVERS.md`).
- **The −10.4% no-mmap residual** (STATUS) — one historically disputed delta with no clean paired A/B yet.
- **Only-if-needed later:** TurboQuant KV (tq3_0, §D) or YaRN — only if a >256k context target appears.
