# DEPLOY — the consolidated best config and why (2026-08-02)

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
  --host 0.0.0.0 --port 8080
```

**Delivers ~116 t/s decode inside the safe envelope, output identical to greedy.** This single
command already banks three independent wins — placement, CUDA graphs, and MTP — stacked.

Knobs:
- **`--n-cpu-moe 8`**, not 6. ncmoe=6 is the raw decode optimum (~102 t/s) but MTP's draft context
  (~1.15 GB VRAM) pushes it under the 4 GB VRAM reserve. ncmoe=8 seats the draft and MTP still holds
  ~116 t/s (it decouples decode from placement — see §E4).
- **Do NOT set `GGML_CUDA_DISABLE_GRAPHS`.** Graphs are ON by default and worth +27% (§B4).
- **KV `q8_0`** at ≤8k ctx. For the 128k agentic case switch to **`q4_0`** (frees VRAM; null on
  quality/speed at short ctx, load-bearing at long ctx) and raise `--n-cpu-moe` if VRAM binds.
- **No pinning** (`GGML_CUDA_REGISTER_HOST`) — null at this placement (§E1); only helps forced heavy
  offload.
- **MTP shines on structured/tool-heavy output** (higher accept → up to +54% on code); on free-form
  reasoning it is ~+27%. Always exact, so it is free to leave on.

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
- **Builds** llama.cpp at `/home/augus/src/llama.cpp-{master,local,base,rebase,stack}`; ik at
  `/home/augus/src/ik_llama.cpp`. MTP + CUDA graphs confirmed in `-master` (commit `720d7fa40`,
  `GGML_CUDA_GRAPHS=ON`). Models in WSL at `/home/augus/models/`.
- **Gotcha** simple `$VAR` in `wsl.exe -- bash -lc '...'` via the Bash tool expands EMPTY — use
  literal paths or a script file (see the `verify_mtp.py` pattern).

---

## Campaign status — done vs remaining optionals

**DONE & committed** (`b06fb54` → `cd9e975` → `ccab4f3` → `00820e4` → `4707354`):
§E1 placement · §E2 ik engine · §E4 MTP (MoE + dense) · §B4 CUDA graphs · §B1 pinning dose-response
(earlier). The big, safe decode levers are settled.

**Remaining OPTIONAL (lower value, pick after the context compaction):**
- **§B3** — GPU-idle% instrumentation: explains the prefetch tax vs the 3060's win; cheap, adds a
  discriminating metric to the harness.
- **§B5** — `--pin-hot-experts` (upstream #25932): selective vs blanket pin on the generation side.
- **§B2** — pinned KV in RAM: novel for llama.cpp; a context-length dose-response (needs a small patch).
- **Consolidate a build** (LANDSCAPE §1b): upstream master + pinning gated-off + whichever decode
  lever survives, − prefetch. Only worth it once a non-upstream lever is proven; so far the winners
  (placement, graphs, MTP) are ALL upstream — arguably no fork is needed at all.
- **Quality axis** (long-standing OPEN): `quality_bench` starves the thinking model on HumanEval+;
  no pass@1 until the token-budget/thinking issue is fixed.
