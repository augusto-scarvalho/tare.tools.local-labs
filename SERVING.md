# SERVING — the complete stack, in one page (2026-08-02)

Executive summary of everything the campaign settled about **serving Qwen3.6-35B-A3B fast, at quality, at
long context, on one RTX 3090 (24 GB) + 64 GB RAM**. Every number is committed evidence; the detailed
derivations live in `STATUS.md` (per §), `CONTEXT_PLAN.md`, `FORK.md`, `DEPLOY.md`. This is the "what do I
run, and what will it do" page.

## The binary
`llama.cpp-master` branch **`lifecycle`** = upstream `720d7fa40` + four runtime-gated non-upstream levers,
all OFF by default (`FORK.md`). Pinned to `720d7fa40`, NOT fresh master (which regressed draft-mtp
exactness). Blessed: draft-mtp quality-neutral, output coherent. Build: `GGML_CUDA_GRAPHS=ON`, `sm_86`.

## Three serve modes — pick by your workload

> Flag note (2026-08-04): on the deploy binary (`068764d92`) the batch flags are `--batch-size` /
> `--ubatch-size`; the old `--batch` / `--ubatch` now error out. Mode 1 re-validated end-to-end: 127–130
> t/s decode, 83.4% draft accept.

**1. Single user, lowest latency** (interactive chat, one stream)
```bash
llama-server -m Qwen3.6-35B-A3B-mtp.gguf -fa on --n-cpu-moe 8 -c 8192 \
  --cache-type-k q8_0 --cache-type-v q8_0 --spec-type draft-mtp --spec-draft-n-max 4 \
  --batch-size 2048 --ubatch-size 2048 --host 0.0.0.0 --port 8080
```
→ **~116 t/s decode**, quality-neutral. MTP ON (its +27% is a single-stream win).

**2. Long context** (agentic / long documents, one stream)
```bash
llama-server -m Qwen3.6-35B-A3B-mtp.gguf -fa on --n-cpu-moe 8 -c 131072 \
  --cache-type-k q4_0 --cache-type-v q4_0 --spec-type draft-mtp --spec-draft-n-max 4 \
  --batch-size 2048 --ubatch-size 2048 --host 0.0.0.0 --port 8080
```
→ **128k context, 100% multi-hop accuracy, ~60 t/s decode.** q4 KV is lossless here → doubles headroom.
Follow-ups on the same context are **sub-second** (prompt-cache reuse).
> **Re-validated 2026-08-04** (server `/completion` path, ~124.5k-token prompt, `068764d92`): cold prefill
> **137.8 s @ ub512 → 67.9 s @ ub2048 = 2.03× (the `--ubatch-size 2048` doubling holds exactly)**; decode
> at depth **61.8 → 67.6 t/s** (MTP on; MTP costs only ~5.5% of prefill and gives +46% decode: 46.3→67.6);
> warm reuse TTFT **0.24 s (273×**, only 4 new tokens prefilled); needle at 124.5k answered correctly.
> **⚠ Prefill regression vs 08-02 (open):** absolute TTFT is ~1.8× the §PF `llama-bench` figures (68 s vs
> 38 s). It is **NOT** a measurement-path artifact — `llama-bench` on this binary reproduces it exactly
> (pp131072: **926 t/s @ ub512, 1888 t/s @ ub2048** vs §PF's 1663/3441 → a systemic ~1.80×). Ruled out:
> MTP (~5.5%), XMP (active @ 5600), power plan (High performance), GPU clock-lock (−4% only). **Decode is
> UNaffected** (even faster than 08-02), so it's the host/CPU-bound offloaded-expert prefill path. **Leading
> suspect: Memory Integrity / HVCI-VBS is now ON** (`VBS status=2`, `HVCI=1`) — WSL2 runs in a Hyper-V VM and
> VBS taxes guest memory GEMMs; likely toggled on at the 08-03 reboot. Confirm by A/B with Memory Integrity
> off (needs reboot; same feature that blocks the CPU undervolt). The 2× ubatch *ratio* is intact regardless.
> **Envelope caveat:** ub2048 at 128k leaves only **~1.6 GB VRAM free** (22932/24576 MiB) — under the 4 GB
> reserve; drop to ub1024 or accept the tighter margin for long-context serving.

**3. Many users, max throughput** (multi-tenant)
```bash
llama-server -m Qwen3.6-35B-A3B-mtp.gguf -fa on --n-cpu-moe 8 -c 32768 -np 8 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --batch-size 2048 --ubatch-size 2048 --host 0.0.0.0 --port 8080
```
→ **~217 t/s aggregate (2.5×), ~30 t/s/user, +0.6 GB VRAM.** **MTP OFF** — it HALVES aggregate at N=8.

## The measured envelope

| axis | result | § |
|---|---|---|
| **Decode** (deploy, ncmoe=8) | ~116 t/s w/ MTP (100 base) | E1/E4/B4 |
| **Decode levers** | placement +268%, CUDA graphs +27%, MTP +27–83% — all upstream | E1,B4,E4 |
| **Quality** | every lever quality-neutral (pass@1 flat); q4 KV lossless | Q |
| **Context** | native 262k fits in VRAM; **usable ≥136k @ 100% multi-hop**; §B2b unneeded | ctx A–C |
| **Decode at depth** | ~86→60 t/s (8k→136k) — graceful | ctx B |
| **Prefill** | `--ubatch-size 2048` → **2× confirmed** (both tools); but ⚠ **~1.8× slower than 08-02 in absolute terms** (128k TTFT now ~68 s vs 38 s) — real regression, HVCI/VBS suspect (§ note above); cache reuse 15–273× | PF |
| **Concurrency** | N=8 → 2.5× aggregate ~free on VRAM; MTP flips OFF at N≈4 | CC |

## Lever decisions (what's on/off, and why)

| lever | verdict | when to use |
|---|---|---|
| **placement** `--n-cpu-moe 8` | the foundation (+268%) | always |
| **CUDA graphs** | +27%, ON by default | always (don't disable) |
| **MTP** `--spec-type draft-mtp` | +27–83% single-stream; NOT bit-exact but quality-neutral | N≤2; OFF for N≥4 |
| **`--ubatch 2048`** | 2× prefill | always (esp. long context) |
| **q4 KV** | lossless here; doubles context headroom | long context (≥32k) |
| **prompt-cache** | 15× follow-up TTFT | multi-turn on shared context |
| **§B2b KV-host-pin** | UNNECESSARY (KV fits in VRAM to native) | never, for the MoE |
| **prefetch / pinning** | NULL/tax at ncmoe=8 (only heavy offload) | never, for deploy |
| **expert cache** | NULL (load-balanced routing) | never, this model |

## The other model — dense 27B (Gated-DeltaNet hybrid)
MTP gives its biggest uplift (+49–83%) but base decode is slow (~43 t/s — that is 83% of the 3090
weight-bandwidth ceiling, so decode is bandwidth-bound and no kernel helps it; the GDN CUDA kernel runs
on-GPU, it is not disabled — the older "fused GDN kernel disabled" claim was wrong, see `GDN_KERNEL.md`).
GDN is instead the #1 *prefill* kernel here (~15% of dense GPU prefill); the chunk-parallel rewrite was
built + fully evaluated (M1–M4b, `GDN_M4_RESUME.md`) and is a **closed lever** — no B=1 or concurrent win
on this HW, kept only as an OFF-by-default opt-in (`GGML_CUDA_GDN_CHUNKED`, T≥1024 gate, quality-neutral).
Long context is the harder case: servable only ~48–64k (compute-scratch OOMs if `-c` is oversized → match
`-c` to use), needs `-ngl<65` or KV-in-RAM for more. Use the MoE unless you specifically need the dense.

## Quant choice — **Q4_K_M, decisively** (§QN)
On a 24 GB card, bigger weight-quant must offload more experts to CPU → slower, for zero measured quality:

| quant | min-fit ncmoe | decode | pass@1 |
|---|---:|---:|---:|
| **Q4_K_M (21 GB)** | 8 | **97 t/s** | 17/20 |
| Q5_K_M (25 GB) | 16 | 69 t/s | — |
| Q8_0 (35 GB) | 40 | 24 t/s | 16/20 |

**Q8 is 4× slower than Q4 for no quality gain.** Use **Q4_K_M** — same quality as Q8, fastest, most context
headroom. (Bigger quants only if you had far more VRAM.)

---
*Detailed evidence: `STATUS.md` §E1–§E5, §B2–§B5, §Q, §PF, §CC, §QN · `CONTEXT_PLAN.md` · `FORK.md` ·
`DEPLOY.md`. Tooling: `ab_isolate.py`, `context_probe.py`, `multihop_probe.py`, `prefill_probe.py`,
`concurrency_probe.py`, `quant_probe.py`, `quality_bench.py`.*
