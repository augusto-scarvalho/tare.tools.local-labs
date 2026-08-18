# GDN TF32 tensor-core kernel — implementation plan (M3 phase B)

Goal: convert the dense GEMMs of our chunked GDN CUDA kernel to **TF32 tensor cores** via
`ggml_cuda_mma`, keeping the numerically-sensitive parts fp32. Deploy path: scalar gating,
S_v=128, K=1, H=32, RTX 3090 (sm_86). Current fp32 chunked kernel = 1.8× slower than sequential.

## Key insight (GA102-specific)
On the 3090, **TF32 tensor throughput == FP32 (both 35.6 TFLOPS)** — TF32 buys NO raw FLOP/s here.
The win is **shared-memory (LDS) bandwidth relief**: scalar dot loops are LDS-bound at ~4.1 B/FMA
(4× below peak); m16n8k8 mma drops to ~0.75 B/FMA. Per-GEMM ceiling ≈4×; whole-kernel ≈2–2.5×
→ from 1.8× slower to ~1.3–1.6× faster isolated, ~+5–10% E2E. TF32 (10-bit mantissa) preserves
precision far better than bf16, and the win doesn't depend on TF32's (low, on GA102) tensor rate.

## Precision split (from PR #26001 code + fla `allow_tf32=False`)
**Stays fp32** (scalar/register/smem): cumulative-decay prefix sum `cs`; A-build
`A_tr=β_t exp(c_t−c_r)(k_t·k_r)`; T=(I+A)⁻¹ forward-subst; **U=T·W**; the recurrent state
accumulator **`Stile`** (never downcast — TF32 truncation happens only on mma register operands).
**Moves to TF32** (4 GEMMs): (1) readout W `Stile·k̃` [TILE_J×n×S_v]; (2) output = q-readout
`Stile·q̃` fused with `B·U` [n×TILE_J]; (3) carry `Kᵀ·U` [S_v×TILE_J×n]; (4) pre-pass B-build
`Q·Kᵀ` [n×n×S_v].

## API (ggml_cuda_mma, NOT raw wmma — reviewer ORippler required this of #26001)
- TF32 mma at `mma.cuh:1082-1096`: `mma(tile<16,8,float>& D, tile<16,8,float>& A, tile<8,8,float>& B)`
  → `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`. M=16,N=8,K=8, fp32 accum. Gated
  `AMPERE_MMA_AVAILABLE` (__CUDA_ARCH__>=800). Runtime gate: `ampere_mma_available(cc)` (common.cuh:352).
- A/B operands MUST come from `load_ldmatrix` (smem, 16B-aligned, row stride ≡0 mod 4 floats).
  C/D via `tile_C::get_i(l)/get_j(l)`. Copy the pattern verbatim from **`mmf.cuh:76-78,176-221,238-241`**
  (that IS the in-tree TF32 f32 path, Ampere-gated at `mmf.cu:186-188`).
- Padding: change `SP=S_v+1` → **`SP4=S_v+4`** and `CSP=CS+4` (ldmatrix needs stride mult-of-4;
  132/68 also bank-conflict-free). Keep sub-buffers mult-of-4 floats for 16B base alignment.

## Structure: keep 2 kernels (mma is warp-scoped, no extra launch). CS=64 stays
CS=64 tiles as 4 M-tiles × 4 N-tiles = 16 output tiles = our 16 warps (block 512), one tile/warp
for readout+output. CS=16 (#26001) would idle 75% of warps + 4× serial carry iters — reject.
Transposed state `Stile[jl*SP4+i]=S[i][j0+jl]` = already the K-major B operand for readouts → NO repack.
Repacks needed: `UT[jl][r]` (U transposed) for output/carry B-operand; `kTs[i][r]=brow[r]·k[c0+r][i]`
for carry (folds brow in). Main smem shrinks to ~77KB (from 89); pre-pass ~85KB → block 256→512.

## Zeroing obligations (mma contracts full tiles; scalar loops didn't)
1. Pre-pass writes the WHOLE CS×CS B (zero r>t and t≥n). 2. `UT[jl][r]=0` for r∈[n,CS). 
3. `kTs[i][r]=0` for r∈[n,CS). 4. Zero-fill bufK tails to keep NaN/Inf out of fragments.

## Tolerance: per-path `max_nmse_err()` override in `test_gated_delta_net`
TF32 ~10-bit → NMSE ~6e-8; use **2e-7** gated on `head_size>=64 && !kda && K==1 && n_seq_tokens>=64`
(superset of the runtime TF32 gate; do NOT add `!permuted` — our kernel runs permuted). fp32 paths
keep 1e-7. Bless-for-quality-neutrality like MTP. Add ragged S_v=128 rows: (4,128,129,1),
(4,128,191,2), (8,128,200,2,2,true).

## VERDICT (2026-08-03, after 3-worker research + robust 3-rep median sweep)
**The TF32 chunked kernel WINS over sequential at deploy scale, and this is the physics-expected ceiling — not a failure.** Robust sweep (median of 3, H=32 d=128, tf32/seq <1.0 = we win):
| tok | seq_us | tf32_us | tf32/seq | | batch @512tok | tf32/seq |
|----:|-------:|--------:|:--------:|---|:---|:---:|
| 64  | 91.8 | 130.6 | 1.42 slow | | seqs=1 | 1.11 slow |
| 256 | 352 | 390 | 1.11 slow | | seqs=2 | 1.13 slow |
| 512 | 720 | 800 | 1.11 slow | | seqs=4 | **0.99 WIN** |
| **1024** | 1559 | 1533 | **0.98 WIN** | | seqs=8 | **0.91 WIN** |
| **2048** | 3142 | 2989 | **0.95 WIN** | | | |
| **4096** | 6007 | 5835 | **0.97 WIN** | | | |
| 8192 | 11732 | 11984 | 1.02 par | | | |
**Crossover ≈ T=1024** (single seq); or T=512 once batch≥4. Deploy ubatch=2048 → **~5% faster**. TF32 gave a
uniform ~1.2× over fp32-chunked across ALL shapes (the GA102 LDS-relief win). NOTE: earlier single-shot showed
2048 at parity (3076 vs 3048) — that was measurement noise; the 3-rep median shows a clean 5% win. (Robust
methodology mattered.)

### Why (3 workers converged + community reproduced it independently)
1. **GA102 physics:** RTX 3090 TF32 tensor (35.6 TFLOPS) == FP32 CUDA-core (35.7) — 1:1 (A100 is 8×). TF32 buys
   NO FLOP/s here, only LDS relief → parity/small-win is the EXPECTED TF32 ceiling. bf16/fp16 = 71 TFLOPS = **2× TF32**
   is the only precision lever with real throughput headroom on this chip.
2. **B=1 occupancy:** our grid = H·n_seqs·(S_v/32) = 128 blocks, fixed, each serial over n_chunks. fla/Mamba2 split
   into 3 kernels so the FLOP-heavy work runs chunk-parallel (NT·H=1024-wide) — ~8× more parallelism. The batch
   sweep CONFIRMS occupancy: +batch flips us to a win (seqs≥4 @512tok).
3. **Critical-path asymmetry (why W-readout=1.38× but q-readout flat):** W-readout (S0·k) feeds W→U→carry, ON the
   serial inter-chunk critical path → speeding it compounds ×n_chunks. q-readout (S0·q) feeds only output, OFF the
   path → already latency-hidden → flat. (Steiger's breakdown; Amdahl.)
4. **Community reproduced this EXACTLY:** am17an built a TF32-chunked GDN, measured it, ABANDONED it ("not worth the
   complexity, not useful <16K batch·tok"). Neroued's B=1 crossover table: T=1024 1.32× slow, T=2048 0.96× (parity),
   wins only >3k — our curve matches. PR #26001 (live draft) uses **fp16 WMMA + CS=16** (not TF32), gets +3-18.7% E2E
   on ga102-family only at large ubatch. JohannesGaessler's own FA mma kernel on 3090: 0.87-0.96× (SLOWER) at batch
   8-32, wins only ≥128. **No chunked GDN CUDA kernel has landed upstream** (all open-draft or closed-for-maintenance).

### Next levers (ranked by the convergent evidence)
- **(0) GATE the chunked path at T≥1024** (measured crossover) → locks in the deploy win with ZERO regression. What
  #24561/#26001 both do. Lowest effort, highest certainty. DO THIS regardless of what else.
- **(1) bf16/fp16 tiles (mma.m16n8k16, fp32 accum)** — the one precision lever with 2× headroom on GA102; what the
  community converged on. Risk: fp16 overflow (needs clamp/accum care; gaugarg-nv flagged accuracy) AND at B=1 skinny
  matmuls may be latency-bound → may not deliver full 2×. Convert the critical-path GEMMs (W-readout, carry) first.
- **(2) 3-kernel split** (move output+U out of the serial loop, materialize per-chunk states ~2GB@2048tok) — attacks
  the 8× occupancy gap structurally. Biggest potential but biggest refactor.
- **(3) carry→TF32 (original Step 3)** — still on the critical path so likely helps, but capped at TF32's no-headroom
  ceiling; bf16-carry (lever 1) dominates it.

## Progress (perf @ H=32,S_v=128; sequential = 742µs/512, 3048µs/2048)
- **Step 0 DONE** (minimal): SP4 padding only; the bigger repacks (UT/kTs/TT-removal) deferred into
  the steps that need them. 46/46 @ 1e-7.
- **Step 1 DONE (GEMM-R, W readout → TF32).** Env `GGML_CUDA_GDN_TC` (default ON). 46/46 @ 2e-7;
  TC=0 stays 46/46 @ 1e-7 (re-threading is a no-op). Perf: 512tok 1133→**827µs**, 2048tok
  4340→**3132µs** = **~1.38× kernel** (beat the ~1.2× est). vs sequential: 512 1.11× slow,
  **2048 1.03× slow — near parity after ONE GEMM.** Key gotchas hit: (A) block MUST be warp-shaped
  `dim3(32,16)` so threadIdx.x==lane (mma indexes raw threadIdx.x) + `NT` must be constexpr 512 not
  blockDim.x; (B) A from load_ldmatrix, C from get_i/get_j — never mix; (bug) first applied the
  re-threading to the WRONG kernel (pre-pass, 256-thread) → NT=512 skipped half its work, n≥64 failed.
  Fable's Step-1 plan was accurate incl. the exact fragment index math (get_i/get_j landed correct
  first try). Two CUDA_SET_SHARED_MEMORY_LIMIT sites (per-expansion-site static flag).

## Incremental steps (each: build → 46/46 → perf on 32,128,512/2048)
- **0** [DONE, minimal] Pure re-layout, TC template param: SP4 padding. (UT/kTs/TT deferred.)
- **1** [DONE] GEMM-R (W readout) → TF32 (zero repack, biggest lever). 46/46 @ 2e-7. **~1.38× (measured).**
- **2** [DONE] GEMM-O pass1 (q readout) → TF32. 46/46 @ 2e-7. **~1.02× — nearly FLAT (measured).**
  512tok 827→822µs, 2048tok 3132→3076µs. Lesson: the q-readout was NOT a big lever (unlike W-readout);
  the WAR barrier (S2) + only 8/16 warps active (TT=32→2 M-tiles) + s0q smem roundtrip ~cancel the saving.
  Fable's ~1.2-1.35× estimate was too optimistic. Kept (correct, harmless, prerequisite structure).
  **At 2048 (deploy prefill regime): chunked 3076µs vs sequential 3048µs = PARITY (0.9% slow) after 2 GEMMs.**
  → Per-GEMM Amdahl estimates are unreliable; PROFILE to pick the next lever rather than follow plan order.
- **2** GEMM-O pass1 (q readout) → TF32, fused accumulator. ~1.6× cum.
- **3** GEMM-C (carry) → TF32 (uses UT+kTs). ~2.2× cum — **crosses to faster than sequential**.
- **4** GEMM-O pass2 (B·U) → TF32 + triangular kk≤2·mt+1 skip. ~2.5× cum.
- **5** GEMM-B (pre-pass Q·Kᵀ) → TF32, pre-pass block→512. ~2.7× cum.
- **6** tuning (TILE_J=16 tail-wave, unroll, load_generic vs ldmatrix, CS=32).
- **7** E2E llama-bench -p 2048.
Halve the optimistic Amdahl deltas for reality (mma issue/ldmatrix latency, syncs, tail wave).
Do 1→5 in order (step 1 needs no repack → a failure there is unambiguously a fragment-layout bug).

## Risks / stop
Fragment-layout bug (mirror mmf.cuh; symptom NMSE≫1) · missing zero-fill (only ragged rows fail) ·
repack cost eats win (step 0 measured separately; if >5% reconsider carry) · TF32>2e-7 (raise to
1e-6 only after ruling out bug) · **STOP** if after step 3 still slower than sequential @2048 (then
carry/launch overhead is the limiter, not GEMMs). Honest ceiling ~1.3–1.6× isolated / +5–10% E2E.
More than that needs bf16 (2× TF32 rate on GA102) but loses the fp32-state property — separate decision.

Files: `ggml-cuda/gated_delta_net.cu` (kernels+launchers), `mma.cuh` (1082-1096 TF32 mma, 778-859
loads, 239-271 C-layout), `mmf.cuh` (76-78,176-221,238-241 pattern), `test-backend-ops.cpp`
(4091 struct, 9660/10018 rows), `common.cuh` (282 AMPERE_MMA_AVAILABLE, 352 ampere_mma_available).
