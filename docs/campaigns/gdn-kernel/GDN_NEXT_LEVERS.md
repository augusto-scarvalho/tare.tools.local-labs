# GDN kernel — next levers (B: bf16, C: 3-kernel split)

Context: the chunk-parallel Gated-DeltaNet CUDA kernel (`ggml-cuda/gated_delta_net.cu`) is DONE and
**wins at deploy scale** — TF32 tensor cores on the W-readout GEMM, gated at `n_tokens >= 1024`
(env `GGML_CUDA_GDN_CHUNKED_MIN_TOKENS`), ~5% faster than sequential at ubatch=2048, no regression below.
Full verdict + why-parity-is-the-TF32-ceiling in `GDN_TF32_PLAN.md` (VERDICT block). This doc briefs the
two remaining levers so we can resume without re-deriving. **Order: A done → (compact) → B → C=future.**

## Current kernel state (what B/C build on)
- Block `dim3(WARP_SIZE, GDN_CHUNKED_NWARPS)` = `dim3(32,16)` = 16 warps; `threadIdx.x`=lane, `.y`=warp_id;
  `tid=threadIdx.y*32+threadIdx.x`, `constexpr NT=512`. Grid (H, n_seqs, S_v/TILE_J=4). CS=64, TILE_J=32, SP=S_v+4.
- 2 kernels: parallel pre-pass (cs/betas/B/T=(I+A)⁻¹ into `ctx.pool()` scratch) + serial main
  (gather ks → W-readout `S0·k̃` [TF32] → U=T·W [scalar] → output `S0·q̃`[TF32] + B·U[scalar] → carry[scalar]).
- Proven mma idiom: `typedef tile<16,8,float> tile_A/tile_C; tile<8,8,float> tile_B;` `load_ldmatrix(t,ptr,SP)`
  (stride in floats, smem 16B-aligned), `mma(C,A,B)`=`C[m][n]+=Σ_k A[m][k]B[n][k]`, epilogue `C.x[l]` +
  `tile_C::get_i(l)=(l/2)*8+lane/4`, `get_j(l)=(lane%4)*2+l%2`, `ne==4`. Env `GGML_CUDA_GDN_TC` (default ON) →
  `if constexpr(TC&&!KDA){mma}else{scalar}`. Tolerance override 2e-7 in `test_gated_delta_net::max_nmse_err()`.
- Validate: `GGML_CUDA_GDN_CHUNKED=1 GGML_CUDA_GDN_CHUNKED_MIN_TOKENS=2 test-backend-ops test -o GATED_DELTA_NET`
  (forces chunked on small cases). Perf sweep: `python3 ~/gdn_perf_sweep.py 3` (median of 3, seq/fp32/tf32 table).
  Build: `wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /home/augus/src/llama.cpp-master && cmake --build build --target test-backend-ops -j'`.

---

## Lever B — bf16/fp16 tensor cores — DONE (2026-08-03): NEGATIVE RESULT, reverted

**VERDICT: bf16 W-readout is FLAT vs TF32 → the kernel is latency/occupancy-bound, NOT tensor-throughput-bound. Precision levers are exhausted. Reverted to TF32; C is the only remaining path.**

Implemented the bf16 W-readout exactly per the plan below (Fable-5-high plan + hand-verified fragment
layouts): `mma.m16n8k16.f32.bf16.bf16.f32`, A packed from two fp32 ldmatrix loads via
`__floats2bfloat162_rn` (adjacent-k pairs → layout-exact), B built by direct `float2` smem reads
(the fp32 8×8 ldmatrix gives non-adjacent k=c,c+4 → unusable for a k16 pack — the crux), fp32
accumulate + fp32 epilogue kept. **Correctness: 46/46 at NMSE 1e-4** (layout verified correct).
**Perf (median-of-3, bf16 vs the TF32 baseline it replaced), deploy shapes:**

| shape (H32,d128) | TF32 µs | bf16 µs | Δ |
|---|---:|---:|---:|
| 1024 tok | 1484.5 | 1467.4 | −1.2% (noise) |
| **2048 tok (deploy)** | **2878.7** | **2868.8** | **−0.3% (noise)** |
| 2048 tok ×4 seq | 10281.3 | 10230.9 | −0.5% |
| 4096 tok | 5683.8 | 5809.1 | **+2.2% WORSE** |
| 8192 tok | 11293.0 | 11910.4 | **+5.5% WORSE** |

bf16 = 71 TFLOPS = 2× TF32 on GA102, yet doubling tensor FLOP/s bought **nothing** — flat at mid
lengths, worse at long (the extra CVT/pack per k-step is pure overhead on a kernel that waits on
dependencies, not tensor math). `tf32vfp32 ≈ tf32vbf16 ≈ 0.77–0.87`: bf16 gives the SAME gain over
scalar-chunked as TF32, no more. **This is the direct measurement that the B=1 chunked GDN is bound
by mma-issue latency / the serial inter-chunk carry dependency / occupancy (1 block/SM, smem-capped
at 99KB), not by tensor throughput.** Worker-2's latency-bound caveat + Fable risk #6 confirmed
empirically. bf16 is strictly worse here (flat-to-slower AND less accurate: 7-bit mantissa, tol 1e-4
vs TF32's 2e-7) → reverted. **The remaining headroom is structural (occupancy), which only Lever C
attacks.** The original bf16 how-to plan is retained below for the record / future A100-class HW
(where TF32 is 8× and the throughput picture changes).

<details><summary>Original Lever B plan (retained for record — the implementation was correct, the lever just doesn't pay on GA102 at B=1)</summary>

### Lever B — bf16/fp16 tensor cores (the real 2× lever on GA102)

**Thesis:** on RTX 3090, fp16/bf16 mma = 71 TFLOPS = **2× TF32** (35.6). TF32's `mma.m16n8k8` does K-depth 8;
fp16's `mma.m16n8k16` does K-depth 16 → half the instructions for the same reduction. This is the one precision
lever with real throughput headroom on this chip (TF32 bought only LDS relief). Community converged on fp16
(PR #26001 uses fp16 WMMA). Expected: push the ~5% win larger and/or lower the crossover below 1024.

**Scope (critical-path GEMMs only):** convert the **W-readout** (`S0·k̃`, already TF32) and the **carry**
(`Kᵀ·U`, still scalar — do Step-3-as-bf16, skipping the TF32 intermediate) to fp16/bf16. The q-readout is off
the critical path (flat under TF32) — leave it TF32 or scalar; not worth bf16.

**The hard part — operands are fp32 in smem, mma needs `half`:**
- Need `half` tiles: `tile<16,8,half> tile_A/half; tile<8,8,half>`. `mma(tile<16,8,float>& D, tile<16,8,half>,
  tile<8,8,half>)` = `mma.m16n8k16.row.col.f32.f16.f16.f32` — fp32 ACCUMULATE (keep!). See `mmf.cuh`/`mma.cuh`
  for the fp16 tile + load path (it's the mainline fma-fp16 path, well-trodden, unlike our TF32-on-fp32 detour).
- `Stile` (state) and `ks`/`WU` are fp32 in smem. Options: (a) convert to half in a smem staging pass before
  ldmatrix (extra `__half2` packing loop, +smem for the half copy — check the 99KB budget, half copy is HALF
  the bytes so likely fits); (b) keep a half shadow of Stile updated each chunk. **Decision pending — (a) is
  simpler/lower-risk for a first cut; the half copy of Stile[TILE_J*SP] + ks[CS*SP] is ~half their fp32 size.**
- **KEEP the state accumulator (`Stile` carry) in fp32** — only the mma OPERANDS are downcast to fp16 (fp32
  accumulate in the tile). This preserves the fp32-state property that kept us at NMSE 2e-7.

**Accuracy risk (must handle):**
- fp16: 5-bit exponent → overflows at ±65504. The `S0·k` and `Kᵀ·U` products CAN overflow. PR #26001 added
  `isfinite`/clamp; gaugarg-nv flagged ~3% top-1 concern. Mitigation: q/k are L2-normalized (bounded), betas∈[0,1],
  decays exp(c)≤1 → operands are smallish, but VERIFY max magnitude (wikitext showed ~21.6 in #26001, safe).
- bf16: 8-bit exponent (no overflow) but 7-bit mantissa (< TF32's 10) → NMSE rises. Likely need tolerance 1e-6
  (bless for quality-neutrality like MTP). **Prefer bf16 first** (no overflow logic), fall back to fp16 if bf16
  NMSE is unacceptable. `mma.cuh` — check whether the bf16 (`nv_bfloat16`) tile + `mma.m16n8k16.f32.bf16.bf16.f32`
  is present; if only fp16 tiles exist, use fp16 + clamp.

**Latency-bound caveat (why it might underdeliver):** at B=1 the matmuls are skinny (M=n≤64, N=8 per tile) → may
be latency-bound on mma/ldmatrix ISSUE, not FLOP-throughput → bf16's 2× may not materialize. Worker-2 flagged this.
**First action: profile the current TF32 kernel with `ncu` (warp-stall reasons, `sm__pipe_tensor` utilization) to
confirm throughput- vs latency-bound BEFORE investing in bf16.** If latency-bound, bf16 won't help much → skip to C.

**Steps:** (0) ncu the current kernel → throughput or latency bound? (1) if throughput-bound: bf16 the W-readout
(mirror the TF32 block, half tiles + half staging), build → validate (1e-6 tol) → perf sweep. (2) bf16 the carry
(this is Step-3-in-bf16; needs the UT/kTs repacks from the shelved Step-3 Fable plan, but now with half operands).
(3) re-sweep, update the crossover threshold if it dropped. Use Fable-5-high for the bf16-carry (still complex:
transposed repacks + smem budget + half packing).

</details>

---

## Lever C — 3-kernel split (structural, attacks the 8× occupancy gap) — NOW THE PRIMARY PATH

**Elevated from "future research" to the primary remaining lever by the Lever B negative result:** B proved
the kernel is latency/occupancy-bound, so C's extra parallelism (not precision) is the only thing that can
move it. Still gated behind hardware/appetite per the original framing below — it's a large structural rewrite.

**Thesis:** the root limiter at B=1 is occupancy — our grid = H·n_seqs·(S_v/32) = 128 blocks, FIXED, each serial
over n_chunks. fla/Mamba2 split into 3 kernels so the FLOP-heavy work runs **chunk-parallel** (grid over chunks →
NT·H ≈ 1024-wide, ~8× more). The batch sweep confirms occupancy is the lever (batch≥4 flips T=512 to a win).
This is the ONLY lever that changes the fundamental parallelism; precision (B) cannot.

**Design (mirror fla `chunk.py` / Mamba2 SSD):**
1. **Prepass** (chunk-parallel, grid (H, n_seqs, NT)): compute per-chunk W/U/T — no state dependency. We ALREADY
   have a parallel pre-pass; extend it to also emit U (=T·(βV)) and the local terms.
2. **Thin serial state-passing kernel** (grid (H, n_seqs, S_v/TILE_J), serial over chunks): ONLY the inter-chunk
   recurrence `S_t = decay·S_{t-1} − Kᵀ(W·S_{t-1}) + Kᵀ·U` (W-readout + carry). Strip everything else out. Keep
   the TILE_J state-column tiling (it's the standard pattern per fla/Mamba2 — NOT the anti-pattern).
   **Materialize every per-chunk incoming state `S_t` to global scratch** so kernel 3 can read them.
3. **Output kernel** (chunk-parallel, grid (H, n_seqs, NT)): `O_t = scale·(Q_t·S_{t-1} + B·U)` — the q-readout
   (off critical path) + intra-chunk B·U, now at 1024-wide parallelism, fully overlapped.

**Cost:** materializing per-chunk states = n_chunks·H·n_seqs·S_v²·4B. At 2048 tok (32 chunks), H=32, seq=1 =
**~2.1 GB** scratch (8192 tok → ~8.4 GB). Feasible on 24GB via `ctx.pool()` but heavy; may need to cap or tile.

**Effort/risk:** HIGH — essentially rewrites the main kernel into 2 (state-passing + output) + state
materialization wiring. Multiple Fable plans + validation. Design risk LOW (known-good fla/Mamba2 structure),
implementation surface LARGE. Payoff highest-ceiling but B=1 gain unproven ("several-fold" is optimistic; we
already have >1 wave at 128 blocks). **Do C only if the goal becomes "win decisively at B=1 / push crossover far
down", or when moving to A100-class HW where TF32 is 8× (then B+C compound into a real prefill win).**

**Prereq before C:** the ncu profile from B-step-0 (is it occupancy/latency/throughput bound?) directly decides
whether C's extra parallelism is the right fix.

---

## Honest ceiling reminder
GDN op ≈ 20% of prefill GPU time. Kernel 5% → ~1% E2E; kernel 2× → ~3-4% E2E prefill. On GA102 the physics caps
TF32 at ~parity/small-win; bf16 doubles the tensor headroom but may be latency-bound at B=1. A decisive B=1 win
likely needs C (occupancy) AND bf16 (throughput), or different HW. We already extracted more than the community
kept (they abandoned/gated theirs). A is the shipped win; B is the reasonable stretch; C is research.
