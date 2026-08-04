# GDN chunked kernel — faster hybrid prefill (design + progress)

Goal: replace the **sequential** Gated-DeltaNet CUDA kernel with a **chunk-parallel**
(WY / delta-rule-over-sequence) kernel to speed up **prefill** of the Qwen3.6 hybrid
models. This is the upstream `//TODO: Add chunked kernel for even faster pre-fill`
(`ggml/src/ggml-cuda/gated_delta_net.cu:180`). Decode is untouched (bandwidth-bound; see below).

## Why (measured, 2026-08-03)
`nsys` per-kernel share of **prefill** GPU time (`llama-bench -p N -n 0 -ub 2048`, fork `ed4572556`):

| model / ctx | `gated_delta_net_cuda` | rank | flash_attn |
|---|---:|:--:|---:|
| Dense-27B, 8k | 15.0% | #3 | 3.9% |
| **MoE-35B-A3B, 8k** | **22.5%** | **#1** | 4.3% |
| **MoE-35B-A3B, 32k** | **20.4%** | **#1** | 14.4% |

The GDN kernel is the **#1 prefill kernel on the deploy MoE** (sparse FFN is cheap, so the
serial scan dominates). Attention grows as O(N²) and dilutes the share at long ctx, but GDN
stays 20%+ through 32k. A 2–4× cut → **~10–17% MoE prefill** (~4–6 s off a 128k prefill).

Decode is NOT a target: dense decode = 43 t/s = **83% of the 3090 weight-bandwidth ceiling**
(16.74 GiB/tok ÷ 936 GB/s ≈ 52 t/s). No kernel helps decode; a chunked scan only helps prefill
(n_tokens=1 at decode → one scan step).

> Doc fix owed: SERVING.md says the dense "fused GDN kernel is disabled" and dense ≈ 33 t/s.
> Both wrong — the CUDA kernel is present at base `720d7fa40` and enabled; dense decode is 43 t/s.

## The recurrence (what we must match)
Per token t, state `S` is `[S_v,S_v]`, convention `out = Sᵀx` (ggml `ops.cpp`
`gated_delta_net_one_chunk`), scalar gating `d_t=exp(g_t)`, `scale=1/√S_v`:
```
S  <- d_t * S                    # gated decay
u_t = beta_t * (v_t - Sᵀ k_t)    # delta correction, decayed state
S  <- S + k_t ⊗ u_t              # rank-1 update
o_t = scale * Sᵀ q_t             # readout, updated state
```

## Chunk-parallel formulation — VALIDATED to ~1e-16 (M1)
Per chunk of `n` tokens, incoming state `S0`, cumulative decay `P_t = Π_{s≤t} d_s`:
```
KK[t,r]  = k_t·k_r
A[t,r]   = beta_t * (P_t/P_r) * KK[t,r]     for r<t     (strictly lower-tri)
w_t      = beta_t * (v_t - P_t * S0ᵀ k_t)
U        = (I + A)^{-1} W                    # unit lower-tri solve (fwd subst), n×S_v
QK[t,r]  = q_t·k_r
Binc     = tril(P_t/P_r) ⊙ QK               # r<=t inclusive
O        = scale * ( P ⊙ (Q S0) + Binc U )
S_next   = P_{n-1} * S0 + Σ_r (P_{n-1}/P_r) k_r ⊗ u_r
```
numpy ref + validation: `scratchpad/gdn_chunk_proto.py` — max err 8.9e-16 across
C∈{1,63,64,128,256,2048}, chunk∈{32,64,128}. **Exact, not approximate** → bless-friendly.

## Milestones
- [x] **M1** numpy: chunk-parallel == sequential to <1e-4 (got ~1e-16). `gdn_chunk_proto.py`.
- [x] **M2** ggml **CPU** backend port — DONE (2026-08-03), 46/46 `test-backend-ops -o GATED_DELTA_NET -b CPU`.
      New fn `ggml_compute_forward_gated_delta_net_wy_one_chunk` in `ggml-cpu/ops.cpp`; gated in
      `_f32` on `!use_ref && K==1 && n_tokens>1` (kda included). Handles **kda** (per-dim cumsum,
      CS=16), **v_repeat** (GQA, modulo bcast), **permuted** (stride-only), multi-head/seq. `K>1`
      keeps the sequential path (chunked can't emit per-token snapshots). Scratch sized in
      `ggml-cpu.c` GATED_DELTA_NET case. Validation is FREE: `test-backend-ops -b CPU` runs the
      reference backend with `use_ref=true` → the op auto-compares chunked(use_ref=0) vs
      sequential(use_ref=1) in one process. Key fix vs the numpy proto: never form `P_t/P_r`
      (overflows at g∈[-20,0]); exponentiate **differences** `exp(c_t−c_r)≤0` only.
- [x] **M3** CUDA chunked kernel — DONE + **WINS at deploy scale** (2026-08-03). Env-gated `GGML_CUDA_GDN_CHUNKED`
      (default OFF, byte-identical to upstream); when ON, engages only at `n_tokens >= 1024` (env-tunable
      `GGML_CUDA_GDN_CHUNKED_MIN_TOKENS`, the measured crossover) → **strict no-regression**. All in
      `ggml-cuda/gated_delta_net.cu`. Design: grid (H, n_seqs, S_v/TILE_J=4) column-tiling, block **dim3(32,16)**=512
      (warp-shaped for `ggml_cuda_mma`), CS=64, padded smem **SP=S_v+4** (mult-of-4 for `ldmatrix`). Two-kernel split:
      parallel pre-pass (cs/betas/B/T=(I+A)⁻¹) + serial main (W / U=T·W / output / carry).
      **TF32 tensor-core conversion (the decisive lever):** W-readout GEMM (`S0·k̃`) → `mma.m16n8k8` gave **1.38×**
      (it's ON the inter-chunk critical path); q-readout → TF32 was **flat** (OFF the critical path, already latency-hidden).
      Both correct at NMSE 2e-7 (`GGML_CUDA_GDN_TC` default ON; =0 forces scalar, proves re-threading is a no-op).
      **Robust 3-rep median sweep (H=32,d=128, tf32/seq): T=512 1.11× slow → T=1024 0.98 WIN → T=2048 0.95 WIN →
      T=4096 0.97 WIN → T=8192 1.02 par. Crossover ≈1024; batch≥4 wins even at T=512 (occupancy).** Deploy ubatch=2048
      → ~5% faster (kernel-level; single-shot ran 2806 vs 3142µs = 11%, median 5%).
      **Why parity-not-blowout is the EXPECTED ceiling** (3-worker research, all converge + community reproduced):
      (1) GA102 TF32==FP32 (1:1; A100 is 8×) → TF32 buys no FLOP/s, only LDS relief; bf16=2× is the only headroom.
      (2) B=1 occupancy: 128 fixed blocks vs fla/Mamba2's chunk-parallel ~1024-wide (they split into 3 kernels).
      (3) Community: am17an built+ABANDONED a TF32-chunked GDN ("not worth it <16K"); Neroued's B=1 crossover ~T=1800
      matches ours; PR #26001 (live) uses fp16 WMMA+CS=16, +3-18.7% E2E only at large ubatch; no chunked GDN CUDA
      kernel has landed upstream. **Full findings + B/C next-lever briefs in `GDN_TF32_PLAN.md` + `GDN_NEXT_LEVERS.md`.**
- [x] **Lever B (bf16 W-readout) — DONE 2026-08-03: NEGATIVE RESULT, reverted.** Implemented `mma.m16n8k16.f32.bf16`
      on the critical-path W-readout (A packed from 2 fp32 ldmatrix via `__floats2bfloat162_rn`; B via direct
      `float2` smem reads — the fp32 8×8 ldmatrix gives non-adjacent k so it can't build a k16 pack; fp32
      accumulate+epilogue kept). Correct 46/46 @ NMSE 1e-4. **Perf FLAT vs TF32** (deploy 2048: 2868.8 vs 2878.7 µs
      = −0.3% noise; 4096/8192 actually +2–5% WORSE from the CVT/pack overhead). bf16 = 2× TF32 FLOP/s on GA102 yet
      bought nothing → **the B=1 chunked GDN is latency/occupancy-bound (mma-issue latency + serial inter-chunk carry
      + 1 block/SM smem-capped), NOT tensor-throughput-bound.** Precision levers exhausted; reverted (TF32 is
      strictly better: same-or-faster AND more accurate). Only Lever C (3-kernel split → occupancy) attacks the real
      bottleneck. Fable plan + hand-verified fragment layouts + baseline retained in `GDN_NEXT_LEVERS.md`.
- [~] **M4** A/B in `llama-bench` (real prefill) — DONE 2026-08-03 on the deploy MoE (Qwen3.6-35B-A3B UD-Q4_K_M,
      `-p 2048 -p 4096 -ub 2048 -r 3`, seq vs `GGML_CUDA_GDN_CHUNKED=1`). **The isolated ~5% kernel win does NOT
      surface E2E at B=1 in either placement regime:** ncmoe=8 (deploy, experts on CPU) → chunked +1.0%/+0.5%
      (within seq's ±74 noise; GDN is 22.5% of *GPU* time but prefill is partly CPU-bound); ncmoe=0 (all-GPU, 2×
      faster prefill 5490 vs 2889 t/s) → chunked −0.4%/−0.5% (noise; experts-on-GPU shrink GDN's share + the 2-kernel
      prepass adds tiny launch overhead). So at single-stream B=1 the kernel win dilutes into ±1% E2E — consistent
      with the "kernel 5% → ~1% E2E" ceiling. The T≥1024 gate keeps it strict-no-regression. **Where it should
      surface: concurrent multi-slot serving (deploy runs 8 slots, §CC) raises effective batch, and the synthetic
      sweep shows batch≥4 gives a much larger kernel win — that llama-server concurrent A/B is the pending validation.**
      Bless (quality-neutral) still owed. Model-dependence: the perf verdict is SHAPE-bound (H,d,tokens), not
      weight-dependent — verified by the real model matching the synthetic sweep.
      Lever B closed (negative); Lever C (3-kernel split) is now the primary — but future research (needs A100-class
      HW where TF32 is 8× and throughput matters, or a decisive-B=1-win goal).
- [~] **M4b** (concurrent-serving A/B + bless + shape contrast) — **IN FLIGHT, interrupted by a GPU-lost crash mid-run.
      Full state + resume order in `GDN_M4_RESUME.md`.** Key result already secured (Fable, source-read): a **hard
      structural ceiling `n_seqs <= n_ubatch / MIN_TOKENS`** — at deploy `-ub 2048` + gate 1024 the chunked kernel
      **never sees n_seqs > 2**, and ≥2048-tok prompts with `-b 2048` never co-batch across sequences at all. So
      8-slot concurrency does NOT feed the kernel more occupancy unless config deviates (Arm-B `-b 8192 -ub 2048
      -ctxcp 0 MIN_TOKENS=128`, or Arm-C `-ub 8192`). Driver saved: `gdn_conc_bench.py`. Shape investigation: MoE=H32,
      dense-27B=H48 (free "more heads" contrast on-disk); download judged redundant. Occupancy sweep cases (H=48/64)
      compiled into test-backend-ops but the chunked columns still need re-running post-reboot. **GPU fell off the
      PCIe bus** (RTX 3090 "GPU is lost", host-level, needs reboot) during the occupancy sweep — avoid `ncu`, don't
      overlap heavy GPU benches with interactive PC use.

## M3 hazards (known before we start)
- **State too big for shared mem:** `S_v=128 → 128×128×4 = 64 KB` per head > 48 KB smem.
  Options: tile the state, keep in registers across a warp-group, or fp16 accum with fp32 correction.
- Intra-chunk `(I+A)` solve is sequential over `n` (chunk=64) — do it in smem per (head,seq),
  then the KKᵀ/QKᵀ/`U`/output steps as tensor-core matmuls.
- Reduction order differs from the sequential kernel → expect ~1e-3 rel, not bit-identical;
  gate it and bless for quality-neutrality (matches how MTP was handled).
- Chunk size is a tuning knob (32/64/128) — trades solve cost vs parallelism.

## Tooling
- Build: `cmake --build build --target test-backend-ops llama-bench` (Ubuntu-24.04 WSL).
- Correctness: `test-backend-ops test -o GATED_DELTA_NET` (add large-`n_seq_tokens` cases).
- Perf: `nsys profile --stats=true -t cuda ... llama-bench -p N -n 0 -ub 2048`.
