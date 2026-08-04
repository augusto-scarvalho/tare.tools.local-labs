#!/usr/bin/env bash
# mmq-vs-cublas-bench.sh — the standing GEMM-path gate for spec-decode/prefill (IDEAS_BACKLOG S2).
#
# WHY: S2 asked us to BUILD a "fused dequant + INT8-Tensor-Core GEMM so low-bit weights use the 3090's
# INT8 tensor cores instead of dequant-to-FP16-then-cuBLAS." That kernel ALREADY EXISTS and is the DEFAULT:
# llama.cpp's MMQ path. On Ampere `ggml_cuda_should_use_mmq(Q4_K, sm_86, any batch)` returns true
# UNCONDITIONALLY (mmq.cu:310 `turing_mma_available` early-return; the ne11<MAX_BATCH cutoff only applies to
# non-tensor-core cards). MMQ uses `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32` = INT8 TC, loads weights
# quantized + dequants in-register into int8 tiles (never materializes an FP16 weight matrix). Upstream made
# it default in PR #8075 — primarily for VRAM savings, explicitly accepting a large-batch speed hit.
# => S2-as-proposed has ZERO headroom: the int8-TC kernel is already default and already the win.
#
# This gate A/Bs the default (MMQ int8-TC) vs a FORCE_CUBLAS build (the old dequant->FP16->cuBLAS path) =
# exactly the S2 delta, in reverse. Rigorously measured 2026-08-04 (prefill pp512/pp2048, -ub 2048, r5, arch
# sm_86, undervolt clock-stable):
#     shape                 MMQ(default)   cuBLAS(forced)   verdict
#     dense-27B pp512         1336            1203           MMQ +11%  (small batch -> MMQ wins)
#     dense-27B pp2048        1448            1522           cuBLAS +5% (large ubatch -> cuBLAS edges it)
#     MoE-A3B ncmoe=0 pp2048  4494             864           MMQ +420% (grouped per-expert GEMM: cuBLAS dies)
#     MoE-A3B ncmoe=8 pp2048  2140             581           MMQ +268% (deploy config)
# => For the DEPLOY MoE, MMQ int8-TC crushes cuBLAS (small per-expert batches) — forcing cuBLAS is catastrophic.
#    The ONLY place cuBLAS wins is large-ubatch DENSE prefill (+5%), which is (a) the OPPOSITE of what S2 asked,
#    (b) dense-27B only (not the deploy model), (c) VRAM-COSTLY (FP16 dequant buffers) on our VRAM-tight box,
#    (d) off the decode/transfer-bound critical path. NOT adopted. Recorded as a measured knob only.
# Corroboration: upstream PR #8075 (MMQ default, VRAM-motivated), PR #8062 (3090 pp2048 Q4_K_S: MMQ=0.82x
#    cuBLAS — same direction as our dense +5%), PR #7921 (int8-TC k-quant kernels, Q8_1 activation precision
#    "negligible"), docs/build.md (FORCE_CUBLAS "faster large-batch but more VRAM + FP16 overflow risk";
#    MMQ int32-accumulates so it's the MORE numerically robust path). No open/rejected PR proposes a fused
#    int8-TC GEMM beyond MMQ — it IS that kernel and is treated as mature. Quality: the deploy path already
#    USES MMQ and was blessed on HumanEval+ (§Q); FORCE_CUBLAS is the deviation and is not adopted.
# Re-run this if the CUDA backend's MMQ heuristic changes or a new quant/arch lands.
#
# Usage: mmq-vs-cublas-bench.sh              (builds build-cublas on demand, A/Bs dense + MoE prefill)
#        REPS=3 mmq-vs-cublas-bench.sh
set -u
SRC=/home/augus/src/llama.cpp-master
BENCH_MMQ="$SRC/build/bin/llama-bench"                 # default build = MMQ int8-TC
BENCH_CUB="$SRC/build-cublas/bin/llama-bench"          # FORCE_CUBLAS build = dequant->FP16->cuBLAS
DENSE="${DENSE:-/home/augus/models/qwen36-27b-dense/Qwen_Qwen3.6-27B-Q4_K_M.gguf}"
MOE="${MOE:-/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
REPS="${REPS:-5}"

# Build the FORCE_CUBLAS A/B binary on demand (regenerable; not kept in-tree). Compile flag only, no source edit.
if [ ! -x "$BENCH_CUB" ]; then
  echo "### configuring + building build-cublas (GGML_CUDA_FORCE_CUBLAS=ON) ..."
  cmake -S "$SRC" -B "$SRC/build-cublas" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DGGML_CUDA_FORCE_CUBLAS=ON -DLLAMA_CURL=OFF >/dev/null 2>&1 || { echo "cmake configure FAILED"; exit 1; }
  cmake --build "$SRC/build-cublas" --target llama-bench -j "$(nproc)" >/dev/null 2>&1 \
    || { echo "build FAILED"; exit 1; }
fi

row() { grep -E "^\| (qwen|model)"; }   # keep header + data rows from llama-bench markdown

echo "===================== MMQ int8-TC (default) vs FORCE_CUBLAS — prefill A/B ====================="
for pair in "DENSE-27B|$DENSE|" "MoE-A3B ncmoe=0|$MOE|--n-cpu-moe 0" "MoE-A3B ncmoe=8 (deploy)|$MOE|--n-cpu-moe 8"; do
  label="${pair%%|*}"; rest="${pair#*|}"; model="${rest%%|*}"; extra="${rest#*|}"
  echo; echo "### $label"
  echo "--- MMQ (default) ---"
  "$BENCH_MMQ" -m "$model" $extra -p 512,2048 -n 0 -ngl 99 -fa 1 -b 2048 -ub 2048 -r "$REPS" 2>&1 | row
  echo "--- FORCE_CUBLAS ---"
  "$BENCH_CUB" -m "$model" $extra -p 512,2048 -n 0 -ngl 99 -fa 1 -b 2048 -ub 2048 -r "$REPS" 2>&1 | row
done
echo
echo "# VERDICT: MMQ int8-TC is the default and is the win for the deploy MoE (grouped GEMM). cuBLAS edges"
echo "# only large-ubatch DENSE prefill (+~5%, VRAM-costly, opposite of S2, off critical path) -> NOT adopted."
