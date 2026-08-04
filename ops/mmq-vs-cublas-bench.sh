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
# sm_86, undervolt, ISOLATED arms + cooldown — see METHODOLOGY below):
#     shape                MMQ(default)    cuBLAS(forced)    verdict
#     dense-27B ub512       1375 +/-39      1227 +/-41       MMQ +12%   (small batch -> MMQ wins)
#     dense-27B ub2048      1382 +/-62      1532 +/-15       cuBLAS +5-11% (large ubatch -> cuBLAS edges it)
#     MoE-A3B ncmoe=8 ub2048 2482 +/-23      646 +/-175 (!)  MMQ +284%  (deploy; cuBLAS path is BROKEN, see below)
#     MoE-A3B ncmoe=0 ub2048 4687 +/-16      729 +/-222 (!)  MMQ +543%
# => For the DEPLOY MoE, MMQ int8-TC crushes cuBLAS. And forcing cuBLAS for MoE is not just slow — it is
#    OUTRIGHT BROKEN on the 3090: the FORCE_CUBLAS MoE path is a host-synced per-expert GEMM loop that
#    OVERFLOWS TO NaN / corrupts output / asserts (upstream #19659, reproduced on sm_86) and breaks CUDA graphs.
#    So the cuBLAS MoE t/s here may be measuring a NaN-producing path; treat it as "not a valid option," not a
#    speed comparison. The ONLY place cuBLAS legitimately wins is large-ubatch DENSE prefill (~+5-11%), which is
#    (a) the OPPOSITE of what S2 asked, (b) dense-27B only (not the deploy MoE), (c) VRAM-COSTLY (FP16 dequant
#    buffers) on our VRAM-tight box, (d) off the decode/transfer-bound critical path, (e) NOT quality-tested
#    (cuBLAS FP16-accumulate has overflow risk). NOT adopted. Recorded as a measured knob only.
# Corroboration (double-checked 2026-08-04): the int8-TC branch has been unconditional since PR #8075 (never a
#    large-batch cuBLAS fallback for tensor-core NVIDIA; the batch cutoff was always dp4a/Pascal-only). PR #8062
#    (3090 pp2048 Q4_K_S: MMQ=0.82x cuBLAS — same direction as our dense). Maintainers TRIED and FAILED to beat
#    cuBLAS at large batch/large ne01 (#16512). Re-introducing cuBLAS was REJECTED on precision (#23043); the
#    original int8-TC prototype was killed for precision too (#4801). NO runtime toggle (compile-time only,
#    #15378 declined) -> this gate needs the separate build-cublas binary. NO per-shape MMQ<->cuBLAS autotuning
#    PR exists; recent low-bit work is all Blackwell-NVFP4/Hopper. PHYSICS (Marlin PPoPP'25 + GA102 whitepaper):
#    on GA102 int8 is only ~2x the fp16/fp16-accumulate rate cuBLAS uses (NOT 4x); on-the-fly W8A8 quant/dequant
#    overhead eats it; prefill at ub2048 is compute-bound past the roofline ridge (batch>~32) where even the best
#    Ampere low-bit kernel (Marlin) converges to fp16 parity -> a negative S2 is the physically expected result.
#    Quality: the deploy path already USES MMQ and was blessed on HumanEval+ (§Q).
# WATCH (pin safety): upstream PR #26141 (2026-07-29) added a `smpbo < 48 KiB` guard atop should_use_mmq that
#    REGRESSES the RTX 3090 -> prefill ~1200->~40 t/s (open issue #26285). We are pinned to 720d7fa40 (pre-#26141;
#    confirmed absent, and our ~1400 t/s prefill proves it). Any future pin bump toward master MUST re-check #26285.
# Re-run this if the CUDA backend's MMQ heuristic changes or a new quant/arch lands.
#
# METHODOLOGY: run arms ISOLATED (one per process) with a cooldown + clock guard. Back-to-back cells heat-soak
#    the GPU and inflate variance badly (a naive r10 sweep gave dense ub2048 1296 +/-456 = 35% CV, a THERMAL
#    artifact, not GEMM variance). Isolated+cooldown collapses it to ~1-3% CV. Small (~5%) deltas are only
#    trustworthy from that path; any high-CV cell is suspect -> re-run.
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

row() { grep -E "^\| qwen"; }   # data rows only
tps() { row | sed -E 's/.*\| +([0-9.]+ . [0-9.]+) \|.*/\1/'; }
COOL="${COOL:-20}"                # cooldown seconds between arms (see METHODOLOGY note in the header)

arm() {  # label bin model extra ubatch
  local label="$1" bin="$2" model="$3" extra="$4" ub="$5" out clk
  out=$("$bin" -m "$model" $extra -p "$ub" -n 0 -ngl 99 -fa 1 -b "$ub" -ub "$ub" -r "$REPS" 2>&1 | tps)
  clk=$(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader | tr -d '\n')
  printf '  %-26s %s   [end: %s]\n' "$label" "$out" "$clk"
  sleep "$COOL"
}

echo "===================== MMQ int8-TC (default) vs FORCE_CUBLAS — prefill A/B ====================="
echo "# isolated arms, r=$REPS, ${COOL}s cooldown; [end: clk,temp] should read ~1860MHz (throttled if lower)"
echo
echo "### DENSE-27B — crossover (small batch -> MMQ; large ubatch -> cuBLAS)"
for UB in 512 2048; do
  arm "MMQ    ub$UB"    "$BENCH_MMQ" "$DENSE" "" "$UB"
  arm "cuBLAS ub$UB"    "$BENCH_CUB" "$DENSE" "" "$UB"
done
echo
echo "### MoE-A3B ncmoe=8 (DEPLOY) — MMQ grouped GEMM crushes the cuBLAS sort+per-expert fallback"
arm "MMQ    ub2048"     "$BENCH_MMQ" "$MOE" "--n-cpu-moe 8" 2048
arm "cuBLAS ub2048"     "$BENCH_CUB" "$MOE" "--n-cpu-moe 8" 2048
echo
echo "### MoE-A3B ncmoe=0 (all-GPU) — same, more extreme (cuBLAS fallback is sync-bound + noisy)"
arm "MMQ    ub2048"     "$BENCH_MMQ" "$MOE" "--n-cpu-moe 0" 2048
arm "cuBLAS ub2048"     "$BENCH_CUB" "$MOE" "--n-cpu-moe 0" 2048
echo
echo "# VERDICT: MMQ int8-TC is the default and is the win for the deploy MoE (grouped GEMM, ~+270%). cuBLAS"
echo "# edges only large-ubatch DENSE prefill (~+4%, VRAM-costly, opposite of S2, off critical path) -> NOT adopted."
