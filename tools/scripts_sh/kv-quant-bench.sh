#!/usr/bin/env bash
# kv-quant-bench.sh — the standing KV-cache-quant regression gate (IDEAS_BACKLOG A3).
#
# WHY: A3 asked whether we can beat the deployed symmetric q4_0 KV by (a) quantizing K and V
# ASYMMETRICALLY, or (b) a fancier codec (iq4_nl; SAW-INT4; sub-4-bit TurboQuant/KVarN). Robust pass
# on the deploy MoE (Qwen3.6-35B-A3B Q4_K_M, ncmoe=8, -fa on, decode @8k depth, base 720d7fa40;
# 6 reps/arm, each isolated process + 25s cooldown, undervolt clock-stable):
#     type_k  type_v    tg64 @ d8192 (t/s)   95% CI          vs q4_0   on-GPU?
#     q4_0    q4_0       88.55 ± 0.84         [87.7, 89.4]    baseline  yes (fused FA; lossless per §Q)
#     q8_0    q8_0       89.80 ± 3.30         [86.3, 93.3]    ~0        yes (lossless, more VRAM)
#     q8_0    q4_0       38.42 ± 10.65        [27.2, 49.6]    -57%      NO  (CPU offload)
#     iq4_nl  iq4_nl     16.11 ± 0.55         [15.5, 16.7]    -82%      NO  (CPU offload)
# => VERDICT (2026-08-04, double-checked): A3 CLOSED, NEGATIVE / already-optimal. Full record A3_KV_QUANT.md.
#    - MECHANISM (source-verified, ggml-cuda/fattn.cu): the default build compiles only 4 SYMMETRIC FA KV
#      combos (f16/f16, q4_0/q4_0, q8_0/q8_0, bf16/bf16). K!=V, or an unwhitelisted type, -> BEST_FATTN_KERNEL_NONE
#      -> the attention op is scheduled on the CPU backend (a CPU KV buffer is allocated; upstream #20866 shows
#      ~156 MiB). It is NOT "FA disabled". Build-flag-gated: GGML_CUDA_FA_ALL_QUANTS=OFF in our build; ON would put
#      asymmetric on-GPU (#20866: ~25x prefill recovery) but it's still DOMINATED (q4 lossless -> asymmetric =
#      more VRAM, zero quality). iq4_nl has NO FA kernel on ANY arch, flag or not (universal, not sm_86-specific).
#      The CPU-offload penalty GROWS with depth (#20866 prefill: asym 30.6 vs sym 1340 t/s, -98%).
#    - SAW-INT4 is a 4-BIT method (arXiv:2604.19157; block-diagonal-Hadamard + token-wise INT4), near-lossless on
#      H100+Triton+FA3+forked-SGLang; NOT sub-4-bit and NOT stock-llama.cpp-portable. Its value (quality-at-4-bit)
#      is null for us: q4 is already lossless on this GDN hybrid (QK-Norm kills the outliers INT4 fights).
#    - Genuinely sub-4-bit (TurboQuant tq3_0/KVarN/OSCAR) is NOT upstream (discussion #20969; unverified 3090 fork
#      spiritbuun/llama-cpp-turboquant-cuda). Even if added it's dominated: q4 KV ~6.5 MiB/1k -> ~0.83 GB @128k, so
#      it frees < one --n-cpu-moe step (0.46 GB) and BUYS ZERO CONTEXT (q4 already reaches native 262k in VRAM,
#      lossless, ~3 GB free). Physics: batch-1 decode is weight-bound, KV ~3% of bytes moved (arXiv:2605.30571) ->
#      sub-4-bit KV is wall-clock-invisible on Ampere. The free monitor->iGPU replug (~1.4 GB=~3 steps) dominates.
#    - DENSE-27B: growing memory at depth is the full-precision Gated-DeltaNet RECURRENT STATE (Phase A #3),
#      untouched by any --cache-type -> a better KV codec cannot extend dense context either.
#    DEPLOY: keep symmetric q4_0 KV (fast + lossless); q8_0 symmetric if wanted. Never asymmetric (-57%)/iq4_nl (-82%).
#
# RE-OPEN trigger: only if a sub-4-bit fused-FA KV kernel lands UPSTREAM for sm_86 (watch KV-quant PRs / #20969),
# or a future served model becomes genuinely KV-bound rather than weight/ncmoe-bound (it isn't for our hybrids).
#
# Usage: bash ops/kv-quant-bench.sh   (runs in WSL Ubuntu-24.04; ~10 min for 4 arms).
set -u
cd /home/augus/src/slop.cpp-main
export CUDA_VISIBLE_DEVICES=0
M=${MODEL:-/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
NCMOE=${NCMOE:-8}
DEPTH=${DEPTH:-8192}
REPS=${REPS:-6}
BIN=./build/bin/llama-bench

arm() { # ctk ctv label
  echo "########## $3  ctk=$1 ctv=$2 ##########"
  sleep 25   # cooldown between isolated arms (guards heat-soak variance — the S2/S3 GPU-A/B rule)
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader
  # NB: use llama-bench, NOT llama-cli -no-cnv (the latter hangs on stdin and block-buffers stderr to a file).
  # Each arm is a fresh llama-bench process (isolation); the broken CPU-offload arms are intrinsically noisy (~28% CV).
  stdbuf -oL -eL "$BIN" -m "$M" -fa on -ncmoe "$NCMOE" \
    -ctk "$1" -ctv "$2" -p 0 -n 64 -d "$DEPTH" -r "$REPS" 2>&1 | grep -iE "\| qwen|error|unsupported"
  echo
}

arm q4_0  q4_0    "SYMMETRIC q4 (deploy baseline)"
arm q8_0  q8_0    "SYMMETRIC q8 (other lossless on-GPU option)"
arm q8_0  q4_0    "ASYMMETRIC k8/v4 (doc: preserve-K compress-V)"
arm iq4_nl iq4_nl "SYMMETRIC iq4_nl"
echo ALL_DONE
