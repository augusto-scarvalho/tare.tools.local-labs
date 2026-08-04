#!/usr/bin/env bash
# kv-quant-bench.sh — the standing KV-cache-quant regression gate (IDEAS_BACKLOG A3).
#
# WHY: A3 asked whether we can beat the deployed symmetric q4_0 KV by (a) quantizing K and V
# ASYMMETRICALLY, or (b) a fancier 4-bit codec (iq4_nl; and, if ever engine-added, SAW-INT4 /
# TurboQuant sub-4-bit). Measured on the deploy MoE (Qwen3.6-35B-A3B Q4_K_M, ncmoe=8, -fa on,
# decode at 8k depth, 3 reps, GPU clock stable via the undervolt), base 720d7fa40:
#     type_k  type_v    tg64 @ d8192 (t/s)     vs q4_0
#     q4_0    q4_0       87.36 ± 1.81           baseline (deploy; lossless per §Q/CONTEXT_PLAN)
#     q8_0    q4_0       33.23 ± 2.09           -62%   (asymmetric)
#     q4_0    q8_0       33.60 ± 2.28           -62%   (asymmetric, other order — same penalty)
#     iq4_nl  iq4_nl     18.48 ± 4.49           -79%   (symmetric, but off the fused-FA fast path on sm_86)
# => VERDICT (2026-08-04): A3 CLOSED, NEGATIVE / already-optimal.
#    - ASYMMETRIC K/V craters decode ~62% (llama.cpp #20866 CPU-KV fallback; reproduces on our base).
#      The penalty is symmetric in K vs V — any mismatch triggers it. Never run asymmetric KV here.
#    - iq4_nl (the one untested "better 4-bit" flagged in CONTEXT_PLAN §D) falls off the fused flash-attn
#      KV fast path on Ampere sm_86 EVEN symmetric -> -79%. The only fast KV types on this box are q4_0/q8_0.
#    - Sub-4-bit codecs (SAW-INT4/TurboQuant tq3_0/tq4_0) are NOT in the engine (excluded from the fork).
#      Even if added, paper math on Phase-A geometry (MoE q4 KV ~6.5 MiB/1k -> ~0.83 GB @128k) shows they'd
#      free < one --n-cpu-moe step (0.46 GB) at 128k and BUY ZERO CONTEXT (q4 already reaches native 262k in
#      VRAM, lossless, with ~3 GB free). At the 8k deploy default KV is ~52 MiB -> shaving it is a rounding
#      error. The free monitor->iGPU replug (~1.4 GB = ~3 ncmoe steps) strictly dominates this whole path.
#    - DENSE-27B: its growing memory at depth is the full-precision Gated-DeltaNet RECURRENT STATE (Phase A
#      finding #3: q4 barely shrinks dense VRAM), which NO --cache-type touches -> a better KV codec cannot
#      extend dense context either.
#    DEPLOY: keep symmetric q4_0 KV for long context (fast + lossless); q8_0 symmetric if ever wanted.
#
# RE-OPEN trigger: only if a sub-4-bit fused-FA KV kernel lands UPSTREAM for sm_86 (watch llama.cpp KV-quant
# PRs), or a future served model is KV-bound rather than weight/ncmoe-bound (it isn't for our hybrids).
#
# Usage: bash ops/kv-quant-bench.sh   (runs in WSL Ubuntu-24.04; ~10 min for 4 arms).
set -u
cd /home/augus/src/llama.cpp-master
export CUDA_VISIBLE_DEVICES=0
M=${MODEL:-/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
NCMOE=${NCMOE:-8}
DEPTH=${DEPTH:-8192}
REPS=${REPS:-3}
BIN=./build/bin/llama-bench

arm() { # ctk ctv label
  echo "########## $3  ctk=$1 ctv=$2 ##########"
  # NB: use llama-bench, NOT llama-cli -no-cnv (the latter hangs on stdin and block-buffers stderr to a file).
  stdbuf -oL -eL "$BIN" -m "$M" -fa on -ncmoe "$NCMOE" \
    -ctk "$1" -ctv "$2" -p 0 -n 64 -d "$DEPTH" -r "$REPS" 2>&1 | grep -iE "\| qwen|error|unsupported"
  echo
}

arm q4_0  q4_0   "SYMMETRIC q4 (deploy baseline)"
arm q8_0  q4_0   "ASYMMETRIC k8/v4"
arm q4_0  q8_0   "ASYMMETRIC k4/v8"
arm iq4_nl iq4_nl "SYMMETRIC iq4_nl"
echo ALL_DONE
