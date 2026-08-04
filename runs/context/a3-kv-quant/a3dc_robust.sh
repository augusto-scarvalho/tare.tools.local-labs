#!/bin/bash
# A3 double-check — robust KV-quant bench: isolated processes + cooldown + 6 reps + depth point.
# Fixes the first pass's stat gap (iq4_nl had 24% CV, 3 reps, no cooldown between arms).
set -u
cd /home/augus/src/llama.cpp-master
export CUDA_VISIBLE_DEVICES=0
M=/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
OUT=/home/augus/a3dc_out
mkdir -p "$OUT"
BIN=./build/bin/llama-bench

cool() { # wait for GPU to settle (undervolt keeps clocks stable; this guards heat-soak)
  sleep 25
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader
}

arm() { # ctk ctv depth reps tag
  echo "########## $5  ctk=$1 ctv=$2 depth=$3 reps=$4 ##########"
  echo -n "pre-arm GPU (temp,sm_clk,mem): "; cool
  stdbuf -oL -eL timeout 900 "$BIN" -m "$M" -fa on -ncmoe 8 \
    -ctk "$1" -ctv "$2" -p 0 -n 64 -d "$3" -r "$4" 2>&1 | grep -iE "\| qwen|error|unsupported|CPU|offload"
  echo
}

echo "===== 8k depth, 6 reps each, isolated + 25s cooldown (robust stats pass) ====="
arm q4_0   q4_0   8192  6 "SYM q4 (deploy baseline)"
arm q8_0   q8_0   8192  6 "SYM q8 (other lossless on-GPU option)"
arm q8_0   q4_0   8192  6 "ASYM q8/q4 (doc: preserve-K compress-V)"
arm iq4_nl iq4_nl 8192  6 "SYM iq4_nl"

echo "===== depth confirm: the WORKING (symmetric on-GPU) path stays fast at depth ====="
# (The BROKEN paths' depth-worsening is covered by mechanism + #20866 prefill collapse 1340->30 t/s;
#  re-running a 32k CPU-offloaded prefill here would take ~15+ min, so we cite #20866 instead.)
arm q4_0   q4_0   32768 5 "SYM q4 @32k"
echo ALL_DONE
