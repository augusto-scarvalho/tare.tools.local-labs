#!/bin/bash
# A3 — iq4_nl symmetric datapoint (CONTEXT_PLAN flagged it as a better 4-bit at same size).
# Compare to q4_0 baseline (87.4 t/s @ d8192). Same size => no VRAM win; q4 already lossless => no quality win.
set -u
cd /home/augus/src/llama.cpp-master
export CUDA_VISIBLE_DEVICES=0
M=/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
OUT=/home/augus/a3_out
mkdir -p "$OUT"
BIN=./build/bin/llama-bench

echo "########## iq4_nl symmetric ##########"
stdbuf -oL -eL timeout 300 "$BIN" -m "$M" -fa on -ncmoe 8 \
  -ctk iq4_nl -ctv iq4_nl -p 0 -n 64 -d 8192 -r 3 > "$OUT/iq4nl.log" 2>&1
echo "exit=$?"
grep -iE "\| qwen|error|unsupported|not supported" "$OUT/iq4nl.log" | tail -6
echo ALL_DONE
