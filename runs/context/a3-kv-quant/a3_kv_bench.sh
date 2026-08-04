#!/bin/bash
# A3 — asymmetric vs symmetric K/V: buffer placement + decode t/s.
# llama-bench never reads stdin and exits cleanly (unlike llama-cli -no-cnv).
set -u
cd /home/augus/src/llama.cpp-master
export CUDA_VISIBLE_DEVICES=0
M=/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
OUT=/home/augus/a3_out
mkdir -p "$OUT"
BIN=./build/bin/llama-bench

run() {
  local ctk="$1" ctv="$2" tag="$3"
  echo "########## $tag  ctk=$ctk ctv=$ctv ##########"
  # -d 8192: decode at 8k depth so KV is populated; a CPU-KV fallback tanks tg here.
  stdbuf -oL -eL timeout 300 "$BIN" -m "$M" -fa on -ncmoe 8 \
    -ctk "$ctk" -ctv "$ctv" -p 0 -n 64 -d 8192 -r 3 > "$OUT/$tag.log" 2>&1
  echo "exit=$? (124=timeout)"
  echo "--- KV buffer placement (CUDA vs CPU) ---"
  grep -iE "KV buffer|CPU KV|CUDA0 KV|kv_cache" "$OUT/$tag.log" | head -12
  echo "--- decode result ---"
  grep -iE "\btg64\b|tg |n_depth|\| *tg" "$OUT/$tag.log" | tail -6
  echo
}

run q4_0 q4_0 sym_q4
run q8_0 q4_0 asym_k8v4
run q4_0 q8_0 asym_k4v8
echo ALL_DONE
