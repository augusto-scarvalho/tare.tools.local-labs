#!/usr/bin/env bash
# §E5 step 1 — generate a routing profile for Qwen3.6-35B-A3B with llama-moe-trace.
# Routing (which experts fire) is independent of placement, so trace at ncmoe=6 (fast).
set -u
BIN=/home/augus/src/llama.cpp-stack/build/bin/llama-moe-trace
MODEL=/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
OUT=/home/augus/models/qwen36-35b-moe-trace.csv
[ -x "$BIN" ] || { echo "REFUSING: no trace bin"; exit 2; }

PROMPT="Explain in detail, with worked numbers, why memory bandwidth rather than raw compute limits token generation on a single consumer GPU. Then write a Python function that merges two sorted lists, and describe three ways a mixture-of-experts model routes tokens to experts. Cover arithmetic intensity, batch-size-one, PCIe transfer of offloaded experts, and KV-cache growth. Be concrete and thorough."

echo "== tracing routing (ncmoe=6, decode ~400 tok) =="
MOE_TRACE_OUT="$OUT" "$BIN" -m "$MODEL" -fa on --n-cpu-moe 6 --ctx-size 4096 \
  -p "$PROMPT" -n 400 2>/tmp/trace_run.log
echo "exit=$?"
echo "== trace file =="
ls -la "$OUT"
echo "rows: $(wc -l < "$OUT")"
echo "== head =="; head -3 "$OUT"
echo "== decode rows only (pos>=0), per-layer distinct experts touched =="
awk -F, '$1>=0 {for(i=3;i<=NF;i++) seen[$2","$i]=1} END{for(k in seen){split(k,a,",");cnt[a[1]]++} for(l=0;l<40;l++) printf "layer %2d: %d distinct experts\n", l, cnt[l]}' "$OUT" | head -45
