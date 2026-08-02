#!/usr/bin/env bash
# CONTEXT_PLAN Phase A — the real context frontier. Load the model at increasing -c and read
# the ACTUAL KV buffer size (llama.cpp logs it) + total VRAM. Tests the hybrid "KV is cheap"
# claim (only ~1/4 layers hold growing KV) vs the naive all-layers arithmetic. KV is allocated
# in full at load, so just loading (no decode) reveals the wall. No MTP here (isolates KV).
#   MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/projects/local-model-lifecycle/phase_a_ctx.sh <model> <ncmoe> <kv> <ctx...>
set -u
declare -A MODELS=(
  [moe]=/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  [dense]=/home/augus/models/qwen36-27b-mtp/Qwen3.6-27B-Q4_K_M.gguf )
BIN=/home/augus/src/llama.cpp-master/build/bin/llama-server
MK=${1:-moe}; NC=${2:-8}; KV=${3:-q8_0}; shift 3 2>/dev/null || shift $#
CTXS=("$@"); [ ${#CTXS[@]} -eq 0 ] && CTXS=(8192 32768 65536 131072 262144)
MODEL=${MODELS[$MK]}; PORT=8099
PLACE=(--n-cpu-moe "$NC"); [ "$MK" = dense ] && PLACE=(-ngl 99)

echo "== Phase A: $MK  placement=${PLACE[*]}  kv=$KV =="
printf "%-8s %-7s %-11s %-11s %-11s  %s\n" "ctx" "loaded" "vram_used" "vram_free" "KV_MiB" "note"
for CTX in "${CTXS[@]}"; do
  pkill -9 -f "port $PORT" 2>/dev/null; sleep 2
  "$BIN" -m "$MODEL" -fa on "${PLACE[@]}" --cache-type-k "$KV" --cache-type-v "$KV" \
    -c "$CTX" --host 127.0.0.1 --port "$PORT" > /tmp/pa.log 2>&1 &
  SRV=$!; ok=0
  for i in $(seq 1 300); do
    curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"\|ok' && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || break; sleep 1
  done
  # exact KV buffer size from the load log (sum any CUDA KV buffer lines)
  kv_mib=$(grep -iE "KV buffer size|kv_cache.*size|KV self size|KV cache" /tmp/pa.log \
           | grep -oiE "[0-9]+\.?[0-9]* MiB" | grep -oE "[0-9]+\.?[0-9]*" \
           | awk '{s+=$1} END{printf "%.0f", s}')
  if [ "$ok" = 1 ]; then
    read used free < <(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | head -1 | tr ',' ' ')
    fit=$([ "$free" -ge 4096 ] && echo "envelope-OK" || echo "under-4GB-reserve")
    printf "%-8s %-7s %-11s %-11s %-11s  %s\n" "$CTX" "yes" "${used}MiB" "${free}MiB" "${kv_mib:-?}" "$fit"
  else
    err=$(grep -iE "error|oom|out of memory|failed to allocate|cudaMalloc" /tmp/pa.log | tail -1 | cut -c1-50)
    printf "%-8s %-7s %-11s %-11s %-11s  %s\n" "$CTX" "NO" "-" "-" "${kv_mib:-?}" "${err:-died}"
  fi
  kill -9 $SRV 2>/dev/null; sleep 2
done
