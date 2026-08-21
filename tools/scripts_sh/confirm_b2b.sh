#!/usr/bin/env bash
# Confirm the §B2b patch flips the KV buffer from pageable CPU to pinned CUDA_Host, per layer.
set -u
BIN=/home/augus/src/slop.cpp-main/build/bin/llama-server
MODEL=/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
PORT=8095
for arm in base pin; do
  envp=""; [ "$arm" = pin ] && envp="GGML_KV_PIN_HOST=1"
  pkill -9 -f "port $PORT" 2>/dev/null; sleep 1
  echo "======== arm=$arm ($envp) ========"
  env $envp "$BIN" -m "$MODEL" -fa on --n-cpu-moe 6 --no-kv-offload --ctx-size 4096 \
    --cache-type-k q8_0 --cache-type-v q8_0 --verbose --host 127.0.0.1 --port "$PORT" \
    > /tmp/verb_$arm.log 2>&1 &
  SRV=$!
  for i in $(seq 1 120); do
    curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && break
    kill -0 $SRV 2>/dev/null || break; sleep 1
  done
  echo "  per-layer KV dev assignment (count x device):"
  grep -oE "dev = [A-Za-z_()0-9]+" /tmp/verb_$arm.log | sort | uniq -c | sed 's/^/     /'
  kill -9 $SRV 2>/dev/null; sleep 2
done
