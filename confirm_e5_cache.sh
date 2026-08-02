#!/usr/bin/env bash
# §E5 — confirm the stack server ENGAGES the MoE expert cache with our routing profile.
set -u
BIN=/home/augus/src/llama.cpp-stack/build/bin/llama-server
MODEL=/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
PROF=/home/augus/models/qwen36-35b-moe-trace.csv
PORT=8096
[ -x "$BIN" ] || { echo "REFUSING: no stack server"; exit 2; }
[ -f "$PROF" ] || { echo "REFUSING: no profile"; exit 2; }
pkill -9 -f "port $PORT" 2>/dev/null; sleep 1
echo "== launch ncmoe=40 + --moe-cache-slots 32 --moe-cache-profile =="
"$BIN" -m "$MODEL" -fa on --n-cpu-moe 40 --moe-cache-slots 32 --moe-cache-profile "$PROF" \
  --ctx-size 4096 --cache-type-k q8_0 --cache-type-v q8_0 --host 127.0.0.1 --port "$PORT" \
  > /tmp/e5_confirm.log 2>&1 &
SRV=$!
for i in $(seq 1 200); do
  curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && break
  kill -0 $SRV 2>/dev/null || { echo "DIED:"; tail -25 /tmp/e5_confirm.log; exit 1; }
  sleep 1
done
echo "== VRAM while loaded =="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo "== cache/moe log lines =="
grep -iE "moe.?cache|cache.?slot|resident|hot.?pack|profile|expert.*cache|cached" /tmp/e5_confirm.log | head -20
echo "== (context) any 'moe' lines =="
grep -iE "moe" /tmp/e5_confirm.log | head -10
kill -9 $SRV 2>/dev/null
