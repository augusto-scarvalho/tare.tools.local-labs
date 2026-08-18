#!/usr/bin/env bash
# §E5b — the fair comparison: static --n-cpu-moe (no cache) on the SAME stack build, SAME depth,
# swept so its VRAM points bracket the cache's. Overlaid on the §E5 cache curve, this answers the
# fork question: at equal VRAM, does dynamic hot-expert caching beat static layer placement?
# Cache points (stack, depth256): off/ncmoe40=37.2t/s@4229MiB, s8=35.9@4741, s32=38.5@6491, s64=45.8@8823.
set -u
BIN=/home/augus/src/llama.cpp-stack/build/bin/llama-server
MODEL=/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
CTX=4096; GEN=120; DEPTH=256; PORT=8096
[ -x "$BIN" ] || { echo "REFUSING"; exit 2; }

decode_tps() {
  local tps
  tps=$(python3 - "$PORT" "$DEPTH" "$GEN" <<'PY'
import json,sys,urllib.request
port,depth,gen=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
prompt="bandwidth bound decode waits on memory not compute "*(max(1,depth//8))
body=json.dumps({"prompt":prompt,"n_predict":gen,"temperature":0,"cache_prompt":False,"ignore_eos":True}).encode()
req=urllib.request.Request(f"http://127.0.0.1:{port}/completion",data=body,headers={"Content-Type":"application/json"})
try: d=json.load(urllib.request.urlopen(req,timeout=900))
except Exception: print("ERR"); sys.exit(0)
t=d.get("timings",{}); pn=t.get("predicted_n",0)
print(f"{t.get('predicted_per_second',0):.2f}" if pn>=gen*0.8 else "INVALID")
PY
)
  local vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "$tps $vram"
}

echo "== STATIC --n-cpu-moe sweep (stack build, no cache, depth $DEPTH) =="
printf "%-8s %-10s %-10s\n" "ncmoe" "decode t/s" "vram MiB"
for NC in 40 36 32 28 24; do
  pkill -9 -f "port $PORT" 2>/dev/null; sleep 1
  "$BIN" -m "$MODEL" -fa on --n-cpu-moe "$NC" --ctx-size "$CTX" \
    --cache-type-k q8_0 --cache-type-v q8_0 --host 127.0.0.1 --port "$PORT" \
    > /tmp/e5b_$NC.log 2>&1 &
  SRV=$!; ok=0
  for i in $(seq 1 200); do
    curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || break; sleep 1
  done
  [ "$ok" = 1 ] || { echo "  ncmoe=$NC DIED"; tail -8 /tmp/e5b_$NC.log; kill -9 $SRV 2>/dev/null; continue; }
  decode_tps >/dev/null
  read tps vram < <(decode_tps)
  printf "%-8s %-10s %-10s\n" "$NC" "$tps" "$vram"
  kill -9 $SRV 2>/dev/null; sleep 2
done
echo
echo "Overlay vs cache: cache s64=45.8 t/s @ 8823 MiB, s32=38.5 @ 6491. If a static ncmoe hits"
echo "the SAME VRAM with >= the t/s, the cache is redundant with placement for this load-balanced"
echo "model (top-8 only 28% hit). If the cache beats static at equal VRAM, it earns the fork."
