#!/usr/bin/env bash
# §E5 — does the MoE expert cache recover heavy-offload decode? At ncmoe=40 (all experts on
# CPU, ~31 t/s baseline, H2D-bound) keep the top-N hottest experts/layer resident in VRAM via
# --moe-cache-slots N + --moe-cache-profile. Measured skew ceiling: top8=28% hit, top32=59%,
# top64=79%. If decode climbs with N toward the resident (~98 t/s) level, the cache is a real
# lever; the VRAM it costs is the price. Reports decode t/s AND VRAM used per arm.
set -u
BIN=/home/augus/src/slop.cpp-stack/build/bin/llama-server
MODEL=/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
PROF=/home/augus/models/qwen36-35b-moe-trace.csv
NCMOE=40
CTX=4096
GEN=120
DEPTH=256
PORT=8096
[ -x "$BIN" ] || { echo "REFUSING: no stack server"; exit 2; }
[ -f "$PROF" ] || { echo "REFUSING: no profile"; exit 2; }

decode_tps() {  # echoes "tps vram_used"
  local tps
  tps=$(python3 - "$PORT" "$DEPTH" "$GEN" <<'PY'
import json,sys,urllib.request
port,depth,gen=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
word="bandwidth bound decode waits on memory not compute "
prompt=word*(max(1,depth//8))
body=json.dumps({"prompt":prompt,"n_predict":gen,"temperature":0,"cache_prompt":False,"ignore_eos":True}).encode()
req=urllib.request.Request(f"http://127.0.0.1:{port}/completion",data=body,headers={"Content-Type":"application/json"})
try:
    d=json.load(urllib.request.urlopen(req,timeout=900))
except Exception as e:
    print("ERR"); sys.exit(0)
t=d.get("timings",{}); pn=t.get("predicted_n",0)
print(f"{t.get('predicted_per_second',0):.2f}" if pn>=gen*0.8 else "INVALID")
PY
)
  local vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "$tps $vram"
}

declare -A TPS VRAM
for arm in off s8 s32 s64; do
  extra=""
  case $arm in
    s8)  extra="--moe-cache-slots 8 --moe-cache-profile $PROF" ;;
    s32) extra="--moe-cache-slots 32 --moe-cache-profile $PROF" ;;
    s64) extra="--moe-cache-slots 64 --moe-cache-profile $PROF" ;;
  esac
  echo "== arm=$arm (ncmoe=$NCMOE $extra) =="
  pkill -9 -f "port $PORT" 2>/dev/null; sleep 1
  "$BIN" -m "$MODEL" -fa on --n-cpu-moe "$NCMOE" $extra --ctx-size "$CTX" \
    --cache-type-k q8_0 --cache-type-v q8_0 --host 127.0.0.1 --port "$PORT" \
    > /tmp/e5_$arm.log 2>&1 &
  SRV=$!
  ok=0
  for i in $(seq 1 200); do
    curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || break; sleep 1
  done
  [ "$ok" = 1 ] || { echo "  DIED/timeout:"; tail -12 /tmp/e5_$arm.log; kill -9 $SRV 2>/dev/null; continue; }
  decode_tps >/dev/null                       # warm-up
  read tps vram < <(decode_tps)
  TPS[$arm]=$tps; VRAM[$arm]=$vram
  printf "  decode=%-8s t/s   vram_used=%-6s MiB\n" "$tps" "$vram"
  kill -9 $SRV 2>/dev/null; sleep 2
done

echo
echo "============================================================"
echo "§E5  MoE expert cache at ncmoe=$NCMOE (baseline off ~31 t/s; resident ncmoe=6 ~98 t/s)"
echo "============================================================"
printf "%-6s %-10s %-12s %-10s\n" "arm" "decode t/s" "vram MiB" "gain vs off"
for arm in off s8 s32 s64; do
  t=${TPS[$arm]:-NA}; v=${VRAM[$arm]:-NA}
  g=$(python3 -c "o=${TPS[off]:-0};n=$t;print(f'{(n/o-1)*100:+.1f}%' if o and n else 'NA')" 2>/dev/null || echo NA)
  printf "%-6s %-10s %-12s %-10s\n" "$arm" "$t" "$v" "$g"
done
echo
echo "Rising decode with slots => cache recovers heavy-offload decode (real lever, at VRAM cost)."
echo "Compare the VRAM used to what static ncmoe=6 costs (~19 GB) for ~98 t/s: is dynamic caching"
echo "more VRAM-efficient than static placement? That is the fork question."
