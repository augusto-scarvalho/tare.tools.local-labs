#!/usr/bin/env bash
# §B2b — does PINNING the KV host buffer recover the KV-in-RAM decode penalty §B2a measured?
#
# §B2a confirmed KV-in-RAM (--no-kv-offload) is a large, context-scaling transfer-bound regime
# (-70% at ~800 tok, -77% at ~8000). Source (llama-kv-cache.cpp:212) showed the KV lands in a
# PAGEABLE ggml_backend_cpu_buffer_type(), so each token's host->GPU KV copy is bounce-buffered.
# The env-gated patch (GGML_KV_PIN_HOST) swaps it for the device HOST buffer (cudaHostRegister'd)
# so the copy is a direct DMA. Both arms are ONE binary, --no-kv-offload, differing only by the
# env var -- the project's matched-control idiom.
#
#   base : --no-kv-offload                          (pageable KV host buffer)
#   pin  : --no-kv-offload  GGML_KV_PIN_HOST=1      (pinned CUDA_Host KV buffer)
#
# CONFIRMS the patch engaged (KV buffer shows CUDA_Host in the pin arm's log) before trusting any
# delta -- a null is only meaningful if the buffer actually changed.
#
#   MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/projects/local-model-lifecycle/probe_b2b_pin.sh
set -u

BIN=/home/augus/src/llama.cpp-master/build/bin/llama-server
MODEL=/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
NCMOE=6
CTX=16384
GEN=120
DEPTHS="800 4000 8000"
PORT=8093

[ -x "$BIN" ] || { echo "REFUSING: no binary at $BIN"; exit 2; }
[ -f "$MODEL" ] || { echo "REFUSING: no model at $MODEL"; exit 2; }

launch() {  # $1 = pin? (1/0)
  pkill -9 -f "port $PORT" 2>/dev/null; sleep 1
  local envp=""; [ "$1" = 1 ] && envp="GGML_KV_PIN_HOST=1"
  env $envp "$BIN" -m "$MODEL" -fa on --n-cpu-moe "$NCMOE" --no-kv-offload --ctx-size "$CTX" \
    --cache-type-k q8_0 --cache-type-v q8_0 --host 127.0.0.1 --port "$PORT" \
    > /tmp/b2b_server.log 2>&1 &
  SRV=$!
  for i in $(seq 1 200); do
    curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 'ok' && return 0
    kill -0 $SRV 2>/dev/null || { echo "  SERVER DIED; tail:"; tail -15 /tmp/b2b_server.log; return 1; }
    sleep 1
  done
  echo "  never healthy"; return 1
}

decode_tps() {  # $1 = depth
  python3 - "$PORT" "$1" "$GEN" <<'PY'
import json,sys,urllib.request
port,depth,gen=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
word="bandwidth bound decode waits on memory not compute "
prompt=word*(max(1,depth//8))
body=json.dumps({"prompt":prompt,"n_predict":gen,"temperature":0,"cache_prompt":False,
                 "ignore_eos":True}).encode()
req=urllib.request.Request(f"http://127.0.0.1:{port}/completion",data=body,
                           headers={"Content-Type":"application/json"})
try:
    d=json.load(urllib.request.urlopen(req,timeout=900))
except Exception as e:
    print(f"ERR 0 0"); sys.exit(0)
t=d.get("timings",{}); pn=t.get("predicted_n",0); tps=t.get("predicted_per_second",0)
print(f"{tps:.2f} {t.get('prompt_n',0)} {pn}" if pn>=gen*0.8 else f"INVALID {t.get('prompt_n',0)} {pn}")
PY
}

declare -A RES
for arm in base pin; do
  p=0; [ "$arm" = pin ] && p=1
  echo "== launching arm=$arm (pin=$p) =="
  launch "$p" || { echo "ABORT: arm $arm failed"; kill -9 ${SRV:-0} 2>/dev/null; exit 1; }
  echo "  -- KV buffer allocation (must show CUDA_Host for pin, CPU for base) --"
  grep -iE "KV.*buffer|kv_cache.*size|CUDA_Host|CPU_Mapped|KV self" /tmp/b2b_server.log | sed 's/^/     /' | head -6
  decode_tps 256 >/dev/null   # warm-up
  for D in $DEPTHS; do
    read tps pn gn rest < <(decode_tps "$D")
    RES["$arm,$D"]="$tps"
    printf "  depth~%-6s prompt_n=%-6s gen_n=%-4s decode=%-8s t/s\n" "$D" "$pn" "$gn" "$tps"
  done
  kill -9 $SRV 2>/dev/null; sleep 2
done

echo
echo "============================================================"
echo "§B2b  KV-host PIN vs pageable, in the --no-kv-offload regime (ncmoe=$NCMOE)"
echo "============================================================"
printf "%-8s %-12s %-12s %-10s\n" "depth" "base(pageable)" "pin(CUDA_Host)" "recover%"
for D in $DEPTHS; do
  b=${RES["base,$D"]}; n=${RES["pin,$D"]}
  rec=$(python3 -c "b=$b;n=$n;print(f'{(n/b-1)*100:+.1f}' if b else 'NA')" 2>/dev/null || echo NA)
  printf "%-8s %-12s %-12s %-10s\n" "$D" "$b" "$n" "$rec"
done
echo
echo "recover% = pin decode gain over pageable. Positive & growing with depth => pinning the KV"
echo "host buffer is a real lever in the KV-in-RAM regime. ~0 => llama.cpp's copy path already"
echo "avoids the pageable penalty (pinning has nothing to add); document as null."
