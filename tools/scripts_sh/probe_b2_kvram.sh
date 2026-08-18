#!/usr/bin/env bash
# §B2a precondition — is KV-in-RAM (--no-kv-offload) a transfer-bound decode regime on this
# box, and does the penalty GROW with context length?
#
# The §B2 card's pin patch (cudaHostRegister on the KV host buffer) only earns a build if
# putting the KV cache in system RAM produces a per-token PCIe transfer whose cost rises with
# context -- the dose-response a pin would ride on. §B5 taught the discipline: measure the
# precondition before building. Stock, no patch, no build.
#
# qwen36-35B at ncmoe=6 (deploy placement) so expert-streaming is IDENTICAL across both arms
# and cancels in the delta -- what is left is purely KV placement:
#   base : default (KV offloaded to GPU VRAM)
#   nokv : --no-kv-offload (KV in system RAM, read over PCIe each token)
# At each depth D we prime a ~D-token prompt then decode a short burst and read the server's
# own predicted_per_second. nokv/base decode ratio vs D is the dose-response.
#
#   MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/projects/local-model-lifecycle/probe_b2_kvram.sh
set -u

BIN=/home/augus/src/llama.cpp-master/build/bin/llama-server
MODEL=/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
NCMOE=6
CTX=16384
GEN=120
DEPTHS="800 4000 8000 12000"
PORT=8092

[ -x "$BIN" ] || { echo "REFUSING: no binary at $BIN"; exit 2; }
[ -f "$MODEL" ] || { echo "REFUSING: no model at $MODEL"; exit 2; }
command -v python3 >/dev/null || { echo "REFUSING: need python3 in WSL"; exit 2; }

launch() {  # $1 = extra flag(s)
  pkill -9 -f "port $PORT" 2>/dev/null; sleep 1
  "$BIN" -m "$MODEL" -fa on --n-cpu-moe "$NCMOE" $1 --ctx-size "$CTX" \
    --cache-type-k q8_0 --cache-type-v q8_0 --host 127.0.0.1 --port "$PORT" \
    > /tmp/b2_server.log 2>&1 &
  SRV=$!
  for i in $(seq 1 200); do
    curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 'ok' && return 0
    kill -0 $SRV 2>/dev/null || { echo "  SERVER DIED; tail:"; tail -15 /tmp/b2_server.log; return 1; }
    sleep 1
  done
  echo "  server never healthy"; return 1
}

# One python call: build a ~D-word prompt, POST /completion, print "tps prompt_n predicted_n".
decode_tps() {  # $1 = depth
  python3 - "$PORT" "$1" "$GEN" <<'PY'
import json,sys,urllib.request
port,depth,gen=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
word="bandwidth bound decode waits on memory not compute "
prompt=(word*((depth//5)+1))
# ignore_eos: force exactly `gen` decode tokens so predicted_per_second is a real decode
# rate at KV depth ~= prompt_n, not a 1-token EOS artifact (the 1e6 sentinel bug).
body=json.dumps({"prompt":prompt,"n_predict":gen,"temperature":0,"cache_prompt":False,
                 "ignore_eos":True}).encode()
req=urllib.request.Request(f"http://127.0.0.1:{port}/completion",data=body,
                           headers={"Content-Type":"application/json"})
try:
    d=json.load(urllib.request.urlopen(req,timeout=900))
except Exception as e:
    print(f"ERR 0 0 ({e})"); sys.exit(0)
t=d.get("timings",{})
# invalid unless the burst actually decoded ~gen tokens
pn=t.get("predicted_n",0)
tps=t.get("predicted_per_second",0)
print(f"{tps:.2f} {t.get('prompt_n',0)} {pn}" if pn>=gen*0.8 else f"INVALID {t.get('prompt_n',0)} {pn}")
PY
}

declare -A RES
for arm in base nokv; do
  flag=""; [ "$arm" = nokv ] && flag="--no-kv-offload"
  echo "== launching arm=$arm ($flag) ncmoe=$NCMOE ctx=$CTX =="
  launch "$flag" || { echo "ABORT: arm $arm failed to load"; kill -9 ${SRV:-0} 2>/dev/null; exit 1; }
  decode_tps 256 >/dev/null    # warm-up, discarded
  for D in $DEPTHS; do
    read tps pn gn rest < <(decode_tps "$D")
    RES["$arm,$D"]="$tps"
    printf "  depth~%-6s  prompt_n=%-6s gen_n=%-4s  decode=%-8s t/s\n" "$D" "$pn" "$gn" "$tps"
  done
  kill -9 $SRV 2>/dev/null; sleep 2
done

echo
echo "============================================================"
echo "§B2a  KV-in-RAM decode penalty vs context depth (ncmoe=$NCMOE)"
echo "============================================================"
printf "%-8s %-13s %-13s %-10s\n" "depth" "base(GPU-KV)" "nokv(RAM-KV)" "penalty%"
for D in $DEPTHS; do
  b=${RES["base,$D"]}; n=${RES["nokv,$D"]}
  pen=$(python3 -c "b=$b;n=$n;print(f'{(1-n/b)*100:+.1f}' if b else 'NA')" 2>/dev/null || echo NA)
  printf "%-8s %-13s %-13s %-10s\n" "$D" "$b" "$n" "$pen"
done
echo
echo "Read DOWN penalty%: growing with depth => KV-in-RAM is transfer-bound and scales with"
echo "context => the §B2 pin patch is worth building. Flat/small => null, no build."
