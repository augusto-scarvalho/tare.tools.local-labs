#!/usr/bin/env bash
# GDN concurrent-serving A/B — one arm (OFF or ON), all k, ncmoe.
# Usage: gdn_conc_arm.sh <off|on|detect> <ncmoe>
# Launches server in-session (&), runs driver per k, kills server. All in one WSL lifetime.
set -u
ARM="${1:-off}"
NCMOE="${2:-0}"
BIN=/home/augus/src/llama.cpp-master/build/bin/llama-server
MODEL=/home/augus/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
DRIVER=/mnt/c/projects/local-model-lifecycle/gdn_conc_bench.py
OUT=/mnt/c/projects/local-model-lifecycle/runs/gdn
mkdir -p "$OUT"
REPS=7            # first 2 warmup -> 5 measured
PLEN=1024
KS="${KS:-1 2 4 8}"

# ctx: ncmoe=0 -> 32768 fits 24GB; ncmoe=8 -> 131072
if [ "$NCMOE" = "0" ]; then CTX=32768; else CTX=131072; fi

# env for the arm
unset GGML_CUDA_GDN_CHUNKED GGML_CUDA_GDN_CHUNKED_MIN_TOKENS
case "$ARM" in
  off)    : ;;                                                         # sequential
  on)     export GGML_CUDA_GDN_CHUNKED=1 GGML_CUDA_GDN_CHUNKED_MIN_TOKENS=128 ;;
  detect) export GGML_CUDA_GDN_CHUNKED=1 GGML_CUDA_GDN_CHUNKED_MIN_TOKENS=1000000 ;; # never engages
  *) echo "bad arm $ARM"; exit 2 ;;
esac
echo "ARM=$ARM NCMOE=$NCMOE CTX=$CTX CHUNKED=${GGML_CUDA_GDN_CHUNKED:-unset} MINTOK=${GGML_CUDA_GDN_CHUNKED_MIN_TOKENS:-unset}"

"$BIN" -m "$MODEL" --n-cpu-moe "$NCMOE" -c "$CTX" --parallel 8 -b 8192 -ub 2048 \
  -ctxcp 0 --host 127.0.0.1 --port 8080 > /tmp/gdn_srv_${ARM}.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

# wait for health (<=150s)
ready=0
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then ready=1; echo "server READY after $((i*5))s"; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "SERVER DIED during load"; tail -15 /tmp/gdn_srv_${ARM}.log; exit 3; fi
  sleep 5
done
[ "$ready" = 1 ] || { echo "server not ready"; tail -15 /tmp/gdn_srv_${ARM}.log; exit 3; }

for k in $KS; do
  f="$OUT/armB__${ARM}__ncmoe${NCMOE}__k${k}.json"
  echo "=== running k=$k -> $f ==="
  if python3 "$DRIVER" "$k" "$PLEN" "$REPS" > "$f" 2>/tmp/gdn_drv_err.log; then
    python3 -c "import json;d=json.load(open('$f'));print('  k=%d agg_tps_median=%.1f iqr=%s p50ms=%.1f'%(d['k'],d['agg_tps_median'],tuple('%.1f'%x for x in d['agg_tps_iqr']),d['prompt_ms_p50']))"
  else
    echo "  DRIVER FAILED k=$k"; tail -5 /tmp/gdn_drv_err.log
  fi
done
echo "ARM $ARM DONE"
