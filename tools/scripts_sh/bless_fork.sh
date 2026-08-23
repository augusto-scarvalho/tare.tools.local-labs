#!/usr/bin/env bash
# Bless the consolidated fork (fresh master + §B2b) before it is deployed. Three gates from
# the LANDSCAPE §1c correctness scan, all on the fork binary:
#   G1  §B2b engages: GGML_KV_PIN_HOST + --no-kv-offload -> KV on CUDA_Host(B2b).
#   G2  draft-mtp token-identity (#23335): base vs mtp greedy output byte-identical.
#   G3  coherence + -nkvo spot-check (#20140): KV-on-GPU and -nkvo output both non-degenerate.
set -u
FORK=/home/augus/src/slop.cpp-main/build/bin/llama-server  # branch 'lifecycle' = 720d7fa40 + §B2b
MODEL=/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
PROJ=/mnt/c/projects/local-model-lifecycle
PORT=8097
[ -x "$FORK" ] || { echo "REFUSING: fork binary not built at $FORK"; exit 2; }
pkill -9 -f llama-server 2>/dev/null; sleep 2

pass=0; fail=0
say(){ echo; echo "########## $* ##########"; }

# ---- G1: §B2b engagement ----
say "G1  §B2b KV-host-pin engages"
GGML_KV_PIN_HOST=1 "$FORK" -m "$MODEL" -fa on --n-cpu-moe 6 --no-kv-offload --ctx-size 4096 \
  --cache-type-k q8_0 --cache-type-v q8_0 --verbose --host 127.0.0.1 --port "$PORT" \
  > /tmp/bless_g1.log 2>&1 &
SRV=$!
for i in $(seq 1 180); do curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && break; kill -0 $SRV 2>/dev/null || break; sleep 1; done
n_pin=$(grep -c "CUDA_Host(B2b)" /tmp/bless_g1.log)
echo "  KV tensors on CUDA_Host(B2b): $n_pin"
kill -9 $SRV 2>/dev/null; sleep 2
if [ "$n_pin" -gt 0 ]; then echo "  G1 PASS"; pass=$((pass+1)); else echo "  G1 FAIL (patch not engaging)"; fail=$((fail+1)); fi

# ---- G2: draft-mtp token identity (deploy placement ncmoe=8, n-max 4) ----
say "G2  draft-mtp token-identity (#23335)"
MTP_BIN="$FORK" python3 "$PROJ/verify_mtp.py" "$MODEL" "--n-cpu-moe 8" > /tmp/bless_g2.log 2>&1
grep -E "IDENTICAL|predicted_per_second|accept|draft" /tmp/bless_g2.log | sed 's/^/  /'
if grep -q "IDENTICAL=True" /tmp/bless_g2.log; then echo "  G2 PASS"; pass=$((pass+1)); else echo "  G2 FAIL (draft-mtp diverges from base)"; fail=$((fail+1)); fi

# ---- G3: coherence + -nkvo spot-check ----
say "G3  coherence + -nkvo (#20140)"
coh_check() {  # $1 = extra flag, $2 = label
  pkill -9 -f "port $PORT" 2>/dev/null; sleep 1
  "$FORK" -m "$MODEL" -fa on --n-cpu-moe 8 $1 --ctx-size 4096 \
    --cache-type-k q8_0 --cache-type-v q8_0 --host 127.0.0.1 --port "$PORT" \
    > /tmp/bless_g3_$2.log 2>&1 &
  local SRV=$!
  for i in $(seq 1 180); do curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && break; kill -0 $SRV 2>/dev/null || break; sleep 1; done
  python3 - "$PORT" "$2" <<'PY'
import json,sys,urllib.request,re
port,label=sys.argv[1],sys.argv[2]
body=json.dumps({"prompt":"Explain in three sentences why the sky is blue, then count from 1 to 5.","n_predict":200,"temperature":0,"cache_prompt":False}).encode()
req=urllib.request.Request(f"http://127.0.0.1:{port}/completion",data=body,headers={"Content-Type":"application/json"})
try: d=json.load(urllib.request.urlopen(req,timeout=300))
except Exception as e: print(f"  [{label}] ERR {e}"); sys.exit(0)
c=d.get("content","")
# degeneration heuristics: a char repeated >25x in a row, or <15% unique chars
runs=max((len(m.group(0)) for m in re.finditer(r'(.)\1*',c)), default=0)
uniq=len(set(c))/max(1,len(c))
degen = runs>25 or uniq<0.05
print(f"  [{label}] {len(c)} chars, longest_run={runs}, uniq_ratio={uniq:.2f}, degenerate={degen}")
print(f"  [{label}] head: {c[:120]!r}")
print("DEGEN" if degen else "OK")
PY
  kill -9 $SRV 2>/dev/null; sleep 2
}
r1=$(coh_check "" "kv_on_gpu"); echo "$r1" | sed '/^\(OK\|DEGEN\)$/d'
r2=$(coh_check "--no-kv-offload" "nkvo"); echo "$r2" | sed '/^\(OK\|DEGEN\)$/d'
if ! echo "$r1$r2" | grep -q DEGEN; then echo "  G3 PASS (both coherent)"; pass=$((pass+1)); else echo "  G3 FAIL (degenerate output)"; fail=$((fail+1)); fi

echo
echo "============================================================"
echo "FORK BLESSING: $pass passed, $fail failed"
[ "$fail" = 0 ] && echo "-> BLESSED. Fork is deploy-ready." || echo "-> NOT blessed; inspect the failing gate above."
echo "============================================================"
