#!/usr/bin/env bash
# kv_recall_sweep.sh — Phase 3 gate: pick the SMALLEST KV-quant that holds long-context recall.
#
# WHY: our standing t/s gate `ops/kv-quant-bench.sh` already proved (on the Qwen3.6-35B-A3B MoE)
# that symmetric q4_0 KV is ~LOSSLESS on GDN hybrids and asymmetric K!=V falls off-GPU (-57%). This
# script re-confirms the LOSSLESS claim on the DENSE 27B and, unlike the t/s gate, measures QUALITY:
# multi-needle retrieval accuracy at long depth. Symmetric arms ONLY (asymmetric is off-GPU, skip it).
#
# METHOD: build a long context (~depth tokens) of labeled records, hide K unique needles at known
# positions, ask for each needle's value, score exact-match accuracy. Run per KV-quant arm on a fresh
# server. f16/f16 is the gold reference; q8_0 and q4_0 are the candidates.
#
# DECISION: if q4_0 accuracy == f16 accuracy at the target depth -> ship q4_0/q4_0 @256k (4.1GB, smallest).
# If q4_0 degrades on the dense variant (it did NOT on the MoE) -> step to q8_0/q8_0; if still short, 128k.
#
# Usage: MODEL=/home/augus/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf DEPTH=131072 NEEDLES=24 \
#        bash ops/qwen38-bringup/kv_recall_sweep.sh
set -u
LLAMA=${LLAMA:-/home/augus/src/llama.cpp-master}
BIN="$LLAMA/build/bin/llama-server"
MODEL=${MODEL:-/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf}
PORT=${PORT:-8080}
DEPTH=${DEPTH:-131072}        # target context depth in tokens (start 131072; repeat at 262144)
NEEDLES=${NEEDLES:-24}        # needles spread evenly across depth
export CUDA_VISIBLE_DEVICES=0

CSV=$(mktemp); echo "kv,needle,depth_frac,ok" > "$CSV"

# Generate the needle bank once (byte-stable so cache_prompt reuse is honest across needles).
NEEDLE_JSON=$(python3 - "$DEPTH" "$NEEDLES" <<'PY'
import json,sys
depth,n=int(sys.argv[1]),int(sys.argv[2])
# ~depth tokens of filler; ~4 chars/token heuristic -> depth*4 chars of records.
approx_records = max(n*4, depth // 12)   # each record ~ a dozen tokens
needles={}
recs=[]
step=approx_records//n
for i in range(approx_records):
    if i % step == 0 and len(needles) < n:
        code=f"ZK{len(needles):03d}Q"          # unique magic value
        needles[f"slot_{len(needles):03d}"]=code
        recs.append(f"Record {i:05d}: config_{i} secret={code} region=r{i%5}.")
    else:
        recs.append(f"Record {i:05d}: config_{i} value={i*7%1000} region=r{i%5}.")
print(json.dumps({"ctx":"You are a config lookup service. Answer only from the records.\n"+"\n".join(recs),
                  "needles":needles}))
PY
)

run_arm() {  # ctk ctv label
  local ctk="$1" ctv="$2" label="$3" slog; slog=$(mktemp)
  echo "########## KV=$label  (ctk=$ctk ctv=$ctv) ##########"
  "$BIN" -m "$MODEL" -c "$((DEPTH+2048))" -ngl 999 -fa 1 --no-mmproj \
    --cache-type-k "$ctk" --cache-type-v "$ctv" --jinja -np 1 \
    --host 127.0.0.1 --port "$PORT" </dev/null > "$slog" 2>&1 &
  local pid=$!
  for i in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
  if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
     echo "  !! server failed to start for $label (OOM at this depth?) — see $slog"; grep -iE "error|oom|out of memory" "$slog" | tail -5
     kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -f "$slog"; return
  fi
  NEEDLE_JSON="$NEEDLE_JSON" python3 - "$PORT" "$label" "$CSV" <<'PY'
import json,os,sys,urllib.request
port,label,csv=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.loads(os.environ["NEEDLE_JSON"]); ctx=d["ctx"]; needles=d["needles"]
keys=list(needles); N=len(keys)
for idx,k in enumerate(keys):
    q=(ctx+f"\n\nQUESTION: What is the exact secret value at {k}? "
           f"Reply with ONLY the value, no other text.")
    body={"messages":[{"role":"user","content":q}],"max_tokens":12,"temperature":0.0,"top_k":1,
          "stream":False,"cache_prompt":True,"chat_template_kwargs":{"enable_thinking":False}}
    try:
        r=json.load(urllib.request.urlopen(urllib.request.Request(
           f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(body).encode(),
           headers={"Content-Type":"application/json"}),timeout=600))
        out=r["choices"][0]["message"]["content"]
    except Exception as e:
        out=f"<err:{e}>"
    ok=1 if needles[k] in out else 0
    open(csv,"a").write(f"{label},{k},{idx/max(N-1,1):.3f},{ok}\n")
print(f"  {label}: scored {N} needles")
PY
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -f "$slog"; sleep 3
}

run_arm f16  f16  "f16 (GOLD)"
run_arm q8_0 q8_0 "q8_0"
run_arm q4_0 q4_0 "q4_0"

echo "===================== RECALL SUMMARY (accuracy per KV arm @ depth=$DEPTH) ====================="
python3 - "$CSV" <<'PY'
import csv,sys,statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
for kv in ["f16 (GOLD)","q8_0","q4_0"]:
    v=[int(r["ok"]) for r in rows if r["kv"]==kv]
    if not v: continue
    print(f"  {kv:12s}  acc={sum(v)/len(v)*100:5.1f}%  ({sum(v)}/{len(v)})")
print("\n# DECISION: smallest KV whose acc == f16 acc wins. On our 35B MoE that was q4_0 (lossless).")
print("# If q4_0 < f16 on the DENSE 27B, step to q8_0; if still short, reduce ctx to 128k.")
PY
rm -f "$CSV"
