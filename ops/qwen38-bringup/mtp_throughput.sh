#!/usr/bin/env bash
# mtp_throughput.sh — Phase 4 gate: measure the MTP spec-decode decode-speedup on the DENSE Qwen3.8-27B.
#
# WHY: MTP (`--spec-type draft-mtp`, built-in nextn head confirmed present in Phase 1) is our decode
# lever. The standing gate `ops/spec-drafter-bench.sh` is MoE-flavored (hardcodes --n-cpu-moe); this is
# the DENSE analogue (no ncmoe). Arms: no-spec floor vs draft-mtp (n-max 2 and 3) on coding prompts.
# Metric = predicted_per_second (decode t/s), temp 0, enable_thinking:false, fresh server per arm.
#
# RESULT 2026-08-16 (UD-Q4_K_XL, 3090, build 068764d92, 5 reps, temp 0, enable_thinking:false):
#     regime   no-spec   draft-mtp-n2       draft-mtp-n3
#     GEN       39.5      76.6 (+94%)        83.6 (+112%)
#     EDIT      39.4      83.3 (+111%)       88.0 (+123%)
#   => draft-mtp ~2.1-2.2x on code; n-max 3 > n-max 2 clearly. WINNER: --spec-type draft-mtp --spec-draft-n-max 3.
#   (Far exceeds the +33% public reference; nextn head is high-acceptance on deterministic code.)
#
# Usage: MODEL=/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf REPS=5 bash ops/qwen38-bringup/mtp_throughput.sh
set -u
LLAMA=${LLAMA:-/home/augus/src/llama.cpp-master}
BIN="$LLAMA/build/bin/llama-server"
MODEL=${MODEL:-/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf}
PORT=${PORT:-8080}; REPS=${REPS:-5}; NTOK=${NTOK:-400}
export CUDA_VISIBLE_DEVICES=0
CSV=$(mktemp); echo "arm,regime,rep,tps" > "$CSV"
GEN=$(mktemp); EDIT=$(mktemp)
cat > "$GEN" <<'EOF'
Write a Python class TaskScheduler with methods: add_task(name, priority, deps), remove_task(name), run_order() returning a topological order honoring priority ties by name, and detect_cycle() raising ValueError on cyclic deps. Include type hints and docstrings. Output only the code.
EOF
cat > "$EDIT" <<'EOF'
Reproduce the module below VERBATIM inside one ```python block, changing only: rename every `svc` to `service` and add a return type hint `-> dict` to load(). Output the full file, nothing else.

```python
import json, os
def load(svc):
    svc = dict(svc)
    svc.setdefault("host", "127.0.0.1"); svc.setdefault("port", 8080)
    if svc["port"] < 1 or svc["port"] > 65535: raise ValueError("bad port")
    return svc
```
EOF

req() { # arm regime rep promptfile
  python3 - "$1" "$2" "$3" "$4" "$NTOK" "$CSV" "$PORT" <<'PY'
import json,sys,urllib.request,urllib.error
arm,regime,rep,pf,ntok,csv,port=sys.argv[1:8]
p={"messages":[{"role":"user","content":open(pf).read()}],"max_tokens":int(ntok),"temperature":0.0,
   "top_k":1,"stream":False,"chat_template_kwargs":{"enable_thinking":False},"cache_prompt":False}
try:
    r=json.load(urllib.request.urlopen(urllib.request.Request(
      f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(p).encode(),
      headers={"Content-Type":"application/json"}),timeout=600))
    tps=round(r.get("timings",{}).get("predicted_per_second",0),2)
except urllib.error.HTTPError as e:
    print("HTTP",e.code,e.read()[:200].decode(errors='replace')); tps=0
open(csv,"a").write(f"{arm},{regime},{rep},{tps}\n")
PY
}

runarm() { # label specflag...
  local label="$1"; shift; local slog; slog=$(mktemp)
  echo "### arm=$label  spec='$*'"
  sleep 15  # cooldown between isolated arms (heat-soak guard)
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader 2>/dev/null | head -1
  "$BIN" -m "$MODEL" -c 8192 -ngl 999 -fa 1 --no-mmproj \
    --cache-type-k q4_0 --cache-type-v q4_0 "$@" \
    --host 127.0.0.1 --port "$PORT" -np 1 </dev/null > "$slog" 2>&1 &
  local pid=$!
  for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
  if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "  !! server failed for $label"; grep -iE "error|unknown|invalid" "$slog" | tail -5
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -f "$slog"; return
  fi
  for regime in GEN EDIT; do
    local pf=$GEN; [ "$regime" = EDIT ] && pf=$EDIT
    req "$label" "$regime" 0 "$pf" >/dev/null      # warmup
    for rep in $(seq 1 "$REPS"); do req "$label" "$regime" "$rep" "$pf"; done
  done
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -f "$slog"; sleep 3
}

runarm "no-spec"
runarm "draft-mtp-n2" --spec-type draft-mtp --spec-draft-n-max 2
runarm "draft-mtp-n3" --spec-type draft-mtp --spec-draft-n-max 3

echo "===================== SUMMARY (mean decode t/s, 95% CI, vs no-spec floor) ====================="
python3 - "$CSV" "$REPS" <<'PY'
import csv,sys,statistics as st,math
rows=list(csv.DictReader(open(sys.argv[1]))); reps=int(sys.argv[2])
tcrit={2:12.71,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365}.get(reps,2.776)
arms=["no-spec","draft-mtp-n2","draft-mtp-n3"]; base={}
for reg in ["GEN","EDIT"]:
    print(f"\n== {reg} ==")
    for arm in arms:
        v=[float(r["tps"]) for r in rows if r["arm"]==arm and r["regime"]==reg and float(r["tps"])>0]
        if not v: print(f"  {arm:14s} (no data)"); continue
        m=st.mean(v); sd=st.stdev(v) if len(v)>1 else 0.0
        ci=tcrit*sd/math.sqrt(len(v)) if len(v)>1 else 0.0
        if arm=="no-spec": base[reg]=m
        rel=f"{(m/base[reg]-1)*100:+.0f}%" if base.get(reg) else "n/a"
        print(f"  {arm:14s} {m:7.1f} t/s  std {sd:4.2f}  95%CI[{m-ci:6.1f},{m+ci:6.1f}]  vs floor {rel}")
PY
rm -f "$CSV" "$GEN" "$EDIT"
echo "=== DONE ==="
