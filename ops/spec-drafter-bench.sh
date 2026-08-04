#!/usr/bin/env bash
# spec-drafter-bench.sh — the standing drafter-regression gate for spec-decode (IDEAS_BACKLOG S3).
#
# WHY: the fork ships a multi-drafter priority chain (`--spec-type a,b,...`; cheap-first order in
# common/speculative.cpp:2357 — matches upstream docs/speculative.md "draftless decoding has higher precedence").
# Stacking a low-quality/long drafter on top of MTP REDUCES throughput: the higher-priority ngram draft preempts
# MTP's better draft and the target still pays to verify the (often wrong) span. Rigorously measured 2026-08-04
# (deploy model, temp 0, enable_thinking:false, 6 reps/cell, 95% CI, GPU clock stable) vs a NO-SPEC FLOOR ~87 t/s:
#     regime      no-spec   draft-mtp        ngram-simple     draft-mtp,ngram-simple
#     GEN (code)   87.1     150.8 (+73%)     82.4 (-5%)       131.7   (< mtp alone)
#     EDIT         88.1     149.8 (+70%)     49.8 (-44%)      65.3    (below floor!)
#     PURE-COPY    86.6     132.2 (+53%)     107.3 (+24%)     122.9   (< mtp alone)
# => draft-mtp ALONE wins every regime (~1.7x). ngram only pays on ~verbatim copy (its §35/PLD niche) and even
#    there MTP is better; stacking ALWAYS loses to mtp-alone. Keep `--spec-type draft-mtp` alone.
# NOTE (exactness): ngram-simple is greedy-EXACT; draft-mtp deterministically DIVERGES from greedy (quality-neutral
#    on HumanEval+, not bit-exact) — so outputs are NOT identical across drafters; t/s is the decision metric.
# Re-run this if the drafter config or the MTP head changes. A new drafter ships only if it beats draft-mtp here.
#
# Usage: spec-drafter-bench.sh                 (4 configs x 3 regimes, 6 reps, prints mean/std/95%CI vs floor)
#        REPS=3 NTOK=256 spec-drafter-bench.sh
set -u
BIN=/home/augus/src/llama.cpp-master/build/bin/llama-server
MODEL="${MODEL:-/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
PORT="${PORT:-8080}"; REPS="${REPS:-6}"; NTOK="${NTOK:-400}"
CSV="$(mktemp)"; echo "config,regime,rep,tps" > "$CSV"
GEN="$(mktemp)"; EDIT="$(mktemp)"; COPY="$(mktemp)"

cat > "$GEN" <<'EOF'
Write a Python class UserProfileValidated with @property getters and setters for these 20 fields, each setter validating the type and raising TypeError on mismatch: user_id:int, username:str, email:str, full_name:str, age:int, is_active:bool, is_admin:bool, created_at:str, updated_at:str, last_login:str, login_count:int, bio:str, avatar_url:str, phone_number:str, country_code:str, timezone:str, locale:str, email_verified:bool, phone_verified:bool, two_factor_enabled:bool. Output only the code.
EOF
cat > "$EDIT" <<'EOF'
Reproduce the module below VERBATIM inside one ```python block, changing only: rename every `cfg` to `config`. Output the full file, nothing else.

```python
import os, sys, json, logging
logger = logging.getLogger(__name__)
def load_settings(cfg):
    cfg = dict(cfg)
    cfg.setdefault("host", "127.0.0.1"); cfg.setdefault("port", 8080)
    cfg.setdefault("timeout", 30); cfg.setdefault("retries", 3); cfg.setdefault("verbose", False)
    if cfg["verbose"]: logger.setLevel(logging.DEBUG)
    if cfg["port"] < 1 or cfg["port"] > 65535: raise ValueError("port out of range")
    if cfg["timeout"] <= 0: raise ValueError("timeout must be positive")
    if cfg["retries"] < 0: raise ValueError("retries must be non-negative")
    return cfg
```
EOF
cat > "$COPY" <<'EOF'
Repeat the following text EXACTLY VERBATIM, no changes, no commentary, no code fence:

A journey of a thousand miles begins with a single step taken with courage and determination toward a distant and uncertain horizon. The rain in Spain falls mainly on the plain during the cold and windy months of early spring when the flowers begin to bloom again. All work and no play makes Jack a very dull boy who never learns to enjoy the simple pleasures of an ordinary quiet afternoon at home near the river.
EOF

req() {  # cfg regime rep promptfile
  python3 - "$1" "$2" "$3" "$4" "$NTOK" "$CSV" <<'PY'
import json,sys,urllib.request
cfg,regime,rep,pf,ntok,csv=sys.argv[1:7]
p={"messages":[{"role":"user","content":open(pf).read()}],"max_tokens":int(ntok),"temperature":0.0,
   "top_k":1,"stream":False,"chat_template_kwargs":{"enable_thinking":False},"cache_prompt":False}
r=json.load(urllib.request.urlopen(urllib.request.Request(
  "http://127.0.0.1:8080/v1/chat/completions",data=json.dumps(p).encode(),
  headers={"Content-Type":"application/json"}),timeout=600))
tps=round(r.get("timings",{}).get("predicted_per_second",0),2)
open(csv,"a").write(f"{cfg},{regime},{rep},{tps}\n")
PY
}

runcfg() {  # label spec
  local label="$1" spec="$2" slog; slog=$(mktemp)
  echo "### config=$label spec='${spec:-<none>}'"
  "$BIN" -m "$MODEL" -fa on --n-cpu-moe 8 --ctx-size 8192 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    ${spec:+--spec-type "$spec"} --spec-draft-n-max 4 \
    --batch-size 2048 --ubatch-size 2048 \
    --host 127.0.0.1 --port "$PORT" -np 1 </dev/null > "$slog" 2>&1 &
  local pid=$!
  for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
  for regime in GEN EDIT COPY; do
    local pf=$GEN; [ "$regime" = EDIT ] && pf=$EDIT; [ "$regime" = COPY ] && pf=$COPY
    req "$label" "$regime" 0 "$pf" >/dev/null    # warmup
    for rep in $(seq 1 "$REPS"); do req "$label" "$regime" "$rep" "$pf"; done
  done
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -f "$slog"; sleep 3
}

runcfg "none"      ""
runcfg "draft-mtp" "draft-mtp"
runcfg "ngram"     "ngram-simple"
runcfg "mtp+ngram" "draft-mtp,ngram-simple"

echo "===================== SUMMARY (mean t/s, 95% CI, vs no-spec floor) ====================="
python3 - "$CSV" "$REPS" <<'PY'
import csv,sys,statistics as st,math
rows=list(csv.DictReader(open(sys.argv[1]))); reps=int(sys.argv[2])
tcrit={2:12.71,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365}.get(reps,2.776)
cfgs=["none","draft-mtp","ngram","mtp+ngram"]; base={}
for reg in ["GEN","EDIT","COPY"]:
    print(f"\n== {reg} ==")
    for cfg in cfgs:
        v=[float(r["tps"]) for r in rows if r["config"]==cfg and r["regime"]==reg]
        if not v: continue
        m=st.mean(v); sd=st.stdev(v) if len(v)>1 else 0.0
        ci=tcrit*sd/math.sqrt(len(v)) if len(v)>1 else 0.0
        if cfg=="none": base[reg]=m
        rel=f"{(m/base[reg]-1)*100:+.1f}%" if base.get(reg) else "n/a"
        print(f"  {cfg:12s} {m:7.1f} t/s  std {sd:4.2f}  95%CI[{m-ci:6.1f},{m+ci:6.1f}]  vs floor {rel}")
print("\n# winner = highest mean t/s; a drafter is net-positive only if its CI clears the no-spec floor.")
PY
rm -f "$CSV" "$GEN" "$EDIT" "$COPY"
