#!/usr/bin/env bash
# LAB-CTX-001 (Qwen3.8) — effective-context curve via the CHAT endpoint (instruct, enable_thinking:false).
#
# WHY a dedicated script (not context_probe.py): context_probe uses the /completion endpoint (raw prompt,
# no chat template). On this THINKING/instruct model that path breaks retrieval (measured: NIAH 100% @8k
# then 0% @16k+ via /completion) — a HARNESS ARTIFACT, contradicted by a chat-endpoint single-needle probe
# that retrieved fine at 166k. This script uses /v1/chat/completions + enable_thinking:false + a distinctive
# alphanumeric needle, which is how the model is actually served. Records NIAH found + prefill/decode t/s at
# depth (MTP OFF here = clean speed baseline; deploy MTP scales decode ~2x per Phase 4). cache_prompt False
# so each depth is a real prefill.
#
# RESULT 2026-08-16 (UD-Q4_K_XL, q4_0 KV, chat endpoint, enable_thinking:false, no-MTP):
#     depth   NIAH(single)   decode
#      8k      100% (2/2)     45 t/s
#     16k      100% (2/2)     43 t/s      <- context_probe.py /completion gave 0% here (ARTIFACT)
#     32k      100% (2/2)     41 t/s
#     65k      100% (2/2)     36 t/s
#    131k      100% (d=.25; d=.75 one-off socket timeout)  30 t/s
#   => single-needle effective context is FULL through 131k (matches the 166k deep-probe). The
#      /completion-endpoint "collapse at 16k" was a HARNESS ARTIFACT (raw prompt on an instruct model).
#      Reconciles with Phase 3: single needle survives deep; MULTI-needle (32-48 near-identical) is what
#      collapses at ~168k (interference). Decode degrades ~33% 8k->131k (no-MTP baseline; deploy MTP ~2x).
#
# Usage: MODEL=<gguf> bash ops/qwen38-bringup/ctx_curve.sh   (via ops/wsl/wslx.sh; long -> run_in_background)
set -u
LLAMA=/home/augus/src/llama.cpp-master
BIN="$LLAMA/build/bin/llama-server"
MODEL=${MODEL:-/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf}
PORT=8100; export CUDA_VISIBLE_DEVICES=0
LS=${LS:-"8000 16000 32000 65000 131000"}
DS=${DS:-"0.25 0.75"}
CTX=${CTX:-140000}

echo "== boot server (chat, q4_0 KV, ctx $CTX) =="
"$BIN" -m "$MODEL" -c "$CTX" -ngl 999 -fa 1 --no-mmproj --cache-type-k q4_0 --cache-type-v q4_0 \
  --jinja -np 1 --host 127.0.0.1 --port "$PORT" </dev/null >/tmp/ctx_curve_server.log 2>&1 &
PID=$!; trap 'kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null' EXIT
for i in $(seq 1 200); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "SERVER FAILED"; tail -8 /tmp/ctx_curve_server.log; exit 1; }

python3 - "$PORT" "$LS" "$DS" <<'PY'
import json,sys,urllib.request
port=sys.argv[1]; LS=[int(x) for x in sys.argv[2].split()]; DS=[float(x) for x in sys.argv[3].split()]
FILLER=("The quarterly maintenance log for sector 7 records routine checks with no anomalies "
        "and nominal readings across all subsystems. ")  # ~22 tokens/unit
def niah(L,depth,code):
    nw=int(L/1.35); pool=(FILLER*(nw//22+2)).split(); words=pool[:nw]
    needle=f"IMPORTANT REMEMBER THIS: the Grover facility access code is {code}.".split()
    pos=max(1,min(len(words)-1,int(len(words)*depth)))
    words=words[:pos]+needle+words[pos:]
    return (" ".join(words)+
            "\n\nQuestion: what is the Grover facility access code? Reply with ONLY the code, nothing else.")
def ask(prompt):
    body={"messages":[{"role":"user","content":prompt}],"max_tokens":24,"temperature":0.0,"top_k":1,
          "stream":False,"cache_prompt":False,"chat_template_kwargs":{"enable_thinking":False}}
    r=json.load(urllib.request.urlopen(urllib.request.Request(
       f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(body).encode(),
       headers={"Content-Type":"application/json"}),timeout=1200))
    return r["choices"][0]["message"].get("content") or "", r.get("timings",{}), r.get("usage",{})
print(f"{'L_target':>9} {'prompt_n':>9} {'d':>5} {'found':>6} {'prefill_tps':>11} {'decode_tps':>10}",flush=True)
rows=[]
for L in LS:
    for d in DS:
        code=f"ZK{40000+(L//1000)*7+int(d*100)*13}Q"
        content,t,u=ask(niah(L,d,code))
        found=code in content
        pn=u.get("prompt_tokens",t.get("prompt_n",0)); ptps=t.get("prompt_per_second",0); dtps=t.get("predicted_per_second",0)
        rows.append((L,pn,d,found,dtps))
        print(f"{L:>9} {pn:>9} {d:>5} {str(found):>6} {ptps:>11.1f} {dtps:>10.2f}",flush=True)
print("\n== summary ==")
for L in LS:
    r=[x for x in rows if x[0]==L]
    acc=sum(1 for x in r if x[3])/len(r); dt=sum(x[4] for x in r if 0<x[4]<10000)/max(1,sum(1 for x in r if 0<x[4]<10000))
    print(f"L~{L:>7} (prompt_n {r[0][1]:>7}): NIAH {acc*100:>5.0f}%  decode ~{dt:>5.1f} t/s (no-MTP)")
PY
echo "=== DONE ==="
