#!/usr/bin/env bash
# Hard thinking-budget EMULATION for Qwen3.8 (our llama.cpp v10159 has no native thinking_budget).
# Standard 2-pass: render the chat prompt with reasoning_effort=xhigh + <think> open, generate up to B
# thinking tokens, inject </think>, then generate the answer. Answers the open question: does xhigh-style
# reasoning, HARD-capped at B tokens, recover quality vs the truncation-deflated xhigh=45%? Scored by real
# evalplus over the EXACT market-r0 60. MTP on (speed; this is a quality-at-budget test, not token-count).
#
# Usage: BUDGETS="512 1024 2048" SUBSET_N=60 bash ops/qwen38-bringup/budget_emul.sh   (via wslx; run_in_background)
set -u
LLAMA=/home/augus/src/slop.cpp-main
BIN="$LLAMA/build/bin/llama-server"
MODEL=/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf
PY=/home/augus/sglang-venv/bin/python3
EVALPY=/home/augus/evalplus-venv/bin/python
REPO=/mnt/c/projects/local-model-lifecycle
PROBLEMS="$REPO/workloads/humaneval_plus.jsonl"
SUBSET=/tmp/subset60.txt
OUT=/home/augus/models/qwen38-27b/budget; mkdir -p "$OUT"
BUDGETS=${BUDGETS:-"512 1024 2048"}; SUBSET_N=${SUBSET_N:-60}
# ctx MUST exceed prompt + max(BUDGETS) + answer(768). At budget>=8192 the default 8192 overflows
# (server 400: "request exceeds the available context size") -> pass CTX=16384 for the 8192 arm.
CTX=${CTX:-8192}
PORT=8080; export CUDA_VISIBLE_DEVICES=0

echo "== boot Qwen3.8 (draft-mtp, ctx $CTX) =="
"$BIN" -m "$MODEL" -c "$CTX" -ngl 999 -fa 1 --no-mmproj --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 -np 1 --jinja --host 127.0.0.1 --port "$PORT" \
  </dev/null >/tmp/budget_server.log 2>&1 &
PID=$!; trap 'kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null' EXIT
for i in $(seq 1 200); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "SERVER FAILED"; tail -8 /tmp/budget_server.log; exit 1; }

echo "== generate (budgets: $BUDGETS ; n=$SUBSET_N) =="
"$PY" - "$MODEL" "$PROBLEMS" "$SUBSET" "$OUT" "$PORT" "$BUDGETS" "$SUBSET_N" "$LLAMA" <<'PY'
import json,sys,urllib.request
model_f,problems_f,subset_f,out,port,budgets_s,subn,llama=sys.argv[1:10]
budgets=[int(b) for b in budgets_s.split()]; subn=int(subn)
from gguf import GGUFReader
r=GGUFReader(model_f); tmpl=None
for k in r.fields:
    if k.lower()=="tokenizer.chat_template":
        fld=r.fields[k]; tmpl=bytes(fld.parts[fld.data[0]]).decode("utf-8",errors="replace")
from jinja2 import Environment
def _raise(m): raise Exception(m)
env=Environment(); env.globals["raise_exception"]=_raise; T=env.from_string(tmpl)
INSTRUCTION=("Complete the following Python function. Reply with the COMPLETE function definition "
             "inside a single ```python code block, and nothing else.\n\n{prompt}")
def extract_code(t):
    if "```" not in t: return t.strip()
    b=t.split("```")[1]
    if b.startswith("python"): b=b[6:]
    return b.strip()
def comp(prompt,n):
    body={"prompt":prompt,"n_predict":n,"temperature":0.0,"top_k":1,"cache_prompt":True,"stream":False}
    try:
        d=json.loads(urllib.request.urlopen(urllib.request.Request(
           f"http://127.0.0.1:{port}/completion",data=json.dumps(body).encode(),
           headers={"Content-Type":"application/json"}),timeout=600).read())
        return d.get("content","")
    except Exception as e:                     # one bad request must not nuke the whole run
        print(f"  WARN comp failed (n_predict={n}): {e}", flush=True)
        return ""
prompts={json.loads(l)["task_id"]:json.loads(l)["prompt"] for l in open(problems_f)}
ids=[x.strip() for x in open(subset_f) if x.strip()][:subn]
for B in budgets:
    sf=open(f"{out}/b{B}__samples.jsonl","w"); closed=0
    for tid in ids:
        P=T.render(messages=[{"role":"user","content":INSTRUCTION.format(prompt=prompts[tid])}],
                   add_generation_prompt=True, enable_thinking=True, reasoning_effort="xhigh")
        t1=comp(P,B)                                   # budgeted thinking
        if "</think>" in t1: closed+=1; P2=P+t1        # closed within budget -> continue answer
        else: P2=P+t1+"\n</think>\n\n"                 # force close, then answer
        a=comp(P2,768)
        full=t1+a; ans=full.split("</think>")[-1]
        sf.write(json.dumps({"task_id":tid,"solution":extract_code(ans)})+"\n")
    sf.close()
    print(f"  budget {B}: closed-within-budget {closed}/{len(ids)}")
PY

echo "== stop server, score with evalplus =="
kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; trap - EXIT; sleep 5
"$EVALPY" - "$OUT" "$SUBSET" "$PROBLEMS" "$BUDGETS" "$SUBSET_N" <<'PY'
import json,os,subprocess,sys
out,subset_f,problems_f,budgets_s,subn=sys.argv[1:6]
budgets=[int(b) for b in budgets_s.split()]; ids=[x.strip() for x in open(subset_f) if x.strip()][:int(subn)]
all_ids=[json.loads(l)["task_id"] for l in open(problems_f)]
def st(info,w):
    if isinstance(info,list): info=info[0]
    v=info.get(w+"_status") if isinstance(info,dict) else None
    return v[0] if isinstance(v,list) else v
print()
for B in budgets:
    sol={json.loads(l)["task_id"]:json.loads(l)["solution"] for l in open(f"{out}/b{B}__samples.jsonl")}
    pad=f"{out}/b{B}__padded.jsonl"
    open(pad,"w").write("\n".join(json.dumps({"task_id":t,"solution":sol.get(t,"")}) for t in all_ids))
    res=f"{out}/b{B}__padded_eval_results.json"
    if os.path.exists(res): os.remove(res)
    subprocess.run([sys.executable,"-m","evalplus.evaluate","--dataset","humaneval","--samples",pad],
                   capture_output=True,text=True)
    d=json.load(open(res)); ev=d.get("eval",d); b=p=0
    for t in ids:
        i=ev.get(t)
        if i: b+=(st(i,"base")=="pass"); p+=(st(i,"plus")=="pass")
    print(f"  xhigh @ budget {B:>4}:  base {b}/{len(ids)} ({b/len(ids)*100:4.1f}%)   plus {p}/{len(ids)} ({p/len(ids)*100:4.1f}%)")
print("\n  context: instruct 95.0% | low 93.3% | med 88.3% | xhigh(cap6144,trunc) 45.0% | ThinkingCap 93.3%")
PY
echo "=== DONE ==="
