#!/usr/bin/env bash
# gsm8k_eval.sh — Stage 2 gate: does THINKING help Qwen3.8-27B on a HARD reasoning task (GSM8K math),
# unlike HumanEval code (where instruct BEAT every thinking budget)? And does GSM8K expose a quant
# quality cliff that code did not?
#
# One MODE per invocation (keeps jobs short — the WSL bg reaper kills long multi-arm jobs):
#   MODE=instruct   -> enable_thinking:false (no reasoning)
#   MODE=low|medium|high|xhigh -> enable_thinking:true + reasoning_effort=<MODE>
# Scores by EXACT numeric match to the gold answer (GSM8K is unambiguous). Records are per-problem so a
# reaped run is still scoreable. draft-mtp on (exact at temp 0, speed only).
#
# Usage: MODEL=<gguf> TAG=<tag> MODE=high SUBSET=60 CTX=8192 bash ops/qwen38-bringup/gsm8k_eval.sh
set -u
LLAMA=/home/augus/src/llama.cpp-master
BIN="$LLAMA/build/bin/llama-server"
MODEL=${MODEL:-/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf}
REPO=/mnt/c/projects/local-model-lifecycle
PROBLEMS="$REPO/workloads/gsm8k.jsonl"
SUBSET=${SUBSET:-60}
MODE=${MODE:-instruct}
CTX=${CTX:-8192}
TAG=${TAG:-qwen38-q4kxl}
PORT=8080; export CUDA_VISIBLE_DEVICES=0
OUT=/home/augus/models/qwen38-27b/gsm8k; mkdir -p "$OUT"
RECORDS="$OUT/${TAG}__${MODE}__records.jsonl"

echo "== boot server (draft-mtp, ctx $CTX) — MODE=$MODE TAG=$TAG =="
"$BIN" -m "$MODEL" -c "$CTX" -ngl 999 -fa 1 --no-mmproj --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 -np 1 --jinja --host 127.0.0.1 --port "$PORT" \
  </dev/null >/tmp/gsm8k_server.log 2>&1 &
PID=$!; trap 'kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null' EXIT
for i in $(seq 1 200); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "SERVER FAILED"; tail -8 /tmp/gsm8k_server.log; exit 1; }

echo "== generate ($SUBSET problems, MODE=$MODE) =="
python3 - "$PROBLEMS" "$RECORDS" "$SUBSET" "$PORT" "$MODE" <<'PY'
import json,sys,re,time,urllib.request,urllib.error,random
problems_f,records_f,subset,port,mode=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4],sys.argv[5]
INSTRUCTION=("Solve this math problem. Show your work if needed, then end with the final answer on its "
             "own line in the form: #### <number>\n\n{prompt}")
def gold(a):
    a=str(a).split('####')[-1]
    m=re.findall(r'-?\d[\d,]*\.?\d*', a)
    return m[-1].replace(',','') if m else None
def pred(text):
    t=text.split('</think>')[-1]                      # answer after reasoning
    m=re.search(r'####\s*(-?\d[\d,]*\.?\d*)', t)       # prefer the #### form
    if m: return m.group(1).replace(',','')
    nums=re.findall(r'-?\d[\d,]*\.?\d*', t)            # else last number in the answer
    return nums[-1].replace(',','') if nums else None
def eq(a,b):
    if a is None or b is None: return False
    try: return abs(float(a)-float(b))<1e-6
    except: return a==b
probs=[]
for line in open(problems_f,encoding='utf-8'):
    line=line.strip()
    if line:
        d=json.loads(line); probs.append({'task_id':d['task_id'],'prompt':d['prompt'],'gold':gold(d['answer'])})
probs.sort(key=lambda d:int(d['task_id'].split('/')[-1]))
if subset<len(probs):
    rng=random.Random(20260726); idx=sorted(rng.sample(range(len(probs)),subset)); probs=[probs[i] for i in idx]
kw={'enable_thinking':False} if mode=='instruct' else {'enable_thinking':True,'reasoning_effort':mode}
rf=open(records_f,'w',encoding='utf-8'); correct=0; n=0
for i,p in enumerate(probs):
    body={'messages':[{'role':'user','content':INSTRUCTION.format(prompt=p['prompt'])}],
          'max_tokens':4096,'temperature':0.0,'top_k':1,'stream':False,'chat_template_kwargs':kw}
    t0=time.monotonic()
    try:
        r=json.load(urllib.request.urlopen(urllib.request.Request(
           f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(body).encode(),
           headers={'Content-Type':'application/json'}),timeout=600))
        text=r['choices'][0]['message'].get('content') or ''; err=None
        tm=r.get('timings',{}); pn=tm.get('predicted_n')
    except urllib.error.HTTPError as e:
        text=''; err=f'HTTP {e.code}'; pn=None
    pr=pred(text); ok=eq(pr,p['gold']); correct+=ok; n+=1
    rf.write(json.dumps({'task_id':p['task_id'],'gold':p['gold'],'pred':pr,'correct':bool(ok),
                         'predicted_n':pn,'err':err,'wall_s':round(time.monotonic()-t0,1)})+'\n'); rf.flush()
    if (i+1)%10==0: print(f'    {i+1}/{len(probs)}  running acc={correct/n:.3f}',flush=True)
print(f'  MODE={mode}  acc={correct}/{n}={correct/n*100:.1f}%')
PY
echo "=== DONE ($MODE) ==="
