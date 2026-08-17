#!/usr/bin/env bash
# math_eval.sh — Stage 2 (real): a HARD reasoning task with headroom. GSM8K was ceiling (instruct 96.7%),
# so it couldn't test whether thinking-budget helps. MATH-500 Level-5 (competition math) leaves room.
# Question: does thinking help HERE (unlike easy code/math), and does it expose a quant cliff?
#
# One MODE per invocation (short jobs dodge the WSL bg reaper):
#   MODE=instruct                     -> enable_thinking:false
#   MODE=low|medium|high|xhigh        -> enable_thinking:true + reasoning_effort=<MODE>
# Scoring: extract last \boxed{...} after </think>, normalize, sympy-equivalence vs gold (fallback
# normalized-string eq). Applied identically to every arm, so the instruct-vs-thinking comparison is
# valid even where absolute scoring under-counts. Per-problem records -> a reaped run is still scoreable.
# Uses sglang-venv python (has sympy). draft-mtp on (exact at temp 0).
#
# Usage: MODEL=<gguf> TAG=<tag> MODE=high SUBSET=50 LEVEL=5 CTX=16384 MAXTOK=8192 bash ops/qwen38-bringup/math_eval.sh
set -u
LLAMA=/home/augus/src/llama.cpp-master
BIN="$LLAMA/build/bin/llama-server"
PY=/home/augus/sglang-venv/bin/python3
MODEL=${MODEL:-/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf}
PROBLEMS=/home/augus/data/math500/test.jsonl
SUBSET=${SUBSET:-50}; LEVEL=${LEVEL:-5}; CTX=${CTX:-16384}; MAXTOK=${MAXTOK:-8192}
MODE=${MODE:-instruct}; TAG=${TAG:-qwen38-q4kxl}
PORT=8080; export CUDA_VISIBLE_DEVICES=0
OUT=/home/augus/models/qwen38-27b/math500; mkdir -p "$OUT"
RECORDS="$OUT/${TAG}__L${LEVEL}__${MODE}__records.jsonl"

echo "== boot server (draft-mtp, ctx $CTX) — MODE=$MODE TAG=$TAG L=$LEVEL =="
"$BIN" -m "$MODEL" -c "$CTX" -ngl 999 -fa 1 --no-mmproj --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 -np 1 --jinja --host 127.0.0.1 --port "$PORT" \
  </dev/null >/tmp/math_server.log 2>&1 &
PID=$!; trap 'kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null' EXIT
for i in $(seq 1 200); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "SERVER FAILED"; tail -8 /tmp/math_server.log; exit 1; }

echo "== generate (L$LEVEL subset $SUBSET, MODE=$MODE, maxtok $MAXTOK) =="
"$PY" - "$PROBLEMS" "$RECORDS" "$SUBSET" "$PORT" "$MODE" "$LEVEL" "$MAXTOK" <<'PY'
import json,sys,re,time,urllib.request,urllib.error,random
problems_f,records_f,subset,port,mode,level,maxtok=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4],sys.argv[5],int(sys.argv[6]),int(sys.argv[7])
import sympy
from sympy.parsing.sympy_parser import parse_expr

INSTRUCTION=r"Solve the problem. Give the final answer inside \boxed{}."

def extract_boxed(t):
    t=t.split('</think>')[-1]
    idx=t.rfind(r'\boxed')
    if idx<0:
        # fallback: last $...$ or last number
        m=re.findall(r'-?\d[\d,]*\.?\d*', t); return m[-1].replace(',','') if m else None
    i=idx+len(r'\boxed');
    while i<len(t) and t[i]!='{': i+=1
    if i>=len(t): return None
    depth=0; buf=[]
    for j in range(i,len(t)):
        c=t[j]
        if c=='{': depth+=1;
        elif c=='}':
            depth-=1
            if depth==0: break
        if depth>=1 and not (c=='{' and depth==1): buf.append(c)
    return ''.join(buf).strip()

def norm(s):
    if s is None: return None
    s=str(s)
    s=re.sub(r'\\(left|right|!|,|;|:|\\ )','',s)
    s=s.replace('\\left','').replace('\\right','')
    s=re.sub(r'\\text\{[^}]*\}','',s)
    s=re.sub(r'\\!|\\,|\\;|\\ ','',s)
    s=s.replace('\\dfrac','\\frac').replace('\\tfrac','\\frac')
    s=s.replace('$','').replace('\\%','').replace('%','')
    s=s.replace('\\$','').replace(' ','')
    s=s.replace('^{\\circ}','').replace('^\\circ','')
    return s.strip()

def to_sym(s):
    if s is None: return None
    s=norm(s)
    # latex -> sympy-ish
    s=re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}',r'((\1)/(\2))',s)
    s=re.sub(r'\\sqrt\{([^{}]+)\}',r'sqrt(\1)',s)
    s=re.sub(r'\\sqrt(\w)',r'sqrt(\1)',s)
    s=s.replace('\\pi','pi').replace('\\cdot','*').replace('\\times','*')
    s=s.replace('{','(').replace('}',')')
    s=re.sub(r'\\[a-zA-Z]+','',s)      # drop remaining latex commands
    s=s.replace('^','**')
    try: return sympy.simplify(parse_expr(s,evaluate=True))
    except Exception: return None

def equiv(pred,gold):
    if pred is None: return False
    if norm(pred)==norm(gold): return True
    a,b=to_sym(pred),to_sym(gold)
    if a is not None and b is not None:
        try:
            return sympy.simplify(a-b)==0
        except Exception:
            return False
    return False

probs=[]
for line in open(problems_f,encoding='utf-8'):
    d=json.loads(line)
    if int(d['level'])==level:
        probs.append({'task_id':d['unique_id'],'prompt':d['problem'],'gold':d['answer']})
probs.sort(key=lambda d:d['task_id'])
if subset<len(probs):
    rng=random.Random(20260726); idx=sorted(rng.sample(range(len(probs)),subset)); probs=[probs[i] for i in idx]
kw={'enable_thinking':False} if mode=='instruct' else {'enable_thinking':True,'reasoning_effort':mode}
rf=open(records_f,'w',encoding='utf-8'); correct=0; n=0; trunc=0
for i,p in enumerate(probs):
    body={'messages':[{'role':'user','content':INSTRUCTION+"\n\n"+p['prompt']}],
          'max_tokens':maxtok,'temperature':0.0,'top_k':1,'stream':False,'chat_template_kwargs':kw}
    t0=time.monotonic()
    try:
        r=json.load(urllib.request.urlopen(urllib.request.Request(
           f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(body).encode(),
           headers={'Content-Type':'application/json'}),timeout=900))
        text=r['choices'][0]['message'].get('content') or ''; err=None
        tm=r.get('timings',{}); pn=tm.get('predicted_n'); fr=r['choices'][0].get('finish_reason')
    except urllib.error.HTTPError as e:
        text=''; err=f'HTTP {e.code}'; pn=None; fr=None
    if fr=='length': trunc+=1
    pr=extract_boxed(text); ok=equiv(pr,p['gold']); correct+=ok; n+=1
    rf.write(json.dumps({'task_id':p['task_id'],'gold':p['gold'],'pred':pr,'correct':bool(ok),
                         'predicted_n':pn,'finish':fr,'err':err,'wall_s':round(time.monotonic()-t0,1)})+'\n'); rf.flush()
    if (i+1)%10==0: print(f'    {i+1}/{len(probs)}  acc={correct/n:.3f} trunc={trunc}',flush=True)
print(f'  MODE={mode} L{level}  acc={correct}/{n}={correct/n*100:.1f}%  truncated={trunc}')
PY
echo "=== DONE ($MODE) ==="
