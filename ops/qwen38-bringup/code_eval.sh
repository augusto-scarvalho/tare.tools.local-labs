#!/usr/bin/env bash
# LAB-CODE-001 (Qwen3.8) — HumanEval+ pass@1 via the serve endpoint, scored by REAL evalplus.
#
# Honors the repo discipline: same workloads/humaneval_plus.jsonl, same INSTRUCTION + extract_code as
# quality_bench.py, per-problem records, and evalplus (process-isolated executor) does the running —
# this script GENERATES and records, never executes model code. draft-mtp is EXACT at temp 0 (quality-
# neutral, just faster). Scoring uses the dedicated /home/augus/evalplus-venv (where evalplus lives).
#
# RESULT 2026-08-16 (UD-Q4_K_XL, full 164, temp 0, enable_thinking:false, draft-mtp-n3):
#     HumanEval  base  pass@1 = 0.933 (93.3%)
#     HumanEval+ (base+extra) pass@1 = 0.890 (89.0%)
#     format: 164/164 fenced (zero format failures); decode median ~92 t/s.
#   => Qwen3.8-27B at 4-bit is a STRONG coder. (draft-mtp is exact at temp 0 -> speed only, not quality.)
#
# Usage: MODEL=<gguf> SUBSET=164 bash ops/qwen38-bringup/code_eval.sh   (via ops/wsl/wslx.sh)
set -u
LLAMA=/home/augus/src/slop.cpp-main
BIN="$LLAMA/build/bin/llama-server"
MODEL=${MODEL:-/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf}
EVALPY=/home/augus/evalplus-venv/bin/python
REPO=/mnt/c/projects/local-model-lifecycle
PROBLEMS="$REPO/workloads/humaneval_plus.jsonl"
SUBSET=${SUBSET:-164}
TAG=${TAG:-qwen38-udq4kxl}
PORT=8080; export CUDA_VISIBLE_DEVICES=0
OUT=/home/augus/models/qwen38-27b/code_eval; mkdir -p "$OUT"
SAMPLES="$OUT/${TAG}__samples.jsonl"; RECORDS="$OUT/${TAG}__records.jsonl"

echo "== boot server (draft-mtp, ctx 8192) =="
"$BIN" -m "$MODEL" -c 8192 -ngl 999 -fa 1 --no-mmproj --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 -np 1 --jinja --host 127.0.0.1 --port "$PORT" \
  </dev/null >/tmp/code_eval_server.log 2>&1 &
PID=$!; trap 'kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null' EXIT
for i in $(seq 1 200); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "SERVER FAILED"; tail -8 /tmp/code_eval_server.log; exit 1; }

echo "== generate $SUBSET completions =="
python3 - "$PROBLEMS" "$SAMPLES" "$RECORDS" "$SUBSET" "$PORT" <<'PY'
import json,sys,time,urllib.request,urllib.error,random
problems_f,samples_f,records_f,subset,port=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),sys.argv[5]
INSTRUCTION=("Complete the following Python function. Reply with the COMPLETE function definition "
             "inside a single ```python code block, and nothing else: no explanation, no tests, no "
             "example usage.\n\n{prompt}")
def extract_code(text):
    if "```" not in text: return text.strip()
    parts=text.split("```"); block=parts[1] if len(parts)>1 else text
    if block.startswith("python"): block=block[len("python"):]
    return block.strip()
probs=[]
for line in open(problems_f,encoding="utf-8"):
    line=line.strip()
    if line: d=json.loads(line); probs.append({"task_id":d["task_id"],"prompt":d["prompt"]})
probs.sort(key=lambda d:d["task_id"])
# same seeded-subset rule as quality_bench (SUBSET_SEED) so numbers are comparable
if subset<len(probs):
    rng=random.Random(20260726); idx=sorted(rng.sample(range(len(probs)),subset)); probs=[probs[i] for i in idx]
sf=open(samples_f,"w",encoding="utf-8"); rf=open(records_f,"w",encoding="utf-8")
answered=0
for i,p in enumerate(probs):
    body={"messages":[{"role":"user","content":INSTRUCTION.format(prompt=p["prompt"])}],
          "max_tokens":768,"temperature":0.0,"top_k":1,"stream":False,
          "chat_template_kwargs":{"enable_thinking":False}}
    t0=time.monotonic()
    try:
        r=json.load(urllib.request.urlopen(urllib.request.Request(
           f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(body).encode(),
           headers={"Content-Type":"application/json"}),timeout=300))
        text=r["choices"][0]["message"].get("content") or ""; err=None
        t=r.get("timings",{}); pn,dn=t.get("predicted_n"),t.get("predicted_per_second")
    except urllib.error.HTTPError as e:
        text=""; err=f"HTTP {e.code}"; pn=dn=None
    wall=round(time.monotonic()-t0,2)
    code=extract_code(text)
    if text: answered+=1
    sf.write(json.dumps({"task_id":p["task_id"],"solution":code})+"\n")
    rf.write(json.dumps({"task_id":p["task_id"],"answered":bool(text),"fenced":"```" in text,
                         "error":err,"predicted_n":pn,"decode_tps":dn,"wall_s":wall})+"\n")
    if (i+1)%20==0: print(f"    {i+1}/{len(probs)}",flush=True)
sf.close(); rf.close()
print(f"  generated {len(probs)}; answered(non-empty)={answered}")
PY

echo "== stop server before scoring =="
kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; trap - EXIT; sleep 5

echo "== score with real evalplus (isolated executor) — pass@1 printed below =="
[ -x "$EVALPY" ] || { echo "evalplus venv missing at $EVALPY"; exit 2; }
"$EVALPY" -m evalplus.evaluate --dataset humaneval --samples "$SAMPLES" 2>&1 | tail -25
echo
echo "== format-failure count (answered vs empty; low = format failure, not wrong answer) =="
python3 - "$RECORDS" <<'PY'
import json,sys
recs=[json.loads(l) for l in open(sys.argv[1])]
ans=sum(1 for r in recs if r.get("answered")); n=len(recs)
dt=[r["decode_tps"] for r in recs if r.get("decode_tps")]
print(f"  answered(non-empty): {ans}/{n}   fenced: {sum(1 for r in recs if r.get('fenced'))}/{n}")
if dt: print(f"  decode: median ~{sorted(dt)[len(dt)//2]:.0f} t/s")
PY
echo "=== DONE ==="
