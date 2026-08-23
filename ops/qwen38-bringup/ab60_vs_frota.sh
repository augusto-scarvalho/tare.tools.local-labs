#!/usr/bin/env bash
# Clean A/B: Qwen3.8 on the EXACT market-r0 n=60 HumanEval+ subset, vs recorded ThinkingCap/fable refs.
# Arms: instruct (no thinking) and thinking@medium. Measures pass@1 (base+plus via real evalplus) AND
# median generated tokens (the concision axis — the thinking-budget confound the comparison hinges on).
# MTP OFF: draft-mtp is not verified-exact on qwen35 and changes committed-token COUNT (A2 note), so both
# pass@1 and token counts are measured on plain greedy decode. temp 0. Same INSTRUCTION as quality_bench.
#
# RESULT 2026-08-16 (UD-Q4_K_XL, MTP off, temp 0, cap 6144, EXACT market-r0 n=60):
#     arm           HumanEval+ plus   base      empty(trunc@6144)
#     instruct       95.0% (57/60)    98.3%     0/60
#     think-low      93.3% (56/60)    96.7%     0/60
#     think-med      88.3% (53/60)    91.7%     0/60
#     think-xhigh    45.0% (27/60)    45.0%    31/60   <- runaway thinking eats the budget, no code emitted
#   references (same 60): ThinkingCap 93.3% | fable-tc 88.3% | fable-fusion 40.0%
#   => Qwen3.8 INSTRUCT (95.0%, zero thinking) BEATS ThinkingCap-3.6 (93.3%); think-low TIES it. MORE
#      thinking = WORSE on code (instruct>low>med); DEFAULT xhigh is a trap (truncates half at 6144 ->
#      45%; that number is truncation-deflated but the verdict holds: xhigh is impractically verbose for
#      code). DEPLOY for coding: instruct or reasoning_effort=low, NEVER xhigh.
#   NOTE: thinking-token counts were lost (harness killed the log redirect mid-run); trunc 0/0/0/31 is the
#      concision proxy. pass@1 recovered from the on-disk evalplus eval_results (scratchpad score_ab60.py).
#
# Prereq: /tmp/subset60.txt (the 60 task ids) — produced by score_ref.py.
# Usage: bash ops/qwen38-bringup/ab60_vs_frota.sh   (stage via wslx, run via run_in_background)
set -u
LLAMA=/home/augus/src/slop.cpp-main
BIN="$LLAMA/build/bin/llama-server"
MODEL=/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf
EVALPY=/home/augus/evalplus-venv/bin/python
REPO=/mnt/c/projects/local-model-lifecycle
PROBLEMS="$REPO/workloads/humaneval_plus.jsonl"
SUBSET=/tmp/subset60.txt
OUT=/home/augus/models/qwen38-27b/ab60; mkdir -p "$OUT"
PORT=8080; export CUDA_VISIBLE_DEVICES=0

echo "== boot Qwen3.8 (MTP OFF, ctx 16384 for thinking headroom) =="
"$BIN" -m "$MODEL" -c 16384 -ngl 999 -fa 1 --no-mmproj --cache-type-k q4_0 --cache-type-v q4_0 \
  --jinja -np 1 --host 127.0.0.1 --port "$PORT" </dev/null >/tmp/ab60_server.log 2>&1 &
PID=$!; trap 'kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null' EXIT
for i in $(seq 1 200); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "SERVER FAILED"; tail -8 /tmp/ab60_server.log; exit 1; }

echo "== generate (instruct + thinking@medium) over the 60 =="
python3 - "$PROBLEMS" "$SUBSET" "$OUT" "$PORT" <<'PY'
import json,sys,statistics as st,urllib.request,urllib.error
problems_f,subset_f,out,port=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
INSTRUCTION=("Complete the following Python function. Reply with the COMPLETE function definition "
             "inside a single ```python code block, and nothing else: no explanation, no tests, no "
             "example usage.\n\n{prompt}")
def extract_code(t):
    if "```" not in t: return t.strip()
    b=t.split("```")[1]
    if b.startswith("python"): b=b[6:]
    return b.strip()
prompts={json.loads(l)["task_id"]:json.loads(l)["prompt"] for l in open(problems_f,encoding="utf-8")}
ids=[x.strip() for x in open(subset_f) if x.strip()]
# reasoning_effort is a NATIVE Qwen3.8 knob (template-verified): low / medium / xhigh (default; 'high'
# aliases to xhigh). Sweep all + instruct, cap 6144 (headroom so xhigh doesn't truncate), record tokens.
ARMS=[("instruct",{"enable_thinking":False}),
      ("think-low",{"enable_thinking":True,"reasoning_effort":"low"}),
      ("think-med",{"enable_thinking":True,"reasoning_effort":"medium"}),
      ("think-xhigh",{"enable_thinking":True,"reasoning_effort":"xhigh"})]
for arm,ctk in ARMS:
    sf=open(f"{out}/{arm}__samples.jsonl","w"); toks=[]; trunc=0
    for tid in ids:
        body={"messages":[{"role":"user","content":INSTRUCTION.format(prompt=prompts[tid])}],
              "max_tokens":6144,"temperature":0.0,"top_k":1,"stream":False,"chat_template_kwargs":ctk}
        try:
            r=json.load(urllib.request.urlopen(urllib.request.Request(
               f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(body).encode(),
               headers={"Content-Type":"application/json"}),timeout=600))
            ch=r["choices"][0]; text=ch["message"].get("content") or ""
            pn=r.get("timings",{}).get("predicted_n") or 0; trunc+=(ch.get("finish_reason")=="length")
        except urllib.error.HTTPError as e:
            text=""; pn=0
        toks.append(pn)
        sf.write(json.dumps({"task_id":tid,"solution":extract_code(text)})+"\n")
    sf.close()
    print(f"  {arm}: median gen tokens ~{int(st.median(toks))}  (truncated@4096: {trunc}/{len(ids)})")
PY

echo "== stop server before scoring =="
kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; trap - EXIT; sleep 5

echo "== score each arm with real evalplus over the 60 =="
"$EVALPY" - "$OUT" "$SUBSET" "$PROBLEMS" <<'PY'
import json,os,subprocess,sys
out,subset_f,problems_f=sys.argv[1],sys.argv[2],sys.argv[3]
ids=set(x.strip() for x in open(subset_f) if x.strip())
all_ids=[json.loads(l)["task_id"] for l in open(problems_f,encoding="utf-8")]
def status(info,which):
    if isinstance(info,list): info=info[0]
    v=info.get(which+"_status") if isinstance(info,dict) else None
    return v[0] if isinstance(v,list) else v
def score(arm):
    sol={json.loads(l)["task_id"]:json.loads(l)["solution"] for l in open(f"{out}/{arm}__samples.jsonl")}
    padded=f"{out}/{arm}__padded.jsonl"
    with open(padded,"w") as f:
        for tid in all_ids: f.write(json.dumps({"task_id":tid,"solution":sol.get(tid,"")})+"\n")
    res=f"{out}/{arm}__padded_eval_results.json"
    if os.path.exists(res): os.remove(res)   # bust stale cache (harness-bug fix 2)
    subprocess.run([sys.executable,"-m","evalplus.evaluate","--dataset","humaneval","--samples",padded],
                   capture_output=True,text=True)
    d=json.load(open(res)); ev=d.get("eval",d); b=p=0
    for tid in ids:
        info=ev.get(tid)
        if not info: continue
        b+=(status(info,"base")=="pass"); p+=(status(info,"plus")=="pass")
    return b,p,len(ids)
print()
for arm in ["instruct","think-low","think-med","think-xhigh"]:
    b,p,n=score(arm)
    print(f"  Qwen3.8 {arm:12s}  base {b}/{n} ({b/n*100:4.1f}%)   plus {p}/{n} ({p/n*100:4.1f}%)")
print("\n  reference (same 60):  ThinkingCap plus 56/60 (93.3%)  |  fable-tc 53/60 (88.3%)  |  fable-fusion 24/60 (40.0%)")
PY
echo "=== DONE ==="
