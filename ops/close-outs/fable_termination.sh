#!/usr/bin/env bash
# LAB-CLOSE-002 — fable-fusion-711 non-termination probe.
#
# WHY: the concern is that fable-fusion-711 (DavidAU NEO merge, arch qwen35 dense) runs away without
# emitting EOS -> unusable for an agentic loop. Metric = termination_rate: fraction of SHORT prompts that
# finish with finish_reason "stop" (real EOS) vs "length" (hit the cap = non-terminating). Sweep
# enable_thinking {false,true} x cap {512,2048}, temp 0. If it hits "length" on trivial prompts even at
# 2048 -> DISQUALIFIED for the agentic role. --jinja mandatory (thinking-model template).
#
# RESULT 2026-08-16 (Q4_K_M, q4_0 KV, temp 0):
#     think=OFF (instruct): 4/4 terminate (stop), tiny outputs (2/7/4/21 tok) at both caps 512 & 2048.
#     think=ON  (thinking): 2/4 terminate; the rest hit "length" — ran to the 512 AND 2048 ceiling on
#         TRIVIAL prompts (e.g. rambled 1674 tokens to "say hello in three words"). The <think> runs away.
#   => VERDICT: fable-fusion-711 is TERMINATION-SAFE in instruct mode ONLY. With thinking ON it is
#      NON-TERMINATING on trivial prompts -> DISQUALIFIED for an agentic role that uses thinking.
#      (Consistent with its weak 40% HumanEval+; explains the value of ThinkingCap's concision.)
#
# Usage: bash ops/close-outs/fable_termination.sh   (via ops/wsl/wslx.sh; run_in_background if slow)
set -u
LLAMA=/home/augus/src/llama.cpp-master
BIN="$LLAMA/build/bin/llama-server"
MODEL=/home/augus/models/fable-fusion-711/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf
PORT=8080; export CUDA_VISIBLE_DEVICES=0

echo "== boot fable-fusion-711 (dense, q4_0 KV) =="
"$BIN" -m "$MODEL" -c 8192 -ngl 999 -fa 1 --no-mmproj --cache-type-k q4_0 --cache-type-v q4_0 \
  --jinja -np 1 --host 127.0.0.1 --port "$PORT" </dev/null >/tmp/fable_term_server.log 2>&1 &
PID=$!; trap 'kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null' EXIT
for i in $(seq 1 200); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "SERVER FAILED"; tail -8 /tmp/fable_term_server.log; exit 1; }

python3 - "$PORT" <<'PY'
import json,sys,urllib.request
port=sys.argv[1]
PROMPTS=[
  "What is the capital of France? Answer in one word.",
  "Write a Python one-liner that returns the square of n. Only the code.",
  "Say hello in exactly three words.",
  "Is 17 prime? Answer 'yes' or 'no' and one short sentence.",
]
def ask(p, think, cap):
    body={"messages":[{"role":"user","content":p}],"max_tokens":cap,"temperature":0.0,"top_k":1,
          "stream":False,"chat_template_kwargs":{"enable_thinking":think}}
    r=json.load(urllib.request.urlopen(urllib.request.Request(
       f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(body).encode(),
       headers={"Content-Type":"application/json"}),timeout=600))
    ch=r["choices"][0]; fr=ch.get("finish_reason"); pn=r.get("timings",{}).get("predicted_n")
    return fr, pn
print(f"{'think':>6} {'cap':>5} {'term_rate':>10} {'detail (finish_reason / predicted_n)':>40}")
for think in (False, True):
    for cap in (512, 2048):
        rows=[]; term=0
        for p in PROMPTS:
            fr,pn=ask(p,think,cap)
            rows.append(f"{fr}:{pn}"); term += (fr=="stop")
        print(f"{str(think):>6} {cap:>5} {term}/{len(PROMPTS):<8} {'  '.join(rows)}")
print("\n# term_rate = fraction finishing via EOS ('stop'). 'length' on trivial prompts = NON-TERMINATING.")
PY
echo "=== DONE ==="
