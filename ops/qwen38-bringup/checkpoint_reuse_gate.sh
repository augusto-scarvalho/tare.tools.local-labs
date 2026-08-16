#!/usr/bin/env bash
# checkpoint_reuse_gate.sh — Phase 2 gate (CRITICAL): does prefix reuse actually work on this build?
#
# WHY: Qwen3.8-27B is a GDN hybrid (48/64 recurrent layers). llama-server CANNOT do token-granular
# --cache-reuse on recurrent archs; cross-turn prefix reuse happens ONLY via CONTEXT CHECKPOINTS
# (periodic snapshots of recurrent state + KV). There is an OPEN upstream regression (#24055) that,
# on some builds, forces FULL prompt re-processing every turn -> a long-context agent turn re-prefills
# ~50-150k tokens and TTFT collapses to tens of seconds. This gate proves reuse works BEFORE we tune.
#
# METHOD: one server, cache_prompt=true. Turn 1 sends a long fixed CONTEXT + question A. Turn 2 sends
# the SAME context + question B. If reuse works, turn 2 processes only the short new suffix (prompt_n
# small, prompt_ms tiny) and the server log prints `restored context checkpoint`. If broken, turn 2
# re-processes the whole context and the log prints `forcing full prompt re-processing ...`.
#
# PASS: turn-2 prompt tokens << turn-1  AND  log shows "restored context checkpoint".
# FAIL: turn-2 ~= turn-1  OR  log shows "forcing full prompt re-processing due to lack of cache data"
#       -> #24055 regression on this build. Pin a known-good build (Fable flagged b9309-era +
#       --checkpoint-every-n-tokens) and re-run. Do NOT proceed to MTP tuning until this passes.
#
# Usage: MODEL=/home/augus/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf bash ops/qwen38-bringup/checkpoint_reuse_gate.sh
set -u
LLAMA=${LLAMA:-/home/augus/src/llama.cpp-master}
BIN="$LLAMA/build/bin/llama-server"
MODEL=${MODEL:-/home/augus/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf}
PORT=${PORT:-8080}
export CUDA_VISIBLE_DEVICES=0

# --- discover the real checkpoint flag names for THIS build (they differ across builds / the #24055 fix) ---
echo "== checkpoint-related flags in this build =="
"$BIN" --help 2>&1 | grep -iE "checkpoint|cache-reuse|ctx-checkpoint" || echo "  (none found -> old/unsupported build?)"
echo

CKPT_FLAGS=${CKPT_FLAGS:---ctx-checkpoints 32}   # override if --help above shows different names

SLOG=$(mktemp)
echo "== starting server (log: $SLOG) =="
"$BIN" -m "$MODEL" -c 32768 -ngl 999 -fa 1 --no-mmproj \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  $CKPT_FLAGS --jinja -np 1 \
  --host 127.0.0.1 --port "$PORT" </dev/null > "$SLOG" 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; rm -f "$SLOG"' EXIT
for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done

# --- build a long, byte-STABLE context (~6-8k tokens of filler) shared by both turns ---
CTX=$(python3 - <<'PY'
lines=[f"Record {i:04d}: module_{i} exports handler_{i}(payload) -> status_{i%7}; retries={i%4}; owner=team_{i%9}." for i in range(700)]
print("You are reviewing this service registry. Answer only from it.\n" + "\n".join(lines))
PY
)

# pass context via env to avoid arg-length limits
export CTX
turn() {  # label question
  python3 - "$PORT" "$1" "$2" <<'PY'
import json,os,sys,urllib.request
port,label,q=sys.argv[1],sys.argv[2],sys.argv[3]
body={"messages":[{"role":"user","content":os.environ["CTX"]+"\n\nQUESTION: "+q}],
      "max_tokens":16,"temperature":0.0,"top_k":1,"stream":False,"cache_prompt":True,
      "chat_template_kwargs":{"enable_thinking":False}}
r=json.load(urllib.request.urlopen(urllib.request.Request(
   f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(body).encode(),
   headers={"Content-Type":"application/json"}),timeout=600))
t=r.get("timings",{})
print(f"{label}: prompt_n={t.get('prompt_n')}  prompt_ms={t.get('prompt_ms'):.0f}  "
      f"({t.get('prompt_per_second',0):.0f} tok/s)")
PY
}

echo "== turn 1 (cold: must process the whole context) =="
turn "TURN1" "How many records mention team_3?"
echo "== turn 2 (warm: SAME context, new question — should reuse the prefix) =="
turn "TURN2" "What status does module_42 return?"

echo
echo "== checkpoint / reuse evidence in server log =="
grep -iE "restored context checkpoint|forcing full prompt re-processing|checkpoint" "$SLOG" | tail -20 || true
echo
echo ">> PASS if TURN2 prompt_n is a small fraction of TURN1 AND log shows 'restored context checkpoint'."
echo ">> FAIL (a #24055-class regression) if TURN2 ~= TURN1 or log shows 'forcing full prompt re-processing'."
