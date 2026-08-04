#!/usr/bin/env bash
# spec-drafter-bench.sh — the standing drafter-regression gate for spec-decode (IDEAS_BACKLOG S3).
#
# WHY: the fork ships a multi-drafter priority chain (`--spec-type a,b,...`, cheap-first order in
# common/speculative.cpp:2357). Stacking a low-quality drafter on top of MTP can REDUCE throughput because
# the higher-priority cheap drafter preempts MTP's good draft with a bad one and pays wasted verification.
# Measured 2026-08-04 on a repetitive-code corpus (deploy model, enable_thinking:false, temp 0, 512 tok):
#     draft-mtp                 ~135 t/s   92.4% accept (402/435)   <- WINNER, deploy default
#     ngram-simple              ~72  t/s   18.6% accept (21/113)    <- catastrophic on our model
#     draft-mtp,ngram-simple    ~115 t/s   75.3% accept (402/534)   <- -15% regression vs MTP alone
# => keep `--spec-type draft-mtp` alone. Re-run this if the drafter config or the MTP head changes; a new
#    drafter is worth shipping ONLY if it beats plain draft-mtp decode t/s here (quality-neutral, temp 0).
#
# Usage: spec-drafter-bench.sh            (defaults to the 3 configs above)
#        SPECS="draft-mtp ngram-mod" spec-drafter-bench.sh
set -u
BIN=/home/augus/src/llama.cpp-master/build/bin/llama-server
MODEL="${MODEL:-/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
PORT="${PORT:-8080}"
NTOK="${NTOK:-512}"
SPECS="${SPECS:-draft-mtp ngram-simple draft-mtp,ngram-simple}"

PROMPT_FILE="$(mktemp)"
cat > "$PROMPT_FILE" <<'EOF'
You are refactoring a Python data model. Given the dataclass below, write a plain Python class
`UserProfileValidated` that stores the SAME fields but with explicit @property getters and setters, where each
setter validates the type and raises TypeError with a clear message on mismatch. Reproduce every field. Do not
skip any.

@dataclass
class UserProfile:
    user_id: int
    username: str
    email: str
    full_name: str
    age: int
    is_active: bool
    is_admin: bool
    created_at: str
    updated_at: str
    last_login: str
    login_count: int
    bio: str
    avatar_url: str
    phone_number: str
    country_code: str
    timezone: str
    locale: str
    email_verified: bool
    phone_verified: bool
    two_factor_enabled: bool

Write the full class now.
EOF

client() {  # $1 = prompt file
python3 - "$1" "$NTOK" "$PORT" <<'PY'
import json, sys, urllib.request
prompt = open(sys.argv[1]).read(); ntok = int(sys.argv[2]); port = sys.argv[3]
payload = {"messages":[{"role":"user","content":prompt}],"max_tokens":ntok,
           "temperature":0.0,"top_k":1,"stream":False,
           "chat_template_kwargs":{"enable_thinking":False}}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
r = json.load(urllib.request.urlopen(req, timeout=600))
t = r.get("timings",{}); ch = r["choices"][0]
print(f"  decode = {round(t.get('predicted_per_second',0),1)} t/s | "
      f"predicted_n = {t.get('predicted_n')} | draft_n = {t.get('draft_n')} | "
      f"finish = {ch.get('finish_reason')} | content_chars = {len(ch.get('message',{}).get('content') or '')}")
PY
}

for spec in $SPECS; do
  echo "### spec-type = $spec"
  slog="$(mktemp)"
  "$BIN" -m "$MODEL" -fa on --n-cpu-moe 8 --ctx-size 8192 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --spec-type "$spec" --spec-draft-n-max 4 \
    --batch-size 2048 --ubatch-size 2048 \
    --host 127.0.0.1 --port "$PORT" -np 1 </dev/null > "$slog" 2>&1 &
  pid=$!
  ok=0
  for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }; sleep 2; done
  if [ "$ok" = 1 ]; then
    client "$PROMPT_FILE"                    # warm
    client "$PROMPT_FILE"                    # steady (report this one)
    grep -oE "draft acceptance = [0-9.]+ \( *[0-9]+ accepted / *[0-9]+ generated\), mean len = *[0-9.]+" "$slog" | tail -1 | sed 's/^/  accept: /'
  else
    echo "  SERVER FAILED TO START"; tail -8 "$slog"
  fi
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  rm -f "$slog"; sleep 3
done
rm -f "$PROMPT_FILE"
echo "# winner = highest steady decode t/s at temp 0 (output is byte-identical across drafters)."
