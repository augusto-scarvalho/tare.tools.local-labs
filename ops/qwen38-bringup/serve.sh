#!/usr/bin/env bash
# serve.sh — the FROZEN Qwen3.8-27B agentic-coding launcher (all phase results baked in).
#
# Config rationale (all MEASURED — see README.md phases):
#   * weights UD-Q4_K_XL  : 4-bit imatrix knee; MTP head present (Phase 1). IQ4_XS = lighter fallback.
#   * -fa + q4_0/q4_0 KV  : q4_0 symmetric KV is LOSSLESS on this GDN hybrid (Phase 3) and cheap
#                           (~4GB @256k) -> full 256k fits resident. Symmetric only (asymmetric = -57%).
#   * draft-mtp n-max 3   : built-in nextn head, ~2.1-2.2x decode on code (Phase 4). n3 > n2.
#   * ctx-checkpoints     : prefix reuse works, 52s->0.66s warm turn, no #24055 (Phase 2).
#   * --jinja             : REQUIRED for tool-calling; without it tool-call parsing breaks.
#   * --no-mmproj         : vision not needed for code; saves VRAM.
#   * instruct/agent loop : send chat_template_kwargs {enable_thinking:false} per-request (model defaults
#                           to thinking). For CODING, measured best -> WORST is instruct > low > med >>
#                           xhigh; the DEFAULT xhigh runs away and truncates (ab60_vs_frota.sh). So use
#                           enable_thinking:false (or reasoning_effort "low"), NEVER xhigh, for code.
#                           Sampling for code: temp 0.7 top_p 0.8 top_k 20 min_p 0; consider presence_penalty 0-0.5.
#
# CLIENT HYGIENE for prefix reuse: keep the system prompt + tool schemas byte-stable; put volatile
# content (timestamps, session ids) at the END of the prompt, or every turn re-prefills.
#
# VALIDATED 2026-08-16 (this exact config): boots @ -c 262144, health OK, coding request 80.4 t/s decode
# (draft-mtp active). VRAM 24193/24576 MiB = 98.4% -> fits 256k, stable for -np 1 (KV is preallocated, so
# that's the steady peak; won't grow). It's tight though: for headroom (or if the 3090 must share VRAM),
# set CTX=131072 (~22GB) -- and note the model's effective multi-fact retrieval degrades well before 256k
# anyway (Phase 3 observation), so 131072 is the saner everyday default; 256k is there when you need it.
#
# Usage: bash ops/wsl/wslx.sh ops/qwen38-bringup/serve.sh              # default UD-Q4_K_XL @256k
#        MODEL=<gguf> CTX=131072 bash ops/qwen38-bringup/serve.sh      # overrides
set -u
LLAMA=${LLAMA:-/home/augus/src/llama.cpp-master}
BIN="$LLAMA/build/bin/llama-server"
MODEL=${MODEL:-/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf}
CTX=${CTX:-262144}
PORT=${PORT:-8080}
export CUDA_VISIBLE_DEVICES=0

exec "$BIN" -m "$MODEL" -c "$CTX" -ngl 999 -fa 1 --no-mmproj \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 -np 1 \
  --ctx-checkpoints 32 \
  --jinja --host 127.0.0.1 --port "$PORT"
