#!/usr/bin/env bash
# Reproduce the 2026-08-20 Qwen3.8 IQ4_XS requalification server.
# SLOT_SAVE_PATH=/tmp/lab-slot-cache enables explicit slot save/restore experiments;
# omit it to restore the exact baseline argv used by the requalification campaigns.
set -euo pipefail

BIN=${BIN:-/home/augus/src/slop.cpp/build/bin/llama-server}
MODEL=${MODEL:-/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-IQ4_XS.gguf}
TEMPLATE=${TEMPLATE:-/home/augus/models/templates/qwen-sharp.jinja}
PORT=${PORT:-8080}
CTX=${CTX:-32768}
extra=()
if [[ -n "${SLOT_SAVE_PATH:-}" ]]; then
    mkdir -p "$SLOT_SAVE_PATH"
    extra+=(--slot-save-path "$SLOT_SAVE_PATH")
fi

exec "$BIN" -m "$MODEL" --alias qwen38-27b --host 0.0.0.0 --port "$PORT" \
    --ctx-size "$CTX" --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --gpu-layers all --metrics --jinja --chat-template-file "$TEMPLATE" "${extra[@]}"
