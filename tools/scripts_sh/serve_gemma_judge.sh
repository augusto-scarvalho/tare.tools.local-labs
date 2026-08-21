#!/usr/bin/env bash
# Serve the local Gemma-4 uncensored JUDGE (2nd local seat, Google lineage) for A2 Gate 3.
# Run INSIDE WSL (Ubuntu-24.04):
#   wsl.exe -d Ubuntu-24.04 -- bash -lc 'bash /mnt/c/projects/local-model-lifecycle/scratch/serve_gemma_judge.sh'
# Reached from Windows at http://127.0.0.1:8091 (distinct port from the Mistral judge on 8090).
#
# Gemma-4-26B-A4B = a MoE (26B total, ~4B active) -> fast decode, Q4_K_M ~16.8 GB fits fully in the
# 3090's 24 GB (no CPU offload -> -ngl 99). "heretic" = coherence-preserving abliteration (same method
# as the Mistral judge); "antislop" reduces repetitive LLM slop (apt for a prose judge). Build supports
# GEMMA4 arch (verified 2026-08-05). Prefill-tuned like the Mistral serve script.
set -euo pipefail

BIN=/home/augus/src/slop.cpp-main/build/bin/llama-server
MODEL=/home/augus/models/gemma4-26b-heretic/Gemma-4-26B-A4B-it-heretic-antislop.i1-Q4_K_M.gguf
PORT=8091

fuser -k ${PORT}/tcp 2>/dev/null || true    # free the port (NOT `pkill -f ...8091` -- self-matches)
sleep 1

# -c 16384 (not 8192): this is a THINKING model (up to ~2k reasoning tokens/call) and the harness
# fires 4 concurrent requests into a UNIFIED KV -> 4 x (~1.2k in + ~2k gen) overflows an 8k cache
# ("Context size has been exceeded"). 16k gives headroom; KV is cheap on Gemma (GQA + sliding window).
exec "$BIN" -m "$MODEL" \
    -ngl 99 -fa on \
    -c 16384 -b 2048 -ub 2048 \
    --host 0.0.0.0 --port ${PORT}
