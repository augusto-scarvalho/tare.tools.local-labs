#!/usr/bin/env bash
# Serve the local Mistral-Small-24B writing-quality JUDGE for A2 Gate 3 (a2_gate3_judge.py).
# Run INSIDE WSL (Ubuntu-24.04) so paths aren't MSYS-mangled:
#   wsl.exe -d Ubuntu-24.04 -- bash -lc 'bash /mnt/c/projects/local-model-lifecycle/scratch/serve_mistral_judge.sh'
# From Windows the harness reaches it at http://127.0.0.1:8090 (WSL2 localhost forwarding).
#
# Config is tuned for the JUDGE workload on THIS box (RTX 3090, 24 GB):
#   * dense 24B Q4_K_M = 14 GB -> fits ENTIRELY in VRAM, no CPU offload  -> -ngl 99
#   * workload is PREFILL-HEAVY (rubric + two ~500-tok responses ~= 1.5-2k tok in, ~150 tok out),
#     so the project's measured prefill lever applies: -b/-ub 2048 ~= 2x prefill vs the 512 default
#   * -fa on = flash attention (prefill + KV memory)
#   * MMQ (int8-TC GEMM) and CUDA graphs are ON BY DEFAULT on Ampere -> no flags (see IDEAS_BACKLOG S2)
#   * KV kept f16 (8k ctx -> KV is tiny; q4 KV would be lossless but pointless here)
#   * no MTP/spec: Mistral is dense with no draft head and decode is short
set -euo pipefail

BIN=/home/augus/src/slop.cpp-main/build/bin/llama-server
# ABLITERATED judge (Heretic-v1.2-2, imatrix Q4_K_M) -- Heretic preserves coherence better than naive
# abliteration (same tool the project's own heretic_run.py drives). Stock instruct kept for reference:
#   /home/augus/models/mistral-small-24b/mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf
MODEL=/home/augus/models/mistral-small-24b-heretic/Mistral-Small-3.2-24B-Instruct-2506-Heretic-v1.2-2.i1-Q4_K_M.gguf
PORT=8090

fuser -k ${PORT}/tcp 2>/dev/null || true    # free the port (NOT `pkill -f ...8090` -- self-matches)
sleep 1

exec "$BIN" -m "$MODEL" \
    -ngl 99 -fa on \
    -c 8192 -b 2048 -ub 2048 \
    --host 0.0.0.0 --port ${PORT}
