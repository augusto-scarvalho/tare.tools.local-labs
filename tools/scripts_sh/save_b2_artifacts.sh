#!/usr/bin/env bash
# Clean up servers, capture the §B2b patch diff into the project, save probe raw outputs.
set -u
pkill -9 -f llama-server 2>/dev/null; sleep 1
PROJ=/mnt/c/projects/local-model-lifecycle
mkdir -p "$PROJ/patches" "$PROJ/runs/b2-kvram"
echo "== capturing patch diff =="
git -C /home/augus/src/slop.cpp-main diff -- src/llama-kv-cache.cpp > "$PROJ/patches/b2b-kv-host-pin.patch"
wc -l "$PROJ/patches/b2b-kv-host-pin.patch"
echo "== GPU state =="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
