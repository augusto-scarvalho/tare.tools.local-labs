#!/usr/bin/env bash
# One-session serve+bench so the WSL background server survives (lmctl-style detach gotcha).
# $1 = reasoning mode passed to llama-server (e.g. "off", or "" for default), $2 = bench tag,
# $3 = extra llama-server args (e.g. "--reasoning-budget 256"), $4 = n-per-cat (default 25)
set -u
MODE="${1:-off}"
TAG="${2:-gemma-test}"
EXTRA="${3:-}"
NPC="${4:-25}"
BIN=/home/augus/src/slop.cpp-main/build/bin/llama-server
MODEL=/home/augus/models/gemma-4-12b-vision/gemma-4-12B-it-Q4_0.gguf
MMPROJ=/home/augus/models/gemma-4-12b-vision/mmproj-gemma-4-12B-it-Q8_0.gguf
PY=/home/augus/sglang-venv/bin/python
BENCH=/mnt/c/projects/local-model-lifecycle/vlm_vqa_bench.py

fuser -k 8092/tcp 2>/dev/null; sleep 1
REAS=""
[ -n "$MODE" ] && REAS="--reasoning $MODE"
echo ">> serving gemma  reasoning='$MODE'  extra='$EXTRA'"
$BIN -m "$MODEL" --host 0.0.0.0 --port 8092 --mmproj "$MMPROJ" \
     -ngl 99 -fa on --ctx-size 8192 --jinja $REAS $EXTRA > /tmp/gemma_rt.log 2>&1 &
SRV=$!
for i in $(seq 1 90); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8092/health)" = "200" ]; then
    echo ">> HEALTHY after ${i}s"; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo ">> SERVER DIED"; tail -20 /tmp/gemma_rt.log; exit 1; fi
  sleep 1
done
$PY "$BENCH" --tag "$TAG" --n-per-cat "$NPC" 2>&1 | grep -v -iE "warning|generating|examples/s"
echo ">> done, killing server"
kill $SRV 2>/dev/null
