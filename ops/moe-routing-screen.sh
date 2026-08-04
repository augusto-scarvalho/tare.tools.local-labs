#!/usr/bin/env bash
# moe-routing-screen.sh — the standing "new-MoE gate" for learned expert placement (S1b).
#
# WHY: a learned/dynamic expert-residency controller (IDEAS_BACKLOG S1b, §66) only pays on a
# MoE with CONCENTRATED routing. Modern MoE training load-balances by design (aux-loss /
# aux-loss-free), so concentration is rare. Before building any per-model placement work,
# screen the model here — it's ~1 command. Build S1b ONLY if a model FAILS this screen.
#
# READ: uniform baseline = top-10% of (layer,expert) pairs carries ~10% of decode routing.
#       CONCENTRATED (S1b-worthy) = top-10% carries MUCH more (e.g. >=40-50%).
#       As of 2026-08-04 all 5 on-disk MoEs screened 12-22% (all load-balanced → S1b null).
#
# Usage: moe-routing-screen.sh [model-dir-name ...]   (defaults to the on-disk MoE set)
set -u
BIN=/home/augus/src/llama.cpp-master/build/bin/llama-moe-trace
SIM=/home/augus/src/llama.cpp-master/tools/moe-trace/simulate.py
MODELS_ROOT=/home/augus/models
PROMPT="Write a Python function that parses a CSV file, validates each row against a schema, and returns typed records. Include error handling and unit tests."
STEPS="${MOE_SCREEN_STEPS:-128}"

MODELS="${*:-gpt-oss-20b gemma-4-26b-a4b ernie-4.5-21b granite-4.0-h-small qwen36-35b-a3b}"

for d in $MODELS; do
  echo "### $d"
  dir="$MODELS_ROOT/$d"
  [ -d "$dir" ] || { echo "  (dir missing)"; continue; }
  gguf=$(ls "$dir"/*.gguf 2>/dev/null | grep -viE "mmproj|mtmd|draft|trace" | head -1)
  [ -n "$gguf" ] || { echo "  (no servable gguf)"; continue; }
  csv="/tmp/moe_screen_${d}.csv"; log="/tmp/moe_screen_${d}.log"
  MOE_TRACE_OUT="$csv" "$BIN" -m "$gguf" -ngl 99 --n-cpu-moe 99 -fa on \
    -p "$PROMPT" -n "$STEPS" > "$log" 2>&1
  if [ -s "$csv" ]; then
    python3 "$SIM" "$csv" 2>/dev/null | grep -E "trace:|top " | sed 's/^/  /'
  else
    echo "  TRACE FAILED:"; tail -3 "$log" | sed 's/^/    /'
  fi
  rm -f "$csv" "$log"
done
echo "# uniform=10/25/50%; concentrated (build S1b) = top-10% >> 10%."
