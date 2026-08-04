#!/usr/bin/env bash
# Undervolt/clock-lock validation bench. Runs llama-bench under a temp/power/clock sampler.
set -o pipefail
BIN=/home/augus/src/llama.cpp-master/build/bin/llama-bench
M=/home/augus/models/gpt-oss-20b/gpt-oss-20b-Q4_K_M.gguf
S=/tmp/uvsample.csv
LABEL="${1:-baseline}"
rm -f "$S"
( for i in $(seq 1 1200); do nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm --format=csv,noheader,nounits >> "$S"; sleep 0.5; done ) &
SP=$!
echo "=== llama-bench ($LABEL) ==="
"$BIN" -m "$M" -ngl 99 -fa on -p 2048 -n 512 -r 3 -o md
RC=$?
kill "$SP" 2>/dev/null
echo ""
echo "=== load peaks ($LABEL) ==="
echo "PEAK_TEMP_C=$(cut -d, -f1 "$S" | sort -n | tail -1)"
echo "PEAK_POWER_W=$(cut -d, -f2 "$S" | sort -n | tail -1)"
echo "PEAK_SM_MHZ=$(cut -d, -f3 "$S" | sort -n | tail -1)"
echo "SAMPLES=$(wc -l < "$S")"
echo "BENCH_RC=$RC"
