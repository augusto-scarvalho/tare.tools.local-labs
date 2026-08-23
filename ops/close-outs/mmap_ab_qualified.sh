#!/usr/bin/env bash
# LAB-CLOSE-001 qualified evidence collector. Run under WSL after stopping 8080.
set -euo pipefail

OUT=${1:-/mnt/c/projects/tare.tools.local-labs/runs/close-outs/LAB-CLOSE-001-MMAP-2026-08-22/raw.log}
BIN=/home/augus/src/slop.cpp/build/bin/llama-bench
MODEL=/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf

mkdir -p "$(dirname "$OUT")"
exec > >(tee "$OUT") 2>&1

echo "campaign=LAB-CLOSE-001"
echo "commit=$(git -C /home/augus/src/slop.cpp rev-parse HEAD)"
echo "model=$MODEL"
stat --printf='model_bytes=%s\n' "$MODEL"
nvidia-smi --query-gpu=name,power.limit,driver_version --format=csv,noheader

for rep in 0 1 2 3 4 5; do
  if (( rep % 2 == 0 )); then
    arms=("on:1" "off:0")
  else
    arms=("off:0" "on:1")
  fi
  for spec in "${arms[@]}"; do
    label=${spec%%:*}
    mmap=${spec##*:}
    echo "BEGIN rep=$rep arm=$label mmap=$mmap utc=$(date -u +%FT%TZ)"
    sleep 25
    nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,power.limit --format=csv,noheader,nounits
    free -b | awk '/^Mem:/ {print "host_mem_available_bytes=" $7}'
    /usr/bin/time -v "$BIN" -m "$MODEL" -fa on -ngl -1 -ncmoe 6 -mmp "$mmap" \
      -p 0 -n 64 -d 8192 -r 1 -o jsonl
    rc=$?
    echo "END rep=$rep arm=$label mmap=$mmap rc=$rc utc=$(date -u +%FT%TZ)"
  done
done

echo "campaign_complete=1"
