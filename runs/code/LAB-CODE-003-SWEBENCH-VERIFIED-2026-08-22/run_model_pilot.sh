#!/usr/bin/env bash
set -u

export DOCKER_HOST=tcp://127.0.0.1:2375
export MSWEA_CONFIGURED=true

root=/mnt/c/projects/tare.tools.local-labs
run_dir="$root/runs/code/LAB-CODE-003-SWEBENCH-VERIFIED-2026-08-22"
runner="$root/tools/benchmarks/mini_swe_verified_pilot.py"
python=/home/augus/mini-swe-agent-venv/bin/python
base=/home/augus/src/mini-swe-agent-lab-code-003/src/minisweagent/config/benchmarks/swebench.yaml

while true; do
  "$python" "$runner" \
    --manifest "$run_dir/dataset_manifest.json" \
    --base-config "$base" \
    --model-config "$run_dir/mini_swe_qwen38.yaml" \
    --output "$run_dir/model-pilot-qwen38"
  rc=$?
  count=$("$python" -c "import json; from pathlib import Path; p=Path('$run_dir/model-pilot-qwen38/preds.json'); print(len(json.loads(p.read_text())) if p.exists() else 0)")
  echo "ORCHESTRATOR predictions=$count/10 runner_exit=$rc"
  if [[ "$count" -ge 10 ]]; then
    exit 0
  fi
  if [[ "$rc" -ne 2 ]]; then
    exit "$rc"
  fi
done
