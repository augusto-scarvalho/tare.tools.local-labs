#!/usr/bin/env bash
set -euo pipefail

log_path=${1:?usage: launch_fable_termination_server.sh LOG_PATH}
flags=(
  /home/augus/src/slop.cpp/build/bin/llama-server
  -m /home/augus/models/fable-fusion-711/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf
  --alias fable-fusion-711-termination
  --host 0.0.0.0 --port 8092 --ctx-size 8192
  --flash-attn on --gpu-layers all --jinja --no-mmproj
  --cache-type-k q4_0 --cache-type-v q4_0 -np 1
)

mkdir -p "$(dirname "$log_path")"
systemctl stop lab-fable-termination.service 2>/dev/null || true
systemctl reset-failed lab-fable-termination.service 2>/dev/null || true
systemd-run --unit=lab-fable-termination --collect --property=User=augus \
  --property="StandardOutput=append:${log_path}" \
  --property="StandardError=append:${log_path}" \
  "${flags[@]}"
systemctl show lab-fable-termination.service -p ActiveState -p MainPID --no-pager

