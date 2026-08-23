#!/usr/bin/env bash
set -euo pipefail

mode="${1:?usage: launch_cache_test_server.sh nospec|mtp1|mtp2|mtp3 LOG_PATH}"
log_path="${2:?missing log path}"
flags=(
  /home/augus/src/slop.cpp/build/bin/llama-server
  -m /home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf
  --alias "qwen38-cache-${mode}"
  --host 0.0.0.0 --port 8092 --ctx-size 131072
  --flash-attn on --gpu-layers all --metrics --jinja --no-mmproj
  --cache-type-k q4_0 --cache-type-v q4_0
  -np 1 --ctx-checkpoints 32
  --slot-save-path /home/augus/lab-slot-cache-20260822
)

case "$mode" in
  nospec) ;;
  mtp1) flags+=(--spec-type draft-mtp --spec-draft-n-max 1) ;;
  mtp2) flags+=(--spec-type draft-mtp --spec-draft-n-max 2) ;;
  mtp3) flags+=(--spec-type draft-mtp --spec-draft-n-max 3) ;;
  *) echo "unsupported mode: $mode" >&2; exit 2 ;;
esac

mkdir -p "$(dirname "$log_path")"
systemctl stop lab-cache-test.service 2>/dev/null || true
systemctl reset-failed lab-cache-test.service 2>/dev/null || true
systemd-run --unit=lab-cache-test --collect --property=User=augus \
  --property="StandardOutput=append:${log_path}" \
  --property="StandardError=append:${log_path}" \
  "${flags[@]}"
systemctl show lab-cache-test.service -p ActiveState -p MainPID
