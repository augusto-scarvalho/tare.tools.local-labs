#!/usr/bin/env bash
set -euo pipefail

profile=${1:-32k}
case "$profile" in
    32k|131k) ;;
    *) echo "usage: $0 [32k|131k]" >&2; exit 2 ;;
esac
dropin_source="/mnt/c/projects/tare.tools.local-labs/ops/qwen38-bringup/serve_hauhaucs_aggressive_${profile}.conf"
dropin_target=/etc/systemd/system/llm-inference.service.d/serve-hauhaucs-aggressive.conf
expected_model=/home/augus/models/qwen38-27b/hauhaucs-aggressive-993a5971/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf

curl -fsS --max-time 3 http://127.0.0.1:8081/health >/dev/null
sudo install -m 0644 "$dropin_source" "$dropin_target"
sudo systemctl daemon-reload
sudo systemctl reset-failed llm-inference.service
sudo systemctl restart llm-inference.service

for ((attempt=1; attempt<=180; attempt++)); do
    if curl -fsS --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
        if python3 - "$expected_model" <<'PY'
import json
import sys
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8080/props", timeout=3) as response:
    observed = json.load(response).get("model_path")
raise SystemExit(0 if observed == sys.argv[1] else 1)
PY
        then
            printf 'HEALTHY_AFTER_S=%s\n' "$attempt"
            curl -fsS --max-time 3 http://127.0.0.1:8081/health >/dev/null
            exit 0
        fi
    fi
    sleep 1
done

echo HEALTH_TIMEOUT
systemctl status llm-inference.service --no-pager || true
journalctl -u llm-inference.service -n 80 --no-pager || true
exit 1
