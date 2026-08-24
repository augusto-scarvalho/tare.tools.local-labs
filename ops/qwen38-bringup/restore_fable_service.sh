#!/usr/bin/env bash
set -euo pipefail

expected_model=/home/augus/models/merges/fable-tc-l1.0-Q4_K_M.gguf

sudo rm -f /etc/systemd/system/llm-inference.service.d/serve-hauhaucs-aggressive.conf
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
            curl -fsS --max-time 3 http://127.0.0.1:8081/health >/dev/null
            printf 'FABLE_HEALTHY_AFTER_S=%s\n' "$attempt"
            exit 0
        fi
    fi
    sleep 1
done

systemctl status llm-inference.service --no-pager || true
exit 1
