#!/usr/bin/env bash
set -euo pipefail

expected_model=/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf
embedding_pid_before="$(systemctl show llm-embedding.service -p MainPID --value)"

sudo rm -f /etc/systemd/system/llm-inference.service.d/zz-qualified-model-gateway.conf
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
    props = json.load(response)
raise SystemExit(0 if props.get("model_path") == sys.argv[1] else 1)
PY
        then
            embedding_pid_after="$(systemctl show llm-embedding.service -p MainPID --value)"
            test "$embedding_pid_before" = "$embedding_pid_after"
            curl -fsS --max-time 3 http://127.0.0.1:8081/health >/dev/null
            printf 'QWEN38_SINGLE_MODEL_RESTORED_AFTER_S=%s\n' "$attempt"
            exit 0
        fi
    fi
    sleep 1
done

systemctl status llm-inference.service --no-pager || true
exit 1
