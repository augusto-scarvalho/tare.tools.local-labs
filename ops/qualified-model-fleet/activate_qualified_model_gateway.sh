#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/c/projects/tare.tools.local-labs
dropin_source="$repo/ops/qualified-model-fleet/qualified-model-gateway.conf"
dropin_target=/etc/systemd/system/llm-inference.service.d/zz-qualified-model-gateway.conf
embedding_pid_before="$(systemctl show llm-embedding.service -p MainPID --value)"

python3 "$repo/tools/agents/modelctl.py" validate
curl -fsS --max-time 3 http://127.0.0.1:8081/health >/dev/null
sudo install -m 0644 "$dropin_source" "$dropin_target"
sudo systemctl daemon-reload
sudo systemctl reset-failed llm-inference.service
sudo systemctl restart llm-inference.service

for ((attempt=1; attempt<=180; attempt++)); do
    if curl -fsS --max-time 2 http://127.0.0.1:8080/fleet/status >/tmp/qualified-fleet-status.json 2>/dev/null; then
        if python3 - /tmp/qualified-fleet-status.json <<'PY'
import json
import sys
status = json.load(open(sys.argv[1], encoding="utf-8"))
ok = (
    status.get("role") == "qualified-model-gateway"
    and status.get("current_model") == "qwen38"
    and status.get("backend_healthy") is True
    and status.get("max_resident_models") == 1
    and len(status.get("available_models", [])) == 6
)
raise SystemExit(0 if ok else 1)
PY
        then
            embedding_pid_after="$(systemctl show llm-embedding.service -p MainPID --value)"
            test "$embedding_pid_before" = "$embedding_pid_after"
            curl -fsS --max-time 3 http://127.0.0.1:8081/health >/dev/null
            printf 'QUALIFIED_GATEWAY_HEALTHY_AFTER_S=%s\n' "$attempt"
            printf 'EMBEDDING_PID=%s\n' "$embedding_pid_after"
            exit 0
        fi
    fi
    sleep 1
done

echo QUALIFIED_GATEWAY_HEALTH_TIMEOUT
systemctl status llm-inference.service --no-pager || true
journalctl -u llm-inference.service -n 100 --no-pager || true
exit 1
