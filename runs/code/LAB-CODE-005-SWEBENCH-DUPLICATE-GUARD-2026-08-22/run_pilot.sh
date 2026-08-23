#!/usr/bin/env bash
set -u

export DOCKER_HOST=tcp://127.0.0.1:2375
export MSWEA_CONFIGURED=true
export PATH=/mnt/wsl/docker-desktop/cli-tools/usr/bin:$PATH

root=/mnt/c/projects/tare.tools.local-labs
run_dir="$root/runs/code/LAB-CODE-005-SWEBENCH-DUPLICATE-GUARD-2026-08-22"
runner="$root/tools/benchmarks/mini_swe_verified_duplicate_guard.py"
python=/home/augus/mini-swe-agent-venv/bin/python
base=/home/augus/src/mini-swe-agent-lab-code-003/src/minisweagent/config/benchmarks/swebench.yaml
output="$run_dir/model-pilot-qwen38-duplicate-guard"

run_until_count() {
  target="$1"
  limit="$2"
  while true; do
    "$python" "$runner" --manifest "$run_dir/dataset_manifest.json" --base-config "$base" \
      --model-config "$run_dir/mini_swe_qwen38.yaml" --output "$output" --limit "$limit"
    rc=$?
    count=$("$python" -c "import json; from pathlib import Path; p=Path('$output/preds.json'); print(len(json.loads(p.read_text())) if p.exists() else 0)")
    echo "ORCHESTRATOR predictions=$count/$target runner_exit=$rc"
    if [[ "$count" -ge "$target" ]]; then return 0; fi
    if [[ "$rc" -ne 2 ]]; then return "$rc"; fi
  done
}

run_until_count 3 3 || exit $?
"$python" - <<'PY'
import json
from pathlib import Path
p = Path('/mnt/c/projects/tare.tools.local-labs/runs/code/LAB-CODE-005-SWEBENCH-DUPLICATE-GUARD-2026-08-22/model-pilot-qwen38-duplicate-guard/preds.json')
preds = json.loads(p.read_text())
ids = ['astropy__astropy-12907', 'django__django-11603', 'django__django-13401']
missing = [i for i in ids if not preds.get(i, {}).get('model_patch', '').strip()]
if missing:
    print(f'STAGE_A_REJECT empty={missing}')
    raise SystemExit(3)
print('STAGE_A_PASS 3/3 nonempty')
PY
stage_rc=$?
if [[ "$stage_rc" -ne 0 ]]; then exit "$stage_rc"; fi
run_until_count 10 0
