#!/bin/bash
PY=/home/augus/sglang-venv/bin/python3
echo "=== python ==="; "$PY" --version
echo "=== bootstrap pip ==="
"$PY" -m ensurepip --upgrade 2>&1 | tail -3 || echo "ensurepip failed"
echo "=== installing mergekit ==="
"$PY" -m pip install --quiet mergekit 2>&1 | tail -15
echo "=== import check ==="
"$PY" -c "import mergekit; print('mergekit import OK', getattr(mergekit,'__version__','?'))" 2>&1 | tail -3
echo "=== CLI check ==="
ls /home/augus/sglang-venv/bin/ | grep -i mergekit
