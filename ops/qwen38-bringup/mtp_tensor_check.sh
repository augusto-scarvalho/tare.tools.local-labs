#!/usr/bin/env bash
# mtp_tensor_check.sh — Phase 1 gate: does this GGUF carry the MTP draft head?
#
# WHY: MTP spec-decode (`--spec-type draft-mtp`) is our +33-70% decode lever, but community reports
# CONFLICT on whether Qwen3.8-27B GGUFs ship the MTP tensors. A 5-second check settles it per-file.
#
# In these GGUFs the MTP head is the `nextn` layer: tensors `blk.<N>.nextn.{eh_proj,enorm,hnorm,
# shared_head_norm}.weight` at block N = block_count-1, plus metadata `*.nextn_predict_layers`.
#
# NOTE (learned the hard way): the base/login python3 has NO numpy, so the gguf reader silently fails
# and a naive check FALSE-NEGATIVES. This script AUTO-DISCOVERS a python that has numpy+gguf (scans
# known venvs) before falling back. Confirmed working reader: /home/augus/sglang-venv/bin/python3.
#
# PASS: nextn tensors present -> use --spec-type draft-mtp on the single GGUF.
# FAIL: none found -> a4lg/Qwen3.8-27B-MTP-ONLY-GGUF via --model-draft, or graft per their README.
#
# Usage: MODEL=/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf bash ops/qwen38-bringup/mtp_tensor_check.sh
set -u
LLAMA=${LLAMA:-/home/augus/src/slop.cpp-main}
MODEL=${MODEL:-/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf}

# --- find a python3 that can import BOTH numpy and gguf ---
find_py() {
  for py in /home/augus/sglang-venv/bin/python3 \
            /home/augus/miniforge3/envs/*/bin/python3 \
            /home/augus/*venv*/bin/python3 \
            python3; do
    [ -x "$(command -v "$py" 2>/dev/null || echo "$py")" ] 2>/dev/null || continue
    if PYTHONPATH="$LLAMA/gguf-py" "$py" -c "import numpy, gguf" >/dev/null 2>&1; then
      echo "$py"; return 0
    fi
  done
  return 1
}

echo "== inspecting: $MODEL =="
PY="$(find_py)" || PY=""
if [ -n "$PY" ]; then
  echo "(reader: $PY)"
  PYTHONPATH="$LLAMA/gguf-py" "$PY" - "$MODEL" <<'PY'
import sys
from gguf import GGUFReader
r=GGUFReader(sys.argv[1])
names=[t.name for t in r.tensors]
keys=("nextn","mtp","next_n","eh_proj","shared_head")
hit=sorted(n for n in names if any(k in n.lower() for k in keys))
print(f"  total tensors: {len(names)}   MTP-ish: {len(hit)}")
for n in hit[:40]: print("     ", n)
for k in r.fields:
    if "nextn" in k.lower() or "mtp" in k.lower():
        print(f"  META {k} present")
sys.exit(0 if hit else 1)
PY
  rc=$?
else
  echo "(no python with numpy+gguf found; falling back to llama-gguf binary)"
  GB="$LLAMA/build/bin/llama-gguf"
  if [ -x "$GB" ]; then
    "$GB" "$MODEL" r 2>/dev/null | grep -iE "nextn|mtp|eh_proj|shared_head" && rc=0 || rc=1
  else
    echo "  !! no reader available (install numpy+gguf, or build llama-gguf). Cannot verify."; rc=2
  fi
fi

case "${rc:-2}" in
  0) echo ">> PASS: MTP (nextn) head present -> --spec-type draft-mtp on this GGUF." ;;
  1) echo ">> FAIL: no MTP tensors -> use a4lg/Qwen3.8-27B-MTP-ONLY-GGUF as --model-draft." ;;
  *) echo ">> UNKNOWN: reader unavailable; do not assume either way." ;;
esac
