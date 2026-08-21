#!/usr/bin/env bash
# LAB-CLOSE-001 — mmap ON vs OFF decode A/B at fixed placement (real / confounded / noise?).
#
# WHY: ab_isolate.py notes a residual "-10.4% no-mmap cost" on the local build. With CPU-offloaded
# experts (ncmoe>0) the expert weights live in RAM, so mmap vs no-mmap changes how they're paged during
# decode. This isolates it: same model, same ncmoe, same guarded clocks, only --mmap flips. Follows the
# kv-quant-bench rigor: fresh llama-bench process per arm, 25s cooldown, clock check, reps + 95% CI.
#
# RESULT 2026-08-16 (qwen36-35b-a3b UD-Q4_K_M, ncmoe=6, d8192, 6 reps, clock-stable):
#     mmap ON  : 106.70 ± 2.40 t/s
#     mmap OFF : 107.31 ± 1.38 t/s
#   => NOISE. CIs fully overlap -> no real mmap effect on decode at the deploy placement (ncmoe=6). The
#      "-10.4% no-mmap residual" ab_isolate.py noted does NOT reproduce here = CONFOUNDED (other
#      placement / heat / clock), not a real mmap decode cost. Keep default (mmap ON: faster load, less RAM).
#
# Usage: bash ops/close-outs/mmap_ab.sh   (WSL; ~8 min). MODEL/NCMOE/DEPTH/REPS overridable.
set -u
cd /home/augus/src/slop.cpp-main
export CUDA_VISIBLE_DEVICES=0
M=${MODEL:-/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
NCMOE=${NCMOE:-6}
DEPTH=${DEPTH:-8192}
REPS=${REPS:-6}
BIN=./build/bin/llama-bench

arm() { # mmap-flag label
  echo "########## $2  (mmap=$1  ncmoe=$NCMOE depth=$DEPTH) ##########"
  sleep 25   # cooldown between isolated arms (heat-soak guard, per kv-quant-bench)
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader | head -1
  stdbuf -oL -eL "$BIN" -m "$M" -fa on -ncmoe "$NCMOE" -mmp "$1" \
    -p 0 -n 64 -d "$DEPTH" -r "$REPS" 2>&1 | grep -iE "\| qwen|tg|error|unsupported"
  echo
}

arm 1 "mmap ON (default)"
arm 0 "mmap OFF (--no-mmap)"
echo "=== DONE ==="
echo "# DECISION: if the two CIs overlap -> NOISE (no real mmap effect on decode at this placement)."
echo "#           if mmap OFF is materially slower/faster and CIs are clean -> REAL (record magnitude)."
