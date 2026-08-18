#!/bin/bash
# Fail-fast Heretic probe on ThinkingCap fp16: validate it loads the 27B in 4-bit, the residual
# hooks work on the novel qwen3_5 (VL / hybrid-GDN) arch, trials run, and export is non-interactive
# -- BEFORE committing the ~2-3h full (200-trial) run. Low trials on purpose.
set -e
export TOKENIZERS_PARALLELISM=false
/home/augus/sglang-venv/bin/heretic \
  --model /home/augus/models/fp16/tc \
  --quantization BNB_4BIT \
  --n-trials 4 \
  --n-startup-trials 2 \
  --export-strategy MERGE \
  --study-checkpoint-dir /home/augus/heretic-tc-ckpt \
  --no-plot-residuals
echo "HERETIC_PROBE_EXIT=$?"
