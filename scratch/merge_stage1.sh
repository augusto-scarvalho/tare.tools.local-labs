#!/bin/bash
# A2 Stage 1: full-rank task-arithmetic merge  W_out = W_Fable + lambda*(W_TC - W_base)
# for each lambda, then convert -> quantize to Q4_K_M (no imatrix, so ALL arms incl. plain Fable
# are quantized identically = matched). Aggressive cleanup of fp16/f16-gguf intermediates so the
# disk peak stays bounded. Run AFTER the fp16 downloads finish.
#
#   bash merge_stage1.sh 0.4 0.7 1.0
set -e
FP16=/home/augus/models/fp16
OUT=/home/augus/models/merges           # final Q4 GGUFs live here
LC=/home/augus/src/llama.cpp-master
PY=/home/augus/sglang-venv/bin/python3
QUANT=$LC/build/bin/llama-quantize
CONVERT=$LC/convert_hf_to_gguf.py
mkdir -p "$OUT"

convert_quant () {   # $1 = model dir (safetensors), $2 = output tag
  local dir="$1" tag="$2"
  local f16="$OUT/${tag}-f16.gguf" q4="$OUT/${tag}-Q4_K_M.gguf"
  echo "  [convert] $tag ..."
  "$PY" "$CONVERT" "$dir" --outfile "$f16" --outtype f16
  echo "  [quantize] $tag -> Q4_K_M ..."
  "$QUANT" "$f16" "$q4" Q4_K_M
  rm -f "$f16"                            # drop the 54GB f16 gguf, keep the 18GB Q4
  echo "  [done] $q4  ($(du -h "$q4" | cut -f1))"
}

# Matched baseline: plain Fable, quantized by US the same way as the merges.
if [ ! -f "$OUT/fable-plain-Q4_K_M.gguf" ]; then
  echo "[$(date +%H:%M)] baseline: plain Fable"
  convert_quant "$FP16/fable" "fable-plain"
fi

for L in "$@"; do
  tag="fable-tc-l${L}"
  if [ -f "$OUT/${tag}-Q4_K_M.gguf" ]; then echo "skip $tag (exists)"; continue; fi
  echo "[$(date +%H:%M)] === merge lambda=$L (raw task-arithmetic) ==="
  merged="$OUT/${tag}-fp16"
  "$PY" /mnt/c/projects/local-model-lifecycle/a2_merge_raw.py \
      --base "$FP16/base" --tc "$FP16/tc" --fable "$FP16/fable" --lam "$L" --out "$merged"
  convert_quant "$merged" "$tag"
  rm -rf "$merged"                        # drop the 56GB merged fp16, keep the Q4
  echo "[$(date +%H:%M)] lambda=$L complete"
done
echo "[$(date +%H:%M)] STAGE1 MERGES DONE"; ls -lh "$OUT"/*.gguf
