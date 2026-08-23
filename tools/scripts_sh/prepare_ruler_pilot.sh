#!/usr/bin/env bash
set -euo pipefail

RULER_ROOT=${RULER_ROOT:-/home/augus/src/RULER-main}
PYTHON=${PYTHON:-/home/augus/ruler-venv/bin/python}
TOKENIZER=${TOKENIZER:-/home/augus/models/qwen38-27b/tokenizer}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/augus/datasets/ruler-qwen38-pilot}
NUM_SAMPLES=${NUM_SAMPLES:-1}
export PATH="$(dirname "$PYTHON"):$PATH"
TASKS=(
  niah_single_1 niah_single_2 niah_single_3
  niah_multikey_1 niah_multikey_2 niah_multikey_3
  niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2
)
if [[ -n "${TASKS_OVERRIDE:-}" ]]; then
  read -r -a TASKS <<< "$TASKS_OVERRIDE"
fi

cd "$RULER_ROOT/scripts/data"
for target_length in 65536 131072; do
  generation_length=$((target_length - 50))
  for task in "${TASKS[@]}"; do
    "$PYTHON" prepare.py \
      --save_dir "$OUTPUT_ROOT/$target_length" \
      --benchmark synthetic \
      --subset test \
      --task "$task" \
      --tokenizer_path "$TOKENIZER" \
      --tokenizer_type hf \
      --max_seq_length "$generation_length" \
      --model_template_type base \
      --num_samples "$NUM_SAMPLES" \
      --random_seed 42 \
      --prepare_for_ns
    output_file="$OUTPUT_ROOT/$target_length/$task/test.jsonl"
    test -f "$output_file"
    test "$(wc -l < "$output_file")" -eq "$NUM_SAMPLES"
  done
done
