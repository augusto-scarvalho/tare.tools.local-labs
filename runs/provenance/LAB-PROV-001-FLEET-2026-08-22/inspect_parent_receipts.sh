#!/usr/bin/env bash
set -euo pipefail

for parent in base tc fable; do
  root="/home/augus/models/fp16/$parent"
  echo "==== $parent"
  find "$root/.cache/huggingface" -maxdepth 3 -type f \
    -printf '%p|%s\n' 2>/dev/null | sort
done
