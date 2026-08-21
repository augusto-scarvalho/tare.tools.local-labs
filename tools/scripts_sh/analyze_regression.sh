#!/usr/bin/env bash
# Find the fresh-master commit(s) that broke draft-mtp token-exactness for our model, by
# filtering the 720d7fa40..f5919bf45 range to paths that can change greedy decode output.
set -u
cd /home/augus/src/slop.cpp-main
RANGE="720d7fa40..upstream/master"
echo "range $RANGE -> $(git rev-list --count $RANGE) commits"
echo
echo "### (A) SAMPLING / SPEC-DECODE / MTP driver paths ###"
git log --oneline $RANGE -- \
  src/llama-sampling.cpp common/sampling.cpp common/speculative.cpp common/speculative.h \
  src/llama-graph.cpp src/llama-context.cpp tools/server/server.cpp 2>/dev/null
echo
echo "### (B) CUDA kernels that can shift logits (FA / softmax / argsort / topk / moe) ###"
git log --oneline $RANGE -- \
  'ggml/src/ggml-cuda/fattn*' 'ggml/src/ggml-cuda/*softmax*' 'ggml/src/ggml-cuda/*argsort*' \
  'ggml/src/ggml-cuda/*topk*' 'ggml/src/ggml-cuda/*mmid*' 'ggml/src/ggml-cuda/*moe*' 2>/dev/null
echo
echo "### (C) subjects matching decode-exactness keywords, IN RANGE ###"
git log --oneline $RANGE | grep -iE "mtp|nextn|spec|draft|sampl|greedy|argmax|argsort|topk|softmax|flash.?att|fattn|deterministic| tie|rope|rms" 2>/dev/null
echo
echo "### (D) prime suspects — show the diffstat of any MTP/nextn/spec commit in range ###"
for h in $(git log --format=%h $RANGE | head -200); do
  s=$(git log -1 --format=%s "$h")
  case "$s" in
    *mtp*|*MTP*|*nextn*|*NextN*|*spec:*|*speculat*|*draft-mtp*)
      echo "--- $h  $s"
      git show --stat --format="" "$h" | grep -E "\.(cpp|cu|h|cuh)" | head -8 ;;
  esac
done
