#!/usr/bin/env bash
# Inventory the exact fork-specific commits to port onto the lifecycle fork (720d7fa40 + §B2b),
# and gauge conflict risk. Each lever must end up runtime-gated, default == pristine 720d7fa40.
set -u
echo "############## FABLE prefetch/pinning (llama.cpp-rebase = Fable on a master) ##############"
cd /home/augus/src/slop.cpp-rebase 2>/dev/null && {
  echo "branch: $(git rev-parse --abbrev-ref HEAD)   HEAD: $(git log --oneline -1)"
  MB=$(git merge-base HEAD upstream/master 2>/dev/null || git merge-base HEAD origin/master 2>/dev/null)
  echo "merge-base with upstream: $MB"
  echo "-- fork-specific commits (merge-base..HEAD) --"
  git log --oneline "$MB..HEAD" 2>/dev/null | head -20
  echo "-- files they touch --"
  git diff --stat "$MB..HEAD" 2>/dev/null | tail -20
}
echo
echo "############## STACK: expert-cache + turbo-mma + others (llama.cpp-stack) ##############"
cd /home/augus/src/slop.cpp-stack 2>/dev/null && {
  echo "branch: $(git rev-parse --abbrev-ref HEAD)   HEAD: $(git log --oneline -1)"
  MB=$(git merge-base HEAD upstream/master 2>/dev/null)
  echo "merge-base with upstream: $MB  ($(git log -1 --format=%ci $MB 2>/dev/null))"
  echo "-- ALL fork-specific commits (merge-base..HEAD) --"
  git log --oneline "$MB..HEAD" 2>/dev/null
  echo
  echo "-- expert-cache commits --"
  git log --oneline "$MB..HEAD" 2>/dev/null | grep -iE "moe.?cache|moe.?trace|expert cache|hot/cold|mul_mat_id"
  echo "-- turbo-mma commits --"
  git log --oneline "$MB..HEAD" 2>/dev/null | grep -iE "turbo|mma.?fused|GGML_TURBO"
}
echo
echo "############## fork base vs 720d7fa40: how far apart? ##############"
cd /home/augus/src/slop.cpp-main && for r in llama.cpp-rebase llama.cpp-stack; do
  b=$(git -C /home/augus/src/$r merge-base HEAD upstream/master 2>/dev/null)
  echo "$r base $b : 720d7fa40 is $(git rev-list --count $b..720d7fa40 2>/dev/null) ahead / $(git rev-list --count 720d7fa40..$b 2>/dev/null) behind"
done
