#!/usr/bin/env bash
# The turbo<->expert-cache conflict is an ordering artifact: in the stack, turbo is BELOW
# expert-cache, so re-apply in that order. Reset to after prefetch, cherry-pick turbo, then
# expert-cache on top -- matching the stack's authored order (minimal conflicts).
set -u
cd /home/augus/src/llama.cpp-master
git cherry-pick --abort 2>/dev/null
git rev-parse --abbrev-ref HEAD | grep -q lifecycle || { echo "REFUSING: not on lifecycle"; exit 2; }
git fetch -q stacksrc 2>/dev/null

echo "== reset to prefetch commit (3de272efc): keeps 720d7fa40 + §B2b + prefetch =="
git reset --hard 3de272efc
git log --oneline -5

echo; echo "== 1) cherry-pick TURBO range 170322f8d^..40f20368d (on top of prefetch, matching stack) =="
if git cherry-pick -x 170322f8d^..40f20368d > /tmp/cp_turbo2.log 2>&1; then
  echo "TURBO CLEAN"
else
  echo "TURBO STOPPED at $(git log -1 --format=%h CHERRY_PICK_HEAD 2>/dev/null): conflicts:"
  git diff --name-only --diff-filter=U | sed 's/^/    /'
  exit 3
fi

echo; echo "== 2) cherry-pick EXPERT-CACHE range 3d3efff79^..cca05a3ac (on top of turbo, matching stack) =="
if git cherry-pick -x 3d3efff79^..cca05a3ac > /tmp/cp_cache2.log 2>&1; then
  echo "EXPERT-CACHE CLEAN"
else
  echo "CACHE STOPPED at $(git log -1 --format=%h CHERRY_PICK_HEAD 2>/dev/null): conflicts:"
  git diff --name-only --diff-filter=U | sed 's/^/    /'
  exit 4
fi

echo; echo "== lifecycle assembled =="; git log --oneline | head -12
echo "total commits over 720d7fa40: $(git rev-list --count 720d7fa40..HEAD)"
