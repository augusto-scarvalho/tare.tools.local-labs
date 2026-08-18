#!/usr/bin/env bash
# Anthology lever 3/3: port the full TurboQuant KV block + turbo-MMA FA decode onto lifecycle.
# 4 commits (170322f8d TurboQuant-all-backends, 933820717 Q2_0, 4cd54c684 mixed-KV FA select,
# 40f20368d turbo MMA decode). Turbo's GGML_TURBO_MMA_FUSED gate is DEFAULT-ON in the stack;
# we flip it default-OFF after, so the pristine q8_0 FA path is unchanged (G2 verifies).
# These sit BELOW expert-cache in the stack, so applying on top may conflict (llama-graph.cpp,
# ggml.c) -- report the surface rather than fail silently.
set -u
cd /home/augus/src/llama.cpp-master
git rev-parse --abbrev-ref HEAD | grep -q lifecycle || { echo "REFUSING: not on lifecycle"; exit 2; }
git fetch -q stacksrc 2>/dev/null

echo "== cherry-pick turbo range 170322f8d^..40f20368d =="
if git cherry-pick -x 170322f8d^..40f20368d > /tmp/cp_turbo.log 2>&1; then
  echo "RANGE CLEAN"
else
  echo "STOPPED at: $(git log -1 --format=%h%n%s CHERRY_PICK_HEAD 2>/dev/null | tr '\n' ' ')"
  echo "--- conflicting files ---"
  git diff --name-only --diff-filter=U
  echo "--- conflict hunk counts ---"
  for f in $(git diff --name-only --diff-filter=U); do
    echo "  $f : $(grep -c '^<<<<<<<' "$f" 2>/dev/null) hunks"
  done
  echo "(assess: resolve + git cherry-pick --continue, or git cherry-pick --abort)"
  exit 3
fi
git log --oneline -6
