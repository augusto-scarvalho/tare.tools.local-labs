#!/usr/bin/env bash
# Anthology lever 2/3: port the MoE expert cache onto lifecycle. ~22 commits from the stack
# (moe-trace tool + id=-1 skip infra in mul_mat_id + cache loader/wiring + --moe-cache-* flags).
# Flag-gated: no --moe-cache-slots => cache disabled => must be byte-identical to 720d7fa40
# (the G2 bless verifies). Stack base is only 15 commits behind 720d7fa40 and those 15 touch
# only UI/httplib -> no overlap with these files, so the range should apply cleanly.
set -u
cd /home/augus/src/slop.cpp-main
git rev-parse --abbrev-ref HEAD | grep -q lifecycle || { echo "REFUSING: not on lifecycle"; exit 2; }

echo "== wire up stack repo as remote + fetch =="
git remote get-url stacksrc >/dev/null 2>&1 || git remote add stacksrc /home/augus/src/slop.cpp-stack
git fetch -q stacksrc

echo "== cherry-pick the expert-cache range 3d3efff79^..cca05a3ac (linear, no merges) =="
if git cherry-pick -x 3d3efff79^..cca05a3ac > /tmp/cp_cache.log 2>&1; then
  echo "RANGE CLEAN"
else
  echo "STOPPED (conflict). Status:"
  git status --short | head -20
  echo "--- conflicting files ---"
  git diff --name-only --diff-filter=U
  echo "(resolve, then git cherry-pick --continue; or --abort)"
  exit 3
fi
echo "== lifecycle head =="; git log --oneline -6

echo "== incremental rebuild (touches CUDA mmid/mmq/mmvf/mmvq + graph) =="
cmake --build build --target llama-server -j 20 > /tmp/port_cache_build.log 2>&1
echo "build exit=$?"; tail -3 /tmp/port_cache_build.log
ls -la --time-style=+%H:%M:%S build/bin/llama-server
echo "== also build llama-moe-trace (the profiler) =="
cmake --build build --target llama-moe-trace -j 20 > /tmp/port_trace_build.log 2>&1
echo "trace build exit=$?"; ls -la build/bin/llama-moe-trace 2>/dev/null
