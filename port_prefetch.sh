#!/usr/bin/env bash
# Anthology lever 1/3: port Fable prefetch+pinning onto the lifecycle fork (720d7fa40 + §B2b).
# Base of these commits IS exactly 720d7fa40, so cherry-pick is conflict-free. Runtime-gated:
#   GGML_SCHED_PREFETCH_EXPERTS>0  -> overlap expert H2D on a 2nd stream (off by default)
#   GGML_CUDA_REGISTER_HOST=1      -> pin mmap CPU weights (off by default)
# Numerics-neutral (memory transfer, not compute) -> cannot change decode tokens.
set -eu
cd /home/augus/src/llama.cpp-master
git rev-parse --abbrev-ref HEAD | grep -q lifecycle || { echo "REFUSING: not on lifecycle branch"; exit 2; }

echo "== wire up source repos as remotes (idempotent) =="
git remote get-url rebasesrc >/dev/null 2>&1 || git remote add rebasesrc /home/augus/src/llama.cpp-rebase
git fetch -q rebasesrc

echo "== cherry-pick the 3 prefetch/pin commits (in order) =="
for h in 514c420bf e5c411f2d 0288a1b46; do
  echo "  cherry-pick $h : $(git log -1 --format=%s $h 2>/dev/null | cut -c1-60)"
  git cherry-pick -x "$h" || { echo "CONFLICT on $h"; git status --short; exit 1; }
done
echo "== lifecycle head now =="
git log --oneline -5

echo "== incremental rebuild =="
cmake --build build --target llama-server -j 20 > /tmp/port_pf_build.log 2>&1
echo "build exit=$?"; tail -2 /tmp/port_pf_build.log
ls -la --time-style=+%H:%M:%S build/bin/llama-server
