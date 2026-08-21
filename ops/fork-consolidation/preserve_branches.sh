#!/usr/bin/env bash
# Phase 1 — non-destructive preservation. Give the fragile detached Turbo HEAD a real branch,
# then create local branches in the fork repo for every stranded line. Nothing deleted.
set -u
STACK=/home/augus/src/slop.cpp-stack
FORK=/home/augus/src/slop.cpp-main

echo "=== 1. branch the detached Turbo HEAD in the stack tree ==="
if git -C "$STACK" show-ref --verify -q refs/heads/turbo-stack; then
  echo "  turbo-stack already exists in stack tree"
else
  git -C "$STACK" branch turbo-stack HEAD && echo "  created turbo-stack -> $(git -C "$STACK" rev-parse --short turbo-stack)"
fi

echo; echo "=== 2. fork: fetch siblings so the new branch + refs are visible ==="
git -C "$FORK" fetch stacksrc --no-tags 2>&1 | sed 's/^/  /'
git -C "$FORK" fetch rebasesrc --no-tags 2>&1 | sed 's/^/  /'

echo; echo "=== 3. create local preservation branches in the fork (idempotent) ==="
create() { # name  start-ref
  if git -C "$FORK" show-ref --verify -q "refs/heads/$1"; then
    echo "  $1 exists ($(git -C "$FORK" rev-parse --short $1))"
  else
    git -C "$FORK" branch "$1" "$2" && echo "  $1 -> $(git -C "$FORK" rev-parse --short $1)  (from $2)"
  fi
}
create turbo-stack            stacksrc/turbo-stack
create prefetch-skip-pinned   rebasesrc/local/prefetch-skip-pinned
create fable5-prefetch-experts origin/fable5/prefetch-experts

echo; echo "=== 4. resulting local branches in the fork ==="
git -C "$FORK" for-each-ref --format='  %(refname:short)  %(objectname:short)  %(contents:subject)' refs/heads
echo; echo "  current checkout: $(git -C "$FORK" rev-parse --abbrev-ref HEAD) @ $(git -C "$FORK" rev-parse --short HEAD)"
