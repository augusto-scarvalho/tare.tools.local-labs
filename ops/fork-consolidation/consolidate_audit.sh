#!/usr/bin/env bash
# From the lifecycle fork repo (llama.cpp-master), fetch every sibling line and audit what
# unique work each holds that is NOT already in lifecycle. NON-DESTRUCTIVE (fetch + read only).
set -u
M=/home/augus/src/slop.cpp-main
cd "$M" || exit 1

echo "=== current remotes ==="; git remote -v | awk '{print $1, $2}' | sort -u

echo; echo "=== fetching all siblings (read-only) ==="
git fetch --all --no-tags 2>&1 | sed 's/^/  /'

echo; echo "=== every ref now visible, grouped ==="
git for-each-ref --format='%(refname:short)  ->  %(objectname:short)  %(contents:subject)' \
  refs/remotes refs/heads | sed 's/^/  /'

echo; echo "=== per interesting branch: commits NOT in lifecycle (cherry '+') ==="
for ref in \
  origin/fable5/prefetch-experts \
  rebasesrc/fable5/prefetch-experts-rebased \
  rebasesrc/local/prefetch-skip-pinned \
  rebasesrc/lifecycle \
  stacksrc/local/prefetch-skip-pinned \
  stacksrc/fable5/prefetch-experts-rebased ; do
  if git rev-parse --verify -q "$ref" >/dev/null; then
    n=$(git rev-list --count "lifecycle..$ref" 2>/dev/null)
    echo "--- $ref : $n commits ahead of lifecycle ---"
    # cherry: '+' = patch NOT present in lifecycle, '-' = already there (equivalent patch)
    git cherry lifecycle "$ref" 2>/dev/null | awk '{print ($1=="+"?"  NOT-IN-LIFECYCLE ":"  already ")}' | sort | uniq -c
    # show the actual unique (+) subjects, capped
    git cherry -v lifecycle "$ref" 2>/dev/null | grep '^+' | head -12 | sed 's/^/    /'
  else
    echo "--- $ref : (ref not found) ---"
  fi
done

echo; echo "=== stack tree HEAD (Turbo, detached, its own repo) — compare to lifecycle via stacksrc ==="
echo "stacksrc branches:"; git branch -r 2>/dev/null | grep stacksrc | sed 's/^/  /'
