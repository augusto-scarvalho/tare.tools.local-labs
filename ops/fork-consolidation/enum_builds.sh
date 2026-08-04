#!/usr/bin/env bash
# Enumerate all llama.cpp-* build trees: branch, HEAD, bins, sizes, git status.
for d in /home/augus/src/llama.cpp-*; do
  echo "===================== $d ====================="
  echo -n "branch: "; git -C "$d" branch --show-current 2>/dev/null
  echo -n "HEAD:   "; git -C "$d" log --oneline -1 2>/dev/null
  echo -n "dirty:  "; git -C "$d" status --porcelain 2>/dev/null | wc -l;
  echo    "bins:   $(ls "$d"/build/bin/ 2>/dev/null | tr '\n' ' ')"
  echo    "build:  $(du -sh "$d"/build 2>/dev/null | cut -f1)   tree: $(du -sh "$d" 2>/dev/null | cut -f1)"
  # how far from upstream master's merge-base, to see divergence
  echo -n "ahead of pinned 720d7fa40: "; git -C "$d" rev-list --count 720d7fa40..HEAD 2>/dev/null || echo "n/a"
done
echo "===================== TOTAL disk ====================="
du -sh /home/augus/src/llama.cpp-* 2>/dev/null
echo "combined: $(du -sch /home/augus/src/llama.cpp-* 2>/dev/null | tail -1 | cut -f1)"
