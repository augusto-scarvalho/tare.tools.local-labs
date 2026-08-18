#!/usr/bin/env bash
# Inventory the llama.cpp build trees: path, git HEAD, local diff, disk size.
set -u
cd /home/augus/src || exit 1
for d in llama.cpp-base llama.cpp llama.cpp-master llama.cpp-rebase llama.cpp-stack llama.cpp-local ik_llama.cpp; do
  [ -d "$d" ] || { printf "%-22s MISSING\n" "$d"; continue; }
  head=$(git -C "$d" log --oneline -1 2>/dev/null | cut -c1-50)
  br=$(git -C "$d" rev-parse --abbrev-ref HEAD 2>/dev/null)
  diff=$(git -C "$d" diff --shortstat 2>/dev/null)
  [ -z "$diff" ] && diff="(clean)"
  bin="no-bin"; [ -x "$d/build/bin/llama-server" ] && bin="HAS-server"
  sz=$(du -sh "$d" 2>/dev/null | cut -f1)
  bsz=$(du -sh "$d/build" 2>/dev/null | cut -f1)
  printf "%-20s %-6s %-9s HEAD=%-42s br=%-10s diff=%s\n" "$d" "$sz" "$bin" "$head" "$br" "$diff"
  printf "%-20s   build=%s\n" "" "$bsz"
done
