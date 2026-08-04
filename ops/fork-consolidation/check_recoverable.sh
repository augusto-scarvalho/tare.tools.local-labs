#!/usr/bin/env bash
# For each build tree: remotes, and whether HEAD is recoverable (pushed to any remote).
for d in /home/augus/src/llama.cpp-*; do
  echo "===================== $d ====================="
  echo "remotes:"; git -C "$d" remote -v 2>/dev/null | sed 's/^/  /'
  head=$(git -C "$d" rev-parse HEAD 2>/dev/null)
  echo "  HEAD=$head"
  # remote branches containing HEAD => pushed/recoverable
  rc=$(git -C "$d" branch -r --contains HEAD 2>/dev/null | head -5)
  if [ -n "$rc" ]; then echo "  RECOVERABLE — remote refs contain HEAD:"; echo "$rc" | sed 's/^/    /';
  else echo "  *** NOT on any remote branch (HEAD unpushed) ***"; fi
  # also: any local commits not on a remote at all (unique work that would be lost)?
  uniq=$(git -C "$d" log --oneline --branches --not --remotes 2>/dev/null | wc -l)
  echo "  local commits not on any remote: $uniq"
done
