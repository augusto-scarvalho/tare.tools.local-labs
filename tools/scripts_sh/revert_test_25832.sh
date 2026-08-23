#!/usr/bin/env bash
# Prove/deny that #25832 is the draft-mtp exactness regression: revert it in the fresh-master
# fork worktree, incremental-rebuild, and let the caller re-run the G2 identity gate.
set -u
WT=/home/augus/src/slop.cpp-fork
cd "$WT" || exit 2

echo "== current HEAD =="; git log --oneline -2
echo "== attempting clean revert of dee2a846b (#25832) =="
if git revert --no-edit dee2a846b > /tmp/revert.log 2>&1; then
  echo "clean revert OK"
else
  echo "clean revert conflicted; doing SURGICAL revert of the FLASH_ATTN_EXT skip only"
  git revert --abort 2>/dev/null
  # remove just the line that newly skips FA_EXT from weight-backend offload (the hypothesis)
  python3 - <<'PY'
import pathlib
F=pathlib.Path("/home/augus/src/slop.cpp-fork/ggml/src/ggml-backend.cpp")
s=F.read_text()
needle='    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it\n    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;\n\n'
if needle in s:
    F.write_text(s.replace(needle,"",1)); print("surgical: removed FA_EXT skip line")
else:
    # tolerate minor comment-typo drift: match just the code line
    import re
    s2=re.sub(r'\n[ \t]*allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;','',s,count=1)
    if s2!=s: F.write_text(s2); print("surgical (regex): removed FA_EXT skip line")
    else: print("SURGICAL FAILED — line not found"); exit(1)
PY
  git add ggml/src/ggml-backend.cpp
  git -c user.name=test -c user.email=t@t commit -q -m "TEST: revert #25832 FLASH_ATTN_EXT offload skip (regression probe)"
fi
echo "== HEAD after revert =="; git log --oneline -2
echo "== incremental rebuild llama-server =="
cmake --build build --target llama-server -j 20 > /tmp/revert_build.log 2>&1
echo "build exit=$?"; tail -2 /tmp/revert_build.log
ls -la build/bin/llama-server
