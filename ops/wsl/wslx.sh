#!/usr/bin/env bash
# wslx.sh — run a bash script INSIDE WSL reliably from the Windows Git Bash / Claude Bash tool.
#
# WHY (the two gotchas this tool exists to kill — see ops/wsl/README.md):
#   1. Passing $VARs inline through `wsl -d X -- bash -lc '...'` from the Windows side lets an OUTER
#      layer expand them to EMPTY *silently* (even inside single quotes), so redirects hit `/logs/...`
#      at root and die with `No such file or directory`. The only robust fix is to keep every `$` in a
#      FILE on disk (read by WSL bash directly), never in the inline tool command. This wrapper does that.
#   2. `huggingface-cli` was removed (huggingface_hub >=1.24) — the CLI is now `hf` (~/.local/bin/hf).
#
# THE RULE this encodes: your invocation of wslx.sh contains ZERO `$` (only literal path args), so no
# outer layer can mangle anything. All variable logic lives in the staged script file.
#
# Usage:
#   bash ops/wsl/wslx.sh <script.sh> [--detached] [--log <wsl_log>] [--distro <name>] [-- ARGS...]
#     <script.sh>  bash script to run; Windows (C:\..), /mnt/c/.., or WSL (/home/..) path. Its $VARs are safe.
#     --detached   run under setsid and return immediately (long jobs: downloads, training).
#     --log        WSL path for combined stdout+stderr (default ~/.cache/wslx/<name>.log).
#     --distro     WSL distro (default Ubuntu-24.04).
#     -- ARGS...   passed to the script as "$@".
#   Runs under a LOGIN shell so ~/.local/bin (hf, etc.) is on PATH.
#
# Examples:
#   bash ops/wsl/wslx.sh ops/qwen38-bringup/mtp_tensor_check.sh          # run a repo gate in WSL
#   bash ops/wsl/wslx.sh /c/tmp/big_download.sh --detached               # long job, returns at once
set -euo pipefail

DISTRO="Ubuntu-24.04"; DETACHED=0; LOG=""; SRC=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --detached) DETACHED=1; shift;;
    --log)      LOG="$2"; shift 2;;
    --distro)   DISTRO="$2"; shift 2;;
    --)         shift; ARGS=("$@"); break;;
    -*)         echo "wslx: unknown flag $1" >&2; exit 2;;
    *)          SRC="$1"; shift;;
  esac
done
[ -n "$SRC" ] || { echo "usage: wslx.sh <script.sh> [--detached] [--log path] [--distro name] [-- args]" >&2; exit 2; }

# --- translate a Windows / Git-Bash path to a WSL /mnt path; leave real WSL paths as-is ---
to_wsl_path() {
  local p="$1"
  case "$p" in
    /mnt/*|/home/*|/root/*|/tmp/*) printf '%s' "$p" ;;                       # already WSL
    [A-Za-z]:*)                                                              # C:\.. or C:/..
      local d="${p:0:1}" rest="${p:2}"; rest="${rest//\\//}"
      printf '/mnt/%s%s' "$(printf '%s' "$d" | tr 'A-Z' 'a-z')" "$rest" ;;
    /[a-zA-Z]/*)                                                            # Git Bash /c/..
      printf '/mnt%s' "$p" ;;
    *)                                                                       # relative -> resolve, then map
      local abs; abs="$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"
      case "$abs" in /[a-zA-Z]/*) printf '/mnt%s' "$abs" ;; *) printf '%s' "$abs" ;; esac ;;
  esac
}

SRC_WSL="$(to_wsl_path "$SRC")"
NAME="$(basename "$SRC")"
DEST="\$HOME/.cache/wslx/$NAME"                       # $HOME expands INSIDE wsl
[ -n "$LOG" ] || LOG="\$HOME/.cache/wslx/${NAME%.sh}.log"
# Split passthrough into leading NAME=VALUE env assignments vs positional args.
# `-- MODEL=/x foo` -> runs `env MODEL=/x bash <script> foo` (so gates reading ${MODEL:-..} work).
ENVSTR=""; ARGSTR=""; seen_pos=0
for a in "${ARGS[@]:-}"; do
  if [ "$seen_pos" = 0 ] && [[ "$a" == [A-Za-z_]*=* ]]; then
    ENVSTR="$ENVSTR $(printf '%q' "$a")"
  else
    seen_pos=1; ARGSTR="$ARGSTR $(printf '%q' "$a")"
  fi
done
[ -n "$ENVSTR" ] && ENVSTR="env$ENVSTR "

# stage: copy into WSL stripping CR, make executable
wsl -d "$DISTRO" -- bash -lc "mkdir -p \$HOME/.cache/wslx && tr -d '\r' < '$SRC_WSL' > $DEST && chmod +x $DEST && echo \"wslx: staged $DEST\""

if [ "$DETACHED" = 1 ]; then
  # $!/$? don't survive the Git Bash -> wsl.exe boundary, so we don't rely on them.
  # disown + a short settle sleep let setsid fully reparent BEFORE this transient wsl exits — without
  # it there is a race that can kill the job instantly (observed: empty log, no process).
  # For LONG jobs prefer the harness run_in_background with a literal-path wsl command (keeps wsl.exe
  # alive the whole run + notifies on completion) — see ops/wsl/README.md "Long detached jobs".
  wsl -d "$DISTRO" -- bash -lc "setsid bash -lc '${ENVSTR}bash $DEST$ARGSTR' </dev/null > $LOG 2>&1 & disown; sleep 3; echo 'wslx: detached'"
  echo "wslx: log  -> $LOG   (in $DISTRO)"
  echo "wslx: tail -> wsl -d $DISTRO -- bash -lc 'tail -f $LOG'"
  echo "wslx: done -> wsl -d $DISTRO -- bash -lc 'grep -q \"=== DONE\" $LOG && echo FINISHED'   (add a DONE marker in your script)"
else
  # Stream FULL output live while teeing to the log (tail -n truncates verbose gates). pipefail so a
  # failing script still surfaces; tee's exit is 0 so we don't rely on $? across the boundary.
  wsl -d "$DISTRO" -- bash -lc "set -o pipefail; bash -lc '${ENVSTR}bash $DEST$ARGSTR' 2>&1 | tee $LOG"
fi
