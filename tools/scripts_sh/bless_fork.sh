#!/usr/bin/env bash
# Compatibility entrypoint. The canonical qualification harness belongs to slop.cpp.
set -euo pipefail

SLOP_ROOT=${SLOP_ROOT:-/home/augus/src/slop.cpp}
SLOP_BLESS_SCRIPT=${SLOP_BLESS_SCRIPT:-"$SLOP_ROOT/tools/scripts_sh/bless_fork.sh"}

if [[ ! -f "$SLOP_BLESS_SCRIPT" ]]; then
    echo "REFUSING: canonical slop.cpp harness not found: $SLOP_BLESS_SCRIPT" >&2
    echo "Set SLOP_ROOT or SLOP_BLESS_SCRIPT to the current slop.cpp checkout." >&2
    exit 2
fi

exec bash "$SLOP_BLESS_SCRIPT" "$@"
