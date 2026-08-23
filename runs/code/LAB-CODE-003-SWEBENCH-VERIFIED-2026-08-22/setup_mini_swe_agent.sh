#!/usr/bin/env bash
set -euo pipefail

repo=/home/augus/src/mini-swe-agent-lab-code-003
commit=25941c89cfbc91eb40b3f8756348c91d9977d57e

if [[ ! -d "$repo/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/SWE-agent/mini-swe-agent.git "$repo"
fi
git -C "$repo" fetch --depth 1 origin "$commit"
git -C "$repo" checkout --detach "$commit"
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short

venv=/home/augus/mini-swe-agent-venv
if [[ ! -x "$venv/bin/python" ]]; then
  python3 -m venv "$venv"
fi
"$venv/bin/python" -m pip install --upgrade pip
"$venv/bin/python" -m pip install "$repo"
"$venv/bin/python" -c 'import minisweagent; print(minisweagent.__version__)'
