#!/usr/bin/env bash
# Finalize the consolidated fork on the VALIDATED base: 720d7fa40 + §B2b (MTP-exact).
# Fresh master f5919bf45 breaks draft-mtp exactness (regression not #25832; unbisected),
# so we drop that worktree and formalize the patch on llama.cpp-master's 720d7fa40.
set -u
SRC=/home/augus/src/slop.cpp-main
WT=/home/augus/src/slop.cpp-fork

cd "$SRC"
echo "=== drop the broken fresh-master worktree + its lifecycle branch ==="
git worktree remove --force "$WT" 2>&1 || echo "(worktree already gone)"
git branch -D lifecycle 2>&1 || echo "(branch already gone)"
git worktree prune

echo "=== master state (should be detached 720d7fa40 + §B2b working-tree diff) ==="
git log --oneline -1
git diff --stat -- src/llama-kv-cache.cpp

echo "=== create the lifecycle branch here and commit §B2b ==="
git checkout -b lifecycle 2>&1
git add src/llama-kv-cache.cpp
git -c user.name=local-model-lifecycle -c user.email=augustosc4rvalho@gmail.com \
    commit -q -m "B2b: pin the KV host buffer under --no-kv-offload (env GGML_KV_PIN_HOST)

The consolidated fork's one non-upstream lever, on the VALIDATED base 720d7fa40 (where
draft-mtp is token-exact; fresh master f5919bf45 regressed that). With KV in system RAM the
per-token host->GPU copy is bounce-buffered out of a pageable ggml_backend_cpu_buffer_type();
this env-gated branch swaps it for the device host (cudaHostRegister'd) buffer -> direct DMA.
Inert without the env var, so default behaviour == upstream 720d7fa40. Recovers up to +17%
decode in the --no-kv-offload / long-context regime (local-model-lifecycle STATUS B2)."
echo "=== fork branch head ==="
git log --oneline -2
echo "=== the built binary is already 720d7fa40+§B2b (MASTER_BIN); it IS the fork ==="
ls -la --time-style=+%Y-%m-%d build/bin/llama-server
echo "=== disk freed (fork worktree removed) ==="
du -sh "$SRC" 2>/dev/null | cut -f1
