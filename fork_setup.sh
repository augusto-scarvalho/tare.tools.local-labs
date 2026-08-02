#!/usr/bin/env bash
# Establish the consolidated fork: fresh upstream master + §B2b, as a git worktree off the
# llama.cpp-master repo (shares the object store; keeps 720d7fa40/MASTER_BIN intact).
set -eu
SRC=/home/augus/src/llama.cpp-master
WT=/home/augus/src/llama.cpp-fork
KV=src/llama-kv-cache.cpp

cd "$SRC"
# fresh upstream head we fetched
UP=$(git rev-parse upstream/master)
echo "fork base = upstream/master $UP"

# create the worktree on a new 'lifecycle' branch, if not already present
if git worktree list | grep -q "$WT"; then
  echo "worktree already exists: $WT"
else
  git worktree add -b lifecycle "$WT" upstream/master
fi

cd "$WT"
echo "worktree HEAD: $(git log --oneline -1)"

# re-apply §B2b by exact string match (line drift on fresh master is irrelevant this way)
python3 - "$WT/$KV" <<'PY'
import sys, pathlib
F = pathlib.Path(sys.argv[1]); src = F.read_text()
if "GGML_KV_PIN_HOST" in src:
    print("already patched"); sys.exit(0)
OLD = """        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }
"""
NEW = """        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        } else if (getenv("GGML_KV_PIN_HOST")) {
            // [B2b] KV in system RAM (--no-kv-offload): the per-token host->GPU KV copy is
            // bounce-buffered out of a pageable CPU buffer. Pin it (device host buffer =
            // cudaHostRegister'd) so the copy is a direct DMA. Env-gated so both A/B arms are
            // one binary, differing only by this var; falls back to plain CPU when the layer's
            // device exposes no host buffer type (e.g. CPU-resident layers).
            auto * dev = model.dev_layer(il);
            ggml_backend_buffer_type_t hbuft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
            if (hbuft) {
                buft = hbuft;
                dev_name = "CUDA_Host(B2b)";
            }
        }
"""
if OLD not in src:
    print("ANCHOR NOT FOUND — upstream changed the block; manual apply needed"); sys.exit(1)
F.write_text(src.replace(OLD, NEW, 1)); print("patched OK")
PY

git add "$KV"
git -c user.name=local-model-lifecycle -c user.email=augustosc4rvalho@gmail.com \
    commit -q -m "B2b: pin the KV host buffer under --no-kv-offload (env GGML_KV_PIN_HOST)

The consolidated-fork's one non-upstream lever. With KV in system RAM the per-token
host->GPU copy is bounce-buffered out of a pageable ggml_backend_cpu_buffer_type(); this
env-gated branch swaps it for the device host (cudaHostRegister'd) buffer -> direct DMA.
Inert without the env var, so default behaviour is identical to upstream. Recovers up to
+17% decode in the --no-kv-offload / long-context regime (local-model-lifecycle STATUS B2)."
echo "=== fork commit ==="
git log --oneline -2

# configure the build, matching MASTER_BIN's flags (FA_ALL_QUANTS stays OFF: q8_0 FA is in
# the default build, proven by the whole campaign running FA-on-GPU at that setting).
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON -DGGML_NATIVE=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  > /tmp/fork_cmake.log 2>&1 && echo "cmake configured OK" || { echo "cmake FAILED:"; tail -15 /tmp/fork_cmake.log; exit 1; }
