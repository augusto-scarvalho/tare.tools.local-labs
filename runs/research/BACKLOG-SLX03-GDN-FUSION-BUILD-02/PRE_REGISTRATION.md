# BACKLOG-SLX03-GDN-FUSION-BUILD-02 preregistration

Task: Build and callable-binary qualification of SLX-03 GDN cache fusion with explicit CUDA compiler discovery
Evidence class: `external_reproduction`

## Hypothesis

R1 failed before compilation because `/usr/local/cuda/bin` was absent from the WSL command environment. With the installed and independently callable `/usr/local/cuda/bin/nvcc` explicitly selected, immutable source commit `87a416bd` builds a self-contained CUDA `llama-server` containing the audited GDN snapshot-copy fusion path.

## Frozen inputs

- `runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-03/raw/receipt.json`
- `runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-03/REVIEW.json`
- `runs/autonomous/EXPERIMENT-WATCH-2026-08-28-SLX03-BUILD-R1/FINAL.json`
- `runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-01/raw/configure.stderr.log`

The runner freezes the promoted source materialization receipt/review and this admission/preregistration by SHA-256. R1 watcher output and the raw CMake error are retained as causal evidence. The only intentional execution change is an explicit `PATH` plus `CUDACXX=/usr/local/cuda/bin/nvcc` for source commands.

## Command

```powershell
python tools/research/run_slx03_gdn_fusion_build_r2.py
```

## Factors

One fresh build directory `/home/augus/src/slop.cpp-main/build-slx03-gdn-audit-02`, immutable source commit, Release CUDA build for SM86, target `llama-server`, and eight workers. Gateway/service/embedding identities are captured before and after; no candidate model is loaded.

## Acceptance gates

- `source_revision`: `exact_source_commit eq True`
- `tracked_clean`: `tracked_source_clean eq True`
- `configure`: `cmake_configure_exit eq 0`
- `build`: `llama_server_build_exit eq 0`
- `fusion_marker`: `gdn_fusion_marker_present eq True`
- `self_linkage`: `project_libraries_resolve_to_new_build eq True`
- `callability`: `server_version_exit eq 0`
- `service_invariance`: `gateway_and_embedding_unchanged eq True`

## Abort conditions

Abort if the fresh build directory exists or any source identity, build, linkage, callability, service, or provenance gate fails. Preserve all failures. This is build-only evidence; runtime fusion, semantic parity, write reduction, speedup, and production deployment remain forbidden claims.

## Allowed claims

- `SLX03_GDN_FUSION_BUILD_CALLABLE_R2`
- `SLX03_GDN_FUSION_BUILD_NOT_CONFIRMED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
