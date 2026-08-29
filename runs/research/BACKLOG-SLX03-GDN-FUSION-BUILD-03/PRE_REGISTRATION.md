# BACKLOG-SLX03-GDN-FUSION-BUILD-03 preregistration

Task: Build and callable-binary qualification of SLX-03 GDN cache fusion (harness-corrected)
Evidence class: `external_reproduction`

## Hypothesis

With the installed `/usr/local/cuda/bin/nvcc` explicitly selected, immutable source commit `87a416bd` builds a self-contained CUDA `llama-server` containing the audited GDN snapshot-copy fusion path. R1 and R2 are harness/environment failures, not evidence about this hypothesis.

## Frozen inputs

- `runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-03/raw/receipt.json`
- `runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-03/REVIEW.json`
- `runs/autonomous/EXPERIMENT-WATCH-2026-08-28-SLX03-BUILD-R1/FINAL.json`
- `runs/autonomous/EXPERIMENT-WATCH-2026-08-28-SLX03-BUILD-R2/FINAL.json`

The runner freezes the promoted source-materialization receipt/review by SHA-256. R1 and R2 watcher finals remain immutable causal evidence. The fresh build directory is `/home/augus/src/slop.cpp-main/build-slx03-gdn-audit-03`; `PATH` and `CUDACXX` are explicit.

## Command

```powershell
python tools/research/run_slx03_gdn_fusion_build_r3.py
```

## Factors

One immutable commit, one fresh Release CUDA SM86 build, target `llama-server`, eight workers. The wrapper import and `nvcc --version` are preflighted before state advancement. Gateway/service/embedding identities are captured before and after; no candidate model is loaded.

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

Abort if preflight fails, the fresh directory exists, or any identity, build, linkage, callability, service, or provenance gate fails. Preserve all failures. Build-only evidence cannot support runtime fusion, parity, write reduction, speedup, or deployment claims.

## Allowed claims

- `SLX03_GDN_FUSION_BUILD_CALLABLE_R3`
- `SLX03_GDN_FUSION_BUILD_NOT_CONFIRMED_R3`

Claims outside these codes are forbidden even if a metric looks favorable.
