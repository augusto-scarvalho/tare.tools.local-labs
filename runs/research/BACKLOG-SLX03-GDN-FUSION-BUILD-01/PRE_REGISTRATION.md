# BACKLOG-SLX03-GDN-FUSION-BUILD-01 preregistration

Task: Build and callable-binary qualification of SLX-03 GDN cache fusion
Evidence class: `external_reproduction`

## Hypothesis

Immutable WSL source commit 87a416bd builds a self-contained CUDA llama-server whose CUDA library contains the audited GDN snapshot-copy fusion path and whose binary is callable without resolving project libraries from another checkout.

## Frozen inputs

- `runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-03/raw/receipt.json`
- `runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-03/REVIEW.json`

The promoted source-materialization receipt/review and this admission/preregistration are frozen by SHA-256 in `tools/research/run_slx03_gdn_fusion_build.py`. The runner independently verifies WSL commit `87a416bd75d5a64e66e55846b779c0a54eca21bd`.

## Command

```powershell
python tools/research/run_slx03_gdn_fusion_build.py
```

## Factors

One fresh build directory `/home/augus/src/slop.cpp-main/build-slx03-gdn-audit-01`, Release CUDA build for RTX 3090 (SM86), target `llama-server`, eight build workers. Read-only source revision checks precede configuration. Gateway/service/embedding identities are captured before and after; no model is loaded by the candidate.

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

Abort if the exact build directory already exists, tracked source is dirty, commit differs, configuration/build/version fails, project libraries resolve outside the new build, fusion marker is absent, 8080/8081 health or service PID/restart state changes, or provenance is incomplete. Do not claim runtime fusion or performance.

## Allowed claims

- `SLX03_GDN_FUSION_BUILD_CALLABLE_R1`
- `SLX03_GDN_FUSION_BUILD_NOT_CONFIRMED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
