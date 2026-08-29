# BACKLOG-SLX03-GDN-FUSION-INSTRUMENTED-01 preregistration

Task: Build an instrumented Release SLX-03 binary for runtime route observation
Evidence class: `external_reproduction`

## Hypothesis

Immutable source commit `87a416bd` can produce a Release/SM86 `llama-server` with `GGML_CUDA_DEBUG` explicitly defined, preserving the GDN fusion log marker needed to observe branch execution in a later model-loaded experiment.

## Frozen inputs

- `runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-04/raw/receipt.json`
- `runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-04/REVIEW.json`

The runner freezes this admission/preregistration plus the promoted R4 receipt and independent review. It verifies the exact WSL commit and tracked cleanliness before creating `/home/augus/src/slop.cpp-main/build-slx03-gdn-instrumented-01`.

## Command

```powershell
python tools/research/run_slx03_gdn_fusion_instrumented.py
```

## Factors

One fresh Release CUDA build for SM86, target `llama-server`, eight workers, explicit `/usr/local/cuda/bin/nvcc` and `-DCMAKE_CUDA_FLAGS=-DGGML_CUDA_DEBUG`. The experiment verifies the define in compile logs, the exact marker in the final CUDA library, self-linkage, callability, dereferenced size and unchanged services. No model is loaded.

## Acceptance gates

- `source_revision`: `exact_source_commit eq True`
- `tracked_clean`: `tracked_source_clean eq True`
- `configure`: `cmake_configure_exit eq 0`
- `build`: `llama_server_build_exit eq 0`
- `debug_define`: `debug_define_recorded eq True`
- `fusion_marker`: `gdn_fusion_marker_present eq True`
- `self_linkage`: `project_libraries_resolve_to_new_build eq True`
- `callability`: `server_version_exit eq 0`
- `dereferenced_size`: `cuda_library_referent_bytes ge 60000000`
- `service_invariance`: `gateway_and_embedding_unchanged eq True`

## Abort conditions

Abort if the build directory exists, source identity differs, a build/callability/linkage/marker/define/provenance gate fails, or service identity changes. Preserve all negative evidence. This packet cannot claim runtime execution, parity, performance, write reduction or deployment.

## Allowed claims

- `SLX03_GDN_INSTRUMENTED_BUILD_CALLABLE_R1`
- `SLX03_GDN_INSTRUMENTED_BUILD_NOT_CONFIRMED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
