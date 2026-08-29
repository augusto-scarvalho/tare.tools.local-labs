# BACKLOG-SLX03-GDN-FUSION-BUILD-04 preregistration

Task: Forensic qualification of the retained SLX-03 GDN fusion build
Evidence class: `external_reproduction`

## Hypothesis

The retained Release/SM86 artifact from R3 contains the intended SLX-03 GDN fusion implementation and an internal dispatcher call edge, even though the R3 debug-string detector was false. This establishes only compiled inclusion and internal callability, not runtime branch execution.

## Frozen inputs

- `runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-03/raw/receipt.json`
- `runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-03/raw/build_receipts.json`
- `runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-03/raw/end_to_end_artifact.json`
- `runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-03/raw/source_revision.json`
- `runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-03/REVIEW.json`

The runner freezes this admission/preregistration plus the R3 receipt, build receipt, end-to-end artifact, source revision and independent review by SHA-256. It also requires the live retained binary and CUDA-library hashes to match R3 before inspection.

## Command

```powershell
python tools/research/run_slx03_gdn_fusion_build_r4.py
```

## Factors

No rebuild and no model load. Read-only inspection of `/home/augus/src/slop.cpp-main/build-slx03-gdn-audit-03`: response-file object membership; two exact defined ELF symbols; a dispatcher range derived dynamically from `nm -S -a`; an `objdump` call edge within that range; all nonempty `cuobjdump --list-elf` entries targeting SM86; dereferenced artifact size; hashes, self-linkage, callability and service invariance.

## Acceptance gates

- `source_revision`: `exact_source_commit eq True`
- `artifact_identity`: `retained_artifact_hashes_match eq True`
- `linked_objects`: `required_cuda_objects_linked eq True`
- `elf_symbols`: `required_gdn_symbols_defined eq True`
- `dispatcher_edge`: `dispatcher_calls_fused_cache eq True`
- `sm86_code`: `only_sm86_cubins_observed eq True`
- `dereferenced_size`: `cuda_library_referent_bytes ge 60000000`
- `self_linkage`: `project_libraries_resolve_to_retained_build eq True`
- `callability`: `server_version_exit eq 0`
- `service_invariance`: `gateway_and_embedding_unchanged eq True`

## Abort conditions

Abort on any frozen-input or retained-artifact mismatch, missing/ambiguous dispatcher range, failed command, missing object/symbol/call edge, empty or non-SM86 cubin list, incorrect linkage, non-callable binary, service change, or incomplete provenance. Preserve negative evidence. Do not rebuild, load a model, or modify 8080/8081.

## Allowed claims

- `SLX03_GDN_FUSION_BUILD_CALLABLE_R4`
- `SLX03_GDN_FUSION_BUILD_NOT_CONFIRMED_R4`

Claims outside these codes are forbidden even if a metric looks favorable.
