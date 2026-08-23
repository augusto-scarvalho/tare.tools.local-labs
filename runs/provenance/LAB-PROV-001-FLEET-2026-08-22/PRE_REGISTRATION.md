# LAB-PROV-001 fleet inventory — pre-registration

Frozen before execution on 2026-08-22.

Inventory every GGUF at depth four or less under `/home/augus/models`. Source repository mappings
are admitted only when an explicit workspace receipt exists. Hugging Face download metadata and tree
manifests may establish a revision, expected digest and size, but do not count as a recomputed local
content hash. Existing machine-readable benchmark identity receipts may supply a prior full hash.

For this pass, recompute SHA-256 only for the active Qwen3.8 Q4_K_XL artifact; it is the current
deploy artifact and therefore the highest-value integrity check. Do not burn hundreds of GiB of SSD
reads hashing every historical artifact in one wave. Report unknown repository lineage and missing
local hashes explicitly; never fill them from filenames.
