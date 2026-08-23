# LAB-PROV-001 fleet inventory result

Decision: **AUTHORIAL LINEAGE CLOSED / 31 FULLY PINNED / 1 LOCAL DIGEST MISMATCH**

The inventory covered all 33 GGUF files at depth four or less under `/home/augus/models` without
guessing source repositories from filenames. The active Qwen3.8 Q4_K_XL was re-hashed in this run;
its local SHA-256 exactly matched the Hugging Face download digest at revision
`f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`.

## Classification

| Class | Count |
|---|---:|
| Fully pinned: repository + revision + matching local content digest | 31 |
| Repository/revision/digest known; local digest mismatch | 1 |
| Local derivation recipe + matching recorded content digest | 1 |

All 32 GGUF headers parsed successfully and report quantization metadata version 2. Nineteen carry
embedded imatrix lineage fields (dataset/file, entry count and chunk count); thirteen do not expose
embedded imatrix metadata. Absence of those fields is recorded as absence of embedded evidence, not
as proof that no calibration data was used. Exact third-party quantizer builds remain undisclosed for
31 artifacts: a bounded scan of the locally retained model cards and metadata found no commit/build
receipt, so they remain `UNKNOWN` rather than being inferred from format or filenames.

The newly resolved source mappings include the Qwen3.6 base/MTP GGUFs, the Bartowski Qwen3.8 and
ThinkingCap artifacts, the DavidAU Fable artifact, the official Gemma 4 artifacts, GPT-OSS and the
Nomic embedding model. The follow-up recomputed the embedding, Gemma projector, Janvitos Gemma MTP
and ThinkingCap Rank-64 LoRA hashes; all four match their exact upstream digests. The large
ThinkingCap MTP, Mistral Heretic and Gemma Heretic files were tied to exact source revisions, digests
and byte sizes without unnecessary full local re-reads. No repository remains unresolved.

The local Fable-TC task-arithmetic merge is content-pinned as
`052c08ca13d75d8d88c9cc3f201d7bfa9167e2a1e69ad3e1e1f26ff73c1b390b` with the documented
formula `W = Fable + 1.0 * (ThinkingCap - Qwen3.6-27B base)`. Its original Hugging Face cache still
contains exact revision/LFS receipts for all 31 parent weight shards: base 15, ThinkingCap 3 and Fable
13. Their canonical aggregate weight-manifest SHA-256 is
`ee9d33a75716bcec5edd3c69bfdd3cc5d431d7a4a745098228566463eb5eec25`; individual shards and
per-parent manifest hashes are in `parent_receipts.json`.

The preserved `llama-quantize` executable contains commit `068764d92`; the full source revision is
`068764d927ecd6d39665a46d31b1ee533eedabe7`, binary SHA-256
`279e1a5e934fb16b558eeda6d183829a4fc214c9ee8a63fa5e062bf71cd0b4d6`. The conversion script at
that commit matches the preserved file SHA-256
`8f1bed9466221e57e434caa7ee720abe1569deb6bc2fe5a65da950ea66c8e737`. File timestamps place both
before the final GGUF, and the frozen recipe records Q4_K_M without imatrix. This closes the authorial
merge's quantizer and parent-content lineage without hashing the 166.7 GB of parent shards again.

Four promotion-relevant resident candidates were subsequently re-read for compact role screens:
Mistral Small 24B Heretic, Gemma 4 26B Heretic, GPT-OSS 20B and the official Gemma 4 26B QAT.
The continuation then recomputed all 16 remaining hashes (199.448 GiB). Fifteen matched, raising the
fully pinned fleet count to 30. The local ThinkingCap MTP file did not match its pinned revision in either
size or SHA-256 and is explicitly classified `REVISION_PINNED_LOCAL_DIGEST_MISMATCH`; it was preserved
without overwrite. The full receipt and exact values are in `HASH_AUDIT.md`.

The subsequently admitted Ornith 1.5 35B-A3B IQ4_XS artifact also matched its exact revision-pinned
digest, bringing the current fleet to 31 fully pinned artifacts out of 33.

Machine-readable evidence: `inventory.json`, `inventory-full-hash.json` and `parent_receipts.json`. Reproducible collectors:
`tools/analysis/fleet_provenance.py` and `parent_receipts.py`.
