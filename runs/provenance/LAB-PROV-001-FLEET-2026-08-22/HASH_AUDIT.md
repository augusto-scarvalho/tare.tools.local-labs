# LAB-PROV-001 fleet full-hash audit

Date: 2026-08-22  
Decision after new Ornith admission: **31 FULLY PINNED / 1 LOCAL DERIVATION / 1 UPSTREAM MISMATCH**

The 16 artifacts previously marked local-hash-pending were read in full (199.448 GiB total). Fifteen
matched their exact pinned upstream digests. The sole mismatch is preserved without overwrite:

- Local path: `/home/augus/models/thinkingcap-27b-mtp/ThinkingCap-Qwen3.6-27B-Q4_K_M-MTP.gguf`
- Local bytes: `17221641152`
- Local SHA-256: `b0987c4ea581cff1ad07a94a2703cc636d48ce5a52f383eba827198a32ffc6bb`
- Pinned upstream revision: `f015d8b219c68de4a9554832842675afc08ae577`
- Upstream bytes: `16810713408`
- Upstream SHA-256: `0ba445d2d0ca3ec32f429d83701b42f2ea828c934fc6378b836ffaf1b0760c75`

The collector now gives full recomputation receipts precedence over older benchmark identity sidecars
and classifies a recomputed disagreement as `REVISION_PINNED_LOCAL_DIGEST_MISMATCH`, never as hash
pending or fully pinned.

Receipts: `inventory-full-hash.json` (direct recomputation) and `inventory.json` (durable reconciled
inventory).

After this audit, the exact Ornith 1.5 IQ4_XS artifact was admitted and full-hashed, increasing the
fleet boundary to 33 artifacts and the fully pinned count to 31. It did not change the one mismatch.
