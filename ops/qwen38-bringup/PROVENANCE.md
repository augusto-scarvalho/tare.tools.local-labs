# LAB-PROV-001 — Qwen3.8-27B GGUF provenance (recorded 2026-08-16)

Artifact identity for the three downloaded quants. `sha256` pins the exact file (survives HF reuploads);
`chat_template sha8` distinguishes template lineage (see the chat-template fix in `VARIANTS.md`); MTP
status is the measured `nextn` head presence (Phase 1). All three are COMMUNITY_REQUANT (no official GGUF
compared) — genuine `model_sha256` vs an official artifact would be LAB-PROV-002 (not run).

| File | Source | Quant | Bytes | sha256 | template sha8 | MTP | Class |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B-UD-Q4_K_XL.gguf | unsloth | UD-Q4_K_XL (dyn imatrix) | 17,923,394,624 | `bee238bb…c12fb1372` | `12827f24` | present | COMMUNITY_REQUANT |
| Qwen3.8-27B-IQ4_XS.gguf | unsloth | IQ4_XS (static) | 15,705,861,088 | `9fd40d70…e4ace666` | `12827f24` | present | COMMUNITY_REQUANT |
| Qwen3.8-27B-Q4_K_M.gguf | bartowski | Q4_K_M (imatrix, tool-calling calibrated) | 17,772,537,440 | `e103abf9…86bca52b` | `c3cf9e34` | present | COMMUNITY_REQUANT |

Common geometry (from GGUF metadata): arch `qwen35`, block_count 65 (64 + nextn/MTP head at blk.64),
`nextn_predict_layers=1`, 866 tensors, dense (n_expert=0). unsloth UD & IQ4_XS share the corrected
UD-style chat template (`12827f24`); we downloaded post the "all updated" reupload, so neither carries
the "System message must be at the beginning" bug (verified by rendering with a system message).

Full sha256 (untruncated):
- UD-Q4_K_XL: `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`
- IQ4_XS:     `9fd40d7036f5e0918e20aaeebf11468fafd06bb53d4d980eef6bb7e4e4ace666`
- bartowski Q4_K_M: `e103abf9d914d1d7b2f2592f055f2759a71195c350a01c135f71aaae86bca52b`
