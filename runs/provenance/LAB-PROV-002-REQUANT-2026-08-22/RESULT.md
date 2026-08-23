# LAB-PROV-002 — result

**Status:** `PROVENANCE_CLOSED / PARITY_REJECTED`  
**Deployment decision:** retain the revision-pinned Unsloth `UD-IQ4_XS`; do not
promote the authorial plain `IQ4_XS` artifact.

## Decision

The source-to-build lineage is now reproducible and verified. The official
Qwen3.8-27B source was pinned and hashed, converted to BF16 GGUF, then quantized
with a rebuilt pinned llama.cpp quantizer and the exact published Unsloth
importance matrix. All 18 official shards passed their expected SHA-256 and byte
counts.

Behavioral parity is rejected. The authorial artifact passed the agent, cache,
and compact context gates, and was slightly faster, but it regressed the frozen
termination sentinel and changed deterministic generations. It is also larger
and uses more VRAM than the community UD recipe.

## Frozen artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| official BF16 GGUF | 53,808,282,048 | `647ce03af8779a4cf7ec6fe7aa9721b6e2589a2b89c54d4e97f158fe89c7b440` |
| authorial IQ4_XS + imatrix | 15,082,506,944 | `daffef5404a0740316cd1bcca9a1f5f233dd76de9a3eee7a6e92c936f9b9fe22` |
| Unsloth UD-IQ4_XS comparator | 14,252,845,984 | `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199` |
| imatrix | 13,642,656 | `0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1` |

The authorial artifact is 829,660,960 bytes (5.82%) larger. It has 851 tensors:
353 F32, 433 IQ4_XS, 64 Q5_K, and one Q6_K. The Unsloth artifact has 866 tensors
and a much broader 11-type UD mix; its 15 additional tensors are the embedded
layer-64 MTP head excluded intentionally by the authorial `--no-mtp` conversion.
Thus this is a clean same-source quant-class comparator, not an exact reproduction
of unpublished UD tensor-selection rules.

## Compact matched gates

| Gate | Authorial IQ4_XS | Unsloth comparator | Decision |
|---|---:|---:|---|
| 65k residency, total GPU used | 16,684 MiB | 16,004 MiB | both fit; authorial +680 MiB |
| agent suite | 8/8 | retained exact-arm 8/8 | pass |
| cache/cancel/reuse | 4/4 | retained exact-arm 4/4 | pass |
| context 8k/32k/64k | 12/12 | retained broad 36/36 | pass |
| GSM historical-failure replay | 1/5 | retained 1/5 | non-inferior, weak |
| `Mbpp/260`, 2,048-token cap | empty, `length`, truncated | stopped at 412, wrong | authorial regression |

The first authorial MBPP wrapper attempt used a non-canonical provenance enum and
was preserved separately. Its generated result was also empty. The clean rerun
used `VERIFIED_SOURCE` and reproduced the empty 2,048-token truncation, so the
termination finding is model-bearing rather than a receipt error.

## Fresh nonce-controlled performance

Three fresh-process repetitions per workload used identical prompts, nonces,
sampling, engine, template, 65,536 context and q4 KV with the embedding service
resident.

| Median | Authorial | Unsloth UD | Authorial delta |
|---|---:|---:|---:|
| long-prompt prefill | 1,152.00 t/s | 1,109.66 t/s | +3.82% |
| decode | 43.94 t/s | 43.15 t/s | +1.81% |

Only 2/6 deterministic response hashes matched. All three Unsloth decode runs
stopped at 88 tokens. The authorial runs stopped at 93 tokens twice and hit the
256-token cap once. The small throughput gain therefore does not offset the
larger artifact/residency and termination drift.

## Provenance closure

- Official source: `Qwen/Qwen3.8-27B` at
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Source verification: 18/18 shards, 55,563,006,776/55,563,006,776 bytes.
- Converter and quantizer source commit:
  `87a416bd75d5a64e66e55846b779c0a54eca21bd`; quantizer build `10161`.
- Quantization used 496 imatrix entries over 1,251 calibration chunks and took
  346.958 seconds.
- Ports after the packet: 8080 down by the authorized LAB campaign, 8081 healthy,
  8092 down.

## Evidence

- frozen design and official manifest: `DECISION_PACKET.md`,
  `OFFICIAL_SOURCE_MANIFEST.json`
- source verification and build receipts: `SOURCE_VERIFICATION.json`,
  `TOOLCHAIN_RECEIPT.txt`, `ARTIFACT_RECEIPT.json`
- complete conversion/quantization logs: `CONVERT.log`, `QUANTIZE.log`
- fresh compact results: `authorial-agent.json`, `authorial-cache.json`,
  `authorial-context.json`, `authorial-mbpp260-rerun/`, `authorial-gsm5/`
- fresh matched performance: `authorial-performance.json`,
  `unsloth-performance.json`
- server logs: `SERVER_AUTHORIAL.log`, `SERVER_UNSLOTH.log`
