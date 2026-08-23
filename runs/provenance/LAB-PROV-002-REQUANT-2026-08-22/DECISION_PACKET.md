# LAB-PROV-002 — pinned-source requant parity

## Decision to close

Determine whether a locally reproducible `IQ4_XS` quantization of the official
Qwen3.8-27B weights is materially equivalent to the current Unsloth
`UD-IQ4_XS` candidate on the RTX 3090. This is a provenance and parity test; it
does not assume that plain llama.cpp `IQ4_XS` reproduces Unsloth's UD tensor
mix.

## Frozen identities

- Official source: `Qwen/Qwen3.8-27B`
- Official revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- License declared upstream: Apache-2.0
- Expected BF16 safetensors: 18 shards, 55,563,006,776 bytes total
- Exact shard LFS identities: `OFFICIAL_SOURCE_MANIFEST.json`
- Community comparator: `unsloth/Qwen3.8-27B-GGUF`
- Comparator revision: `4ca720788d1e01f1bff70c033e0d0028fd02e502`
- Comparator file: `Qwen3.8-27B-UD-IQ4_XS.gguf`
- Comparator SHA-256: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`
- Importance matrix SHA-256:
  `0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1`

## Authorial toolchain

- Converter source tree commit:
  `87a416bd75d5a64e66e55846b779c0a54eca21bd`
- Converter SHA-256:
  `8f1bed9466221e57e434caa7ee720abe1569deb6bc2fe5a65da950ea66c8e737`
- Quantizer is rebuilt from that same commit before use.
- The post-build executable and linked-library hashes are recorded in
  `TOOLCHAIN_RECEIPT.txt`.
- Runtime comparison uses the same server binary, chat template, KV type,
  context, sampling, prompts, and harness for both artifacts.

## Procedure

1. Download the official source at the frozen revision into a revision-qualified
   directory and verify every safetensors shard against the manifest.
2. Preserve and verify the exact Unsloth importance matrix.
3. Convert the official source to BF16 GGUF with `--use-temp-file --no-mtp`.
4. Quantize from BF16 with the rebuilt quantizer, `--imatrix`, and `IQ4_XS`.
5. Record full hashes, sizes, GGUF metadata, tensor counts and tensor-type mix.
6. Run a matched RTX 3090 smoke/performance arm and compact discriminating
   quality gates against the retained Unsloth comparator.

## Interpretation constraints

- A byte or tensor-layout mismatch is expected because `UD-IQ4_XS` is a
  community mixed-tensor recipe while the authorial arm is plain llama.cpp
  `IQ4_XS` with the same importance matrix.
- The experiment can close source-to-build provenance and deployment parity; it
  cannot infer the unpublished Unsloth quantizer revision or exact UD rules.
- Any crash or architecture-conversion failure is a toolchain blocker, not a
  model-quality loss.
- Existing Qwen3.8 conclusions remain active until the matched gates complete.

## Acceptance

- `PROVENANCE_CLOSED`: all official input and output hashes recorded and the
  conversion/quantization is reproducible from the packet.
- `PARITY_SUPPORTED`: matched gates show no material regression on required
  correctness and no more than 10% median decode-throughput loss.
- Otherwise retain the Unsloth artifact and report the specific divergence.
