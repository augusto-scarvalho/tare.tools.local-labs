# RWKV7 1.5B official checkpoint — result

Date: 2026-08-22  
Decision: **QUALIFIED_MECHANISM / RESEARCH-LOCAL**  
Deployment license: **BLOCKED — weight license unasserted by publisher**

## Outcome

The official RWKV7 1.5B release is a real additional recurrent-model option for this RTX 3090 at
the mechanism and local-runtime level. It is not promoted over the 27B serving incumbent and is not
deployment-cleared.

## Frozen gates

| Gate | Result | Evidence |
|---|---:|---|
| Artifact identity | PASS | weights/config/tokenizer match the bundled manifest SHA-256 values |
| BF16 one-GPU fit | PASS | 3,055,337,472 allocated bytes; 21,174,943,744 free bytes after load |
| Backend | PASS | isolated Transformers 5.15.0 + TileLang 0.1.12; `auto` resolves to `tilelang` |
| Recurrent continuation | PASS | full vs 64+64 cached suffix max abs difference `0.0` |
| Constant state | PASS | `12,779,524` unique storage bytes at 32, 256 and 1,024 tokens |
| Fresh-state isolation | PASS | fresh rerun difference `0.0`; deliberate stale-state delta `40.0` |
| Bounded behavior | DESCRIPTIVE PASS | non-empty 4/4; natural EOS 4/4; no quality-promotion rule |

## Operational observations

- Model load itself took 2.88 seconds and left roughly 19.7 GiB of free VRAM while the embedding
  service remained active.
- The fresh process spent roughly twelve minutes in the complete gate suite, dominated by first-use
  compilation/prefill at very low GPU utilization. Generated completions after warm-up ranged from
  about 24 to 27 tokens/s on the longer two samples. This runtime is functional but not yet a mature
  serving candidate.
- Deterministic outputs answered all four prompts semantically, but two strict-format prompts carried
  a leading `>` from the chat format. This is base-model behavior and is not agent-quality evidence.
- The shared SGLang environment was not upgraded. Transformers 5.15.0, TileLang 0.1.12, updated
  safetensors/tokenizers/regex were installed only in `/home/augus/rwkv7-overlay` and selected by
  `PYTHONPATH`.

## Provenance boundary

Pinned Hub revision: `d2d414ff676d9d9c40a3d7b5c6faec7d2dd76e13`. The local weights hash is
`84ccbb857c84e00cefc48b233937ada79c411e491df25fb21aed23237f39a14f`. The publisher card calls
this an official BlinkDL release and names the source checkpoint/hash, but explicitly does not assert
a license for the weights. Apache-2.0 covers the exported inference bundle, not these weights.

Receipts: `results.json` (SHA-256
`cc75327f7f5c4f4a08933aed9726fd49ee9ef39a993cf214232cc3ec92e32f1e`) and
`backend_receipt.json`.
