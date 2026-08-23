# Current Unsloth Qwen3.8 revision screen — result

**Execution:** 2026-08-21 through 2026-08-22  
**Status:** `EXECUTED / REJECT_SUPERSESSION`  
**Deployment decision:** keep the historical artifacts and service configuration unchanged.  
**MTP decision:** current separate MTP head was not eligible for a live sentinel.

## Decision

Neither current-revision quant supersedes its historical peer.

- Current `UD-Q2_K_XL` saves 847,442,080 bytes (7.94%) and 832 MiB of measured 65k residency, but added an
  8k aggregation miss and changed `Mbpp/260` from a terminating correct historical answer into a
  2,048-token truncation. It failed the compact eligibility gate.
- Current `UD-IQ4_XS` saves 1,453,015,104 bytes (9.25%) and passed agent, cache, GSM8K non-inferiority, and
  replicated context gates. It nevertheless scored 323/378 Base and 280/378 Plus on MBPP+, versus the
  historical 326/378 and 284/378. Those losses exceed the preregistered tolerances by one task on each
  metric, so it failed broad non-inferiority.
- Because the base artifact failed a required correctness gate, testing its separate MTP head could not
  promote anything and was stopped by rule.

This is a rejection of these exact revision-pinned artifacts, not a permanent rejection of later Unsloth
revisions or the published imatrix.

## Frozen identities

All current artifacts came from `unsloth/Qwen3.8-27B-GGUF` revision
`4ca720788d1e01f1bff70c033e0d0028fd02e502` and were stored under
`/home/augus/models/qwen38-27b/unsloth-4ca72078/`.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| historical IQ4_XS | 15,705,861,088 | `9fd40d7036f5e0918e20aaeebf11468fafd06bb53d4d980eef6bb7e4e4ace666` |
| current UD-IQ4_XS | 14,252,845,984 | `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199` |
| historical Q2_K_XL | 10,676,423,744 | `46151b52a5cad673d90a00222103254864326c251130b8fc4381d6f34386b3c8` |
| current UD-Q2_K_XL | 9,828,981,664 | `fd4730dd8aad070517978752b63d530aeb1740d2283cab9fa24f1e404032ddb0` |
| current separate MTP Q4_0 | 1,369,590,656 | `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e` |

Runtime controls for quality arms were llama.cpp `b9863-5e7f6271c`, the external `qwen-sharp.jinja`,
q4_0/q4_0 KV, one slot, 65,536 context tokens, full GPU offload, greedy decoding, and no speculation.

## Compact screen

| Arm | 65k GPU used / free | Agent | Cache | Context 8k/32k/64k | `Mbpp/260` | GSM failure replay |
|---|---:|---:|---:|---:|---|---:|
| historical Q2_K_XL | 13,333 / 10,990 MiB | 8/8 | 4/4 | 4/4, 4/4, 3/4 | stop at 1,481; correct | 4/5 |
| current Q2_K_XL | 12,501 / 11,822 MiB | 8/8 | 4/4 | 3/4, 4/4, 3/4 | length at 2,048; truncated | 4/5 |
| current IQ4_XS | 16,621 / 7,702 MiB | 8/8 | 4/4 | 4/4, 4/4, 3/4 | stop at 412; wrong | 1/5 |

The compact IQ4 aggregation miss at 64k was not treated as a new cliff because the same single-position
sensitivity appeared in both Q2 generations. It was explicitly retested in the broad packet.

## Broad current-IQ4 packet

### MBPP+

Generation captured 378/378 answers and 378/378 fenced solutions at the historical 768-token cap. Two
tasks truncated: `Mbpp/430` and `Mbpp/782`. `Mbpp/260` stopped normally at 412 tokens but was functionally
wrong.

Official EvalPlus 0.2.0 results:

| Metric | Historical IQ4_XS | Current IQ4_XS | Frozen minimum | Result |
|---|---:|---:|---:|---|
| Base | 326/378 (86.24%) | 323/378 (85.45%) | 324/378 | FAIL by 1 task |
| Plus | 284/378 (75.13%) | 280/378 (74.07%) | 281/378 | FAIL by 1 task |

The Plus failure sets overlap on 83 tasks; the current artifact introduces 15 failures and fixes 11 old
failures. This is behavioral drift rather than one isolated broken response.

### GSM8K

The frozen 100-item manifest scored 94/100 strict, 95/100 lenient, 98/100 format adherence, and three
truncations. This meets the preregistered 94/100 floor but trails the historical 95/100. Strict failures were
`gsm8k/584`, `gsm8k/161`, `gsm8k/1019`, `gsm8k/241`, `gsm8k/1312`, and `gsm8k/1183`.

### Context

The three-replicate paired matrix passed 36/36:

| Applied target | Retrieval | Multikey | Multihop | Aggregation | Total |
|---|---:|---:|---:|---:|---:|
| 8,192 | 3/3 | 3/3 | 3/3 | 3/3 | 12/12 |
| 32,768 | 3/3 | 3/3 | 3/3 | 3/3 | 12/12 |
| 64,000 | 3/3 | 3/3 | 3/3 | 3/3 | 12/12 |

The compact 64k aggregation miss did not repeat across these three preregistered seeds. The artifact has
strong local-context evidence; the supersession rejection is driven by coding non-inferiority.

## Harness notes

Two non-model issues were diagnosed fail-closed:

1. an explicit relative GSM dataset path was rejected before campaign creation; the valid run used the
   harness's identical absolute default path;
2. the MBPP post-processor initially expected EvalPlus 0.3.x result fields and displayed a false zero while
   EvalPlus itself reported nonzero scores. `score_mbpp_subset.py` was updated to normalize both 0.2.x and
   0.3.x schemas, the stale score cache was busted, and the official execution was repeated. Only the
   corrected 323/280 receipt is decision-bearing.

The shared EvalPlus environment remains at 0.3.1. A separate task environment at
`/home/augus/.venvs/evalplus-020` supplied EvalPlus 0.2.0 for historical comparability.

## Restoration receipt

After the packet:

- experimental port 8092 had no listener;
- `llm-inference.service` was active on 8080;
- live alias was `qwen38-27b`;
- live model was historical `Qwen3.8-27B-UD-Q4_K_XL.gguf`;
- build was `b9863-5e7f6271c`, context was 131,072, one slot, q4_0/q4_0 KV, MTP `n-max=3`, and 32 context checkpoints;
- embedding port 8081 remained healthy with its original process.

## Evidence map

- preregistration and frozen thresholds: `PRE_REGISTRATION.md`
- compact JSON receipts: `current-*.json` and `historical-*.json`
- compact GSM/MBPP responses: `*-gsm5/` and `*-mbpp260/`
- broad MBPP generation, identity, raw EvalPlus result, and score: `current-iq4xs-mbpp378/`
- broad GSM manifest and rows: `current-iq4xs-gsm100/`
- broad context rows and summary: `current-iq4xs-context-reps3.json`
