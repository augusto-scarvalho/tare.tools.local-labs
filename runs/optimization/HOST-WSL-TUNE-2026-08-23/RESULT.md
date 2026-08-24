# HOST-WSL-TUNE result

Date: 2026-08-23

Decision: **CANONICAL BUILD QUALIFIED / RUNTIME CANDIDATE BOUNDED / NO PRODUCTION SWITCH**

## Canonical-build A/B

The deployed `b9863-5e7f6271c` server and the isolated canonical candidate
`b10165-71676e46c` were tested with identical 131,072-token runtime flags, five
counterbalanced repetitions per prompt class, and 128 forced decode tokens. The embedding endpoint
was down for both arms, so the comparison is symmetric. Three greedy probe outputs were byte-identical.

| Cell | Deployed prompt tok/s | Candidate prompt tok/s | Delta | Deployed decode tok/s | Candidate decode tok/s | Delta | Decode energy delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short | 1028.36 | 1027.60 | -0.07% | 77.39 | 80.48 | +3.99% | -7.04% |
| Long | 1200.24 | 1182.00 | -1.52% | 72.57 | 78.93 | +8.76% | -9.24% |

The candidate passed the frozen no-more-than-3% regression gate and exceeded the 5% gain threshold
on long-prompt decode. Clean-start VRAM was effectively unchanged (about 2.17 GiB free without the
embedding endpoint). The candidate was built in a separate ext4 clone with Release, CUDA, sm_86,
native CPU, OpenMP, CUDA graphs, and quantized FlashAttention settings matching the deployed build.

## Context and runtime envelope

- Candidate n3/ub512 without embeddings: largest tested 4 GiB-reserve context was 57,344 tokens.
  61,440 left 4,076 MiB and failed the 4,096 MiB floor.
- The former 81,920 reserve profile is no longer valid for this runtime: it left 3,516 MiB.
- At 57,344, n4/ub1024 left 3,718 MiB and n4/ub512 left 4,008 MiB; both were rejected before workload.
- At 53,248, n4/ub512 passed with 4,120 MiB free and improved decode by 5.30% over n3/ub512 while
  long-prompt throughput improved 0.54%. Outputs remained byte-identical.
- With the restored embedding endpoint resident, the largest tested n3/ub512 reserve point was
  43,008 tokens at 4,113 MiB free. 45,056 left 4,057 MiB and failed.

These are resource profiles, not replacements for the exclusive 131,072-token canonical endpoint.
The current 131,072 text plus embedding deployment leaves about 1,735 MiB free and therefore remains
an explicitly low-reserve configuration.

## Embedding restoration

Port 8081 had been a manual WSL-held process and was absent. It was restored with its documented argv
as enabled `llm-embedding.service`, with three failures per five minutes and 15-second restart delay.
Both a 768-dimensional embedding request and text generation succeeded afterward.

A bounded CPU-offload test produced cosine similarity 1.0 and median latency 38.83 ms versus 53.24 ms
on the baseline, but recovered only about 108 MiB instead of the pre-registered 256 MiB minimum. The
exact GPU baseline was restored.

## Final state

- Original production binary and 131,072-token runtime restored on port 8080.
- Embedding service enabled and healthy on port 8081.
- SERVE/LAB lock reconciled to SERVE; ports and state are coherent.
- Both services report zero restarts.
- No new `nvlddmkm`, CUDA Xid, WSL `dxgvmb/FORTIFY`, or kernel call-trace alert occurred.
- Canonical Windows source and isolated WSL candidate source are clean. No commit or push was made.

The next production action requires an explicit promotion choice: canary the qualified canonical
binary while retaining 131,072, or adopt a lower reserve profile with the documented context tradeoff.
