# LAB-CTX-003 — LongBench RepoBench-P full baseline

**Status:** COMPLETE / QUALITY FAIL  
**Date:** 2026-08-22  
**Substrate:** historical Qwen3.8-27B Q4_K_XL, slop.cpp `b9863-5e7f6271c`, q4 KV, MTP n3, one RTX 3090

## Result

The model-native-template full run scored **39.56** official LongBench code similarity on all 500
RepoBench-P examples, below the preregistered useful-baseline gate of 55.0.

| Slice | n | Official code similarity |
|---|---:|---:|
| Full | 500 | **39.56** |
| reported length <4k | 278 | 45.21 |
| reported length 4–8k | 180 | 36.78 |
| reported length 8k+ | 42 | **14.10** |
| Python | 236 | 49.83 |
| Java | 264 | 30.38 |

Auxiliary strict first-line exact match was 109/500 = 21.8%. The official upstream `eval.py` independently
reproduced `{"repobench-p": 39.56}`.

The strong length gradient is the main result: all inputs fit without truncation, yet similarity falls by 31.11
points from the <4k to the 8k+ source-length stratum. Repo-context completion quality degrades well before the
model's nominal or operational context ceiling.

## Prompt-mode gate

LongBench's stock runner intentionally omits chat wrappers for RepoBench-P. On this instruct model, the
preregistered 20-example raw `/completion` smoke produced meta-explanations instead of next-line code, two
empty EOS completions, and only **1.90** similarity. The frozen amendment reran the same IDs with the native
chat template and thinking disabled:

- 20/20 nonempty;
- 24.55 similarity, +22.65 points over raw;
- 1/20 exact first lines;
- passed the amendment's expansion gate, so the chat arm continued to all 500.

The full score is therefore **official-data/model-native-template**, not directly comparable to LongBench's
historical stock raw-prompt leaderboard. The raw arm remains valid evidence that this instruct checkpoint is
not usable as a plain causal code completer under the stock protocol.

## Operational evidence

- 500 predictions, 500 receipts, 500 unique IDs, 500 nonempty outputs;
- actual prompt tokens min/median/max 2,780 / 10,084.5 / 43,503;
- no input truncation and no context/HTTP/service failure;
- 116 outputs ended at EOS and 384 exhausted the official 64-token budget;
- generation wall time 5,137.3 s (85.6 min), excluding the raw and chat pilot setup;
- model and server remained unchanged from LAB-CTX-002; endpoints 8080 and 8081 were healthy afterward.

The high output-budget exhaustion rate explains part of the gap: many completions begin with prose before
reaching code. It is still a model/task failure under the frozen prompt and official budget, not a harness error.

## Provenance and artifacts

- official LongBench repo commit `2e00731f8d0bff23dc4325161044d0ed8af94c1e`;
- dataset revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`;
- dataset SHA-256 `919a4439e2a84ebb25bacc39ac3b3269a7641af6e02ae205ed78d8c53dfe3568`;
- `chat-predictions.jsonl` SHA-256 `3daf772bc298248f5ddd1871c5f940f36e10861fb392fd94654f6fb51f9abb1e`;
- `chat-receipts.jsonl` SHA-256 `9bb0ca658119ab71184187c34e494c633346c1e8006ccc9f1a80d48a37abfa55`;
- `full-score.json` SHA-256 `37af984f2c43b89d6c133fbf4c439623946d12e7054fb2b169c9e0a8ff8cf905`;
- `official-eval-result.json` SHA-256 `81f0cb3640104bb7dd04c3e2270dd46775da7f24dfe09ec41fa5e23da095049b`;
- raw pilot `predictions.jsonl` SHA-256 `9da96900bc002068c93f418628006d308c0ea32b4b4abed4935509c1b28c139e`;
- generation and scoring adapters: `tools/benchmarks/longbench_repobench_{local,score}.py`.

## Decision

Do not claim repository-scale coding from MBPP+/BigCodeBench or NIAH success. The incumbent is useful on
shorter Python repo-context, but fails the broad RepoBench-P quality gate, is substantially weaker on Java, and
collapses on the 8k+ slice. A future prompt-concision arm may test an explicit “output only the next line” system
instruction, but it must be reported as a non-official prompt optimization and cannot replace this baseline.
