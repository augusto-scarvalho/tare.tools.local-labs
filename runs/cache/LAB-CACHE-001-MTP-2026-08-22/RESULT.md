# LAB-CACHE-001 persistent slot and MTP rollback closure

**Date:** 2026-08-22  
**Decision:** `NO-SPEC SLOT PASS / MTP CACHE AND PERSISTENCE BLOCKED`  
**Substrate:** Qwen3.8-27B historical Q4_K_XL, slop.cpp `b9863-5e7f6271c`, 131,072-token
context, q4_0/q4_0 KV, one slot, 32 checkpoints, RTX 3090.

## Outcome

Explicit slot save, erase, restore, and cached continuation are qualified when speculation is disabled.
MTP n3 is not qualified for cache/checkpoint promotion: the first cache run returned 256 exclamation marks
instead of `MAGNOLIA` in both cold and warm long-context completions, with 759 drafts and zero accepted.
The first MTP slot-persistence run mechanically saved/restored 490,856,088 bytes but reproduced the same
wrong 64-character output. These correctness failures remain disqualifying even though clean reruns passed.

The deployed chat path did not reproduce the problem: no-spec and MTP n3 each passed 16/16 exact context
cells through 32k, and every answer was byte-identical across arms. A separate untemplated raw-depth sweep
also passed 10/10 in both arms through 32k with repetitive and diverse filler. The defect is therefore
intermittent and scoped to the cache/rollback/persistence lifecycle observed in the first MTP instance; it
is not evidence of a general MTP context cliff.

## Frozen results

| Gate | No spec | MTP n3 | Interpretation |
|---|---:|---:|---|
| Four-case cache/cancel/reuse | 4/4 | run pass rate 4/5; case total 19/20 | MTP blocked by one oracle failure |
| Explicit slot save/erase/restore | 1/1 | run pass rate 1/2 | first MTP run persisted wrong state |
| Chat context, 8k/16k/24k/32k | 16/16 | 16/16 | all cross-arm responses byte-identical |
| Raw context, two prompt shapes to 32k | 10/10 | 10/10 | no general raw-depth cliff |
| Clean raw-trigger cells | 3/4 | 4/4 | no-spec 64-token miss was ordinary truncation; cue or 128 tokens passed |
| Pre/post cache-sequence triggers, reproduction 2 | n/a | 2/2 before and 2/2 after | state transition did not reproduce |

All four post-failure fresh-server cache replicas passed. They reduce the estimated frequency but cannot
erase a known deterministic-oracle violation from a correctness gate. No MTP n1/n2 branch was opened:
the preregistered condition for that branch (a clean n3 raw-depth or clean trigger failure) did not occur.

## Evidence map

- `nospec-cache.json`, `nospec-slot-save-restore.json`: qualifying no-spec controls.
- `mtp3-cache.json`, `mtp3-slot-save-restore.json`: original MTP correctness failures.
- `mtp3-cache-rep2.json` through `mtp3-cache-rep5.json`: four fresh passing reproductions.
- `mtp3-slot-save-restore-rep2.json`: clean passing MTP persistence reproduction.
- `nospec-chat-context.json`, `mtp3-chat-context.json`: paired 16-cell chat localization.
- `nospec-raw-context.json`, `mtp3-raw-context.json`: paired raw-depth localization.
- `*-raw-trigger*.json`: prompt/budget and before/after diagnostic cells.
- `PRE_REGISTRATION.md`: frozen controls and sequential localization amendments.

## Operational disposition

- Close the previously blocked explicit slot capability as **PASS without speculation**.
- Keep cache/checkpoint promotion under MTP **BLOCKED** pending a causal fix and a zero-failure rerun.
- Do not reinterpret the result as a reason to disable the current chat deployment: its paired chat path
  passed, and this tranche did not test serving-level regression. Continue to avoid claiming persistent
  MTP slot correctness.
- The canonical `llm-inference.service` was restored on port 8080 with Q4_K_XL, MTP n3, q4 KV, one slot,
  131,072 context, and 32 checkpoints. Embeddings remained healthy on port 8081 throughout.
