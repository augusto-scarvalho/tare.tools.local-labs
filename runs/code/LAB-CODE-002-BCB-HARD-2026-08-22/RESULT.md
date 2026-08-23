# LAB-CODE-002 BigCodeBench-Hard Tier-1 result

**Date:** 2026-08-22  
**Decision:** `COMPLETE / DISCRIMINATING BASELINE`  
**Score:** official pass@1 **48/148 = 32.43%** (Wilson 95%: 25.42–40.34%)  
**Infrastructure-adjusted:** **48/147 = 32.65%** (Wilson 95%: 25.60–40.59%)

## Bound substrate

- model: historical Qwen3.8-27B Q4_K_XL, SHA-256
  `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`;
- server: slop.cpp `b9863-5e7f6271c`, canonical alias `qwen38-27b`, MTP n3;
- BigCodeBench package: official commit `09dd993f46c3fbf3a799465bb96d524edcb0b199`;
- dataset: BigCodeBench-Hard v0.1.4, 148 tasks, SHA-256
  `cee31f14f29927ca276744b15da05e80ea4d06f0724e6053e3aa6ce17c5b6e7c`;
- evaluator image: `bigcodebench/bigcodebench-evaluate`, digest
  `sha256:a3cd34ec3840a49d6b7afb240f4bdd47c350bc5991043fd0a91773830f7cd405`;
- protocol: Instruct, official prompt/sanitizer, greedy, seed 0, max 1,280 tokens, batch 1,
  thinking disabled, one sample per task.

## Qualification and score

The 12-task spread pilot produced 12/12 nonempty, syntax-valid samples and ground truth 12/12. Its
functional pass@1 was 3/12 = 25%, so the benchmark was already more discriminating than MBPP+; the frozen
infrastructure gate passed and the exact pilot samples were resumed into the full file.

Full generation produced:

- 148/148 unique tasks and 148/148 unique receipts;
- 148/148 `finish=stop`, zero length truncations;
- 148/148 nonempty and AST-valid;
- 148/148 reported compilable by the official `syncheck`;
- completion-token min/median/max: 233 / 446.5 / 1,019.

The official sandbox scorer recorded 48 pass and 100 fail, pass@1 0.324324. Ground-truth validation was
147/148 (0.993243): `BigCodeBench/590` depends on a live Wikibooks request that returned HTTP 403 in two
independent full ground-truth runs. The raw official denominator is preserved; excluding only that
pre-identified infrastructure failure gives 48/147 = 32.65%.

During full scoring, `BigCodeBench/1042` exhausted memory because the generated solution loops on
`client_socket.recv()` until a falsey value while the test mock remains truthy. A worker-one isolated rerun
reproduced the failure and cleanup OOM, so `/1042` remains a genuine model fail rather than an infrastructure
exclusion.

## Tooling deviations retained

- The evaluator image requires full task IDs in `--selective_evaluate`, despite documentation examples using
  numeric IDs. The first pilot scorer call stopped before execution and was retried with canonical IDs.
- Passing scalar `--pass_k 1` triggers an official CLI `TypeError` because Fire converts it to an integer;
  runs used the equivalent iterable `1,1`. No partial score from the failed call was accepted.
- `bigcodebench.syncheck` checks completeness against the Full set and therefore prints irrelevant missing
  IDs for a complete Hard sample file; its compilation phase explicitly accepted all 148 Hard solutions.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `full-samples.jsonl` | `9e536c4650e3d9d589ca753d5c40c65b200f7e3abd2945a09c00ca018c5e467b` |
| `full-receipts.jsonl` | `6fc5db72bdf3eff46cb03dccc14c7d035e048787bb1696f92acd1309d1795e25` |
| `full-samples_eval_results.json` | `9c491d814c4bb8d9249895440e8a3bc5594b6a5c6ae15696ededd34bd21aa830` |
| `full-samples_pass_at_k.json` | `adf2de1f0423ff471f7cce624e8621d8efd6cbcf3734e35f0ae0cea65f15e62f` |
| isolated `/1042` eval | `d9d267bd353c2d04772d9fd81ef8a1ce8aa9e9b2529e42c742b21bed9e76f676` |

This result establishes the first official practical-code baseline for the incumbent; it is not a model
promotion comparison. Docker Desktop was stopped after scoring, its image cache retained, and the canonical
8080/8081 services were healthy at handoff.
