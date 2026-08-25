# BEE-L0 source archaeology - result

## Frozen identities

- Repository: `https://github.com/Anbeeld/beellama.cpp`
- BeeLlama head: `ba27edad2a84ff045a556df06661e821285c2fab`
- Upstream repository: `https://github.com/ggml-org/llama.cpp`
- Common ancestor with current upstream: `74ce15741b420b8d6f12e720398458b576c51c2c`
- Current upstream head observed during the audit:
  `f280b26983ad0fdb705a0d9ebf0503e76f2899b0`
- License reported by the repository: MIT.

The common-ancestor comparison is a net fork delta, not a claim that every
commit or changed line in the interval is Bee-authored. The interval contains
upstream merges and contributions from multiple authors.

## Measured net delta

- 872 commits in `common-ancestor..BeeLlama-head`.
- 607 changed paths.
- 68,778 insertions and 26,441 deletions.
- Path status: 301 added, 249 modified, 57 deleted.

This is too large for a whole-fork transplant. Source history and changed paths
confirm the transcript's claim that the relevant unit of transfer is a contract
or mechanism, not the repository.

## Contract surfaces identified

- Cache representation and placement: `llama-kv-cache-kvarn.*`,
  `llama-kv-cache-tail.*`, `llama-kv-cache-placement.*`, low-bit GGML types.
- Physical execution: CUDA/HIP/Vulkan/CPU attention and storage routes,
  workspace fit, materialization fallback, and route-policy tests.
- Stateful lifecycle: prompt checkpoints, rollback, exact tails, multi-slot
  behavior, speculative state, and shared-capacity changes.
- Server policy: adaptive draft depth, reasoning-loop guard, presets/router,
  prompt reuse, and route/fit observability.
- Qualification: KLD/perplexity plumbing, backend route tests, lifecycle tests,
  and build/release matrices.

## Transfer classification

| Surface | Decision | Reason |
|---|---|---|
| Requested/resolved/realized/exercised receipt vocabulary | `ADOPT` | Low execution risk and directly addresses past silent/non-exercised mechanisms |
| Immutable request plus physical descriptor | `ADAPT` | Valuable, but must reuse current slop.cpp planning structures |
| Transactional restore invariants and fault matrix | `ADAPT` | High safety value; current coverage is partial |
| Multi-family KV qualification pack | `ADOPT` | Codec-independent and useful before implementation |
| Adaptive speculation controller | `EXPERIMENT` | Requires lifecycle receipts and a matched no-spec baseline |
| Standard precision tail | `EXPERIMENT` | Use existing runtime types before new representations |
| KVarN codec/native attention | `EXPERIMENT_LATE` | Large backend and lifecycle surface; external canary first |
| Shared multi-slot capacity | `RESEARCH_LATE` | Highest concurrency and ownership risk |
| Whole-fork import and presets as source of truth | `RETIRE` | Unbounded maintenance and semantic drift |

## Verdict

`BEE_L0 = COMPLETE`. Continue with receipt/lifecycle infrastructure and external
qualification. Do not port KVarN or the whole fork from this result.
