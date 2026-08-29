# Continuous research recovery closeout — 2026-08-28

## Executive summary

The continuous executor, persistent watcher and independent GPT-5.6 Sol xhigh
audit trail completed its first two physical follow-up tracks. It recovered
scoring false negatives, qualified the SLX-03 GDN fusion route and its bounded
performance effect, and extended the Qwen3.8 Q8 KV result to long context and
two concurrent slots.

The most important correction was methodological: the first Q8 long-context
run looked negative because raw `/completion` returned one-token empty EOS
responses for 40 of 48 requests in both arms. The packet was rejected as
"not demonstrated", not as Q8 inferiority. Repeating the frozen task through
the supported chat contract produced 48/48 correct, non-empty responses and
24/24 exact F16/Q8 pairs. This is a caught harness false negative.

## Decisive result ledger

| Packet | Decision | Independent result | Claim boundary |
|---|---|---|---|
| `BACKLOG-BLIND-NUMERIC-RELABEL-03` | PROMOTED | Two blind raters resolved the remaining numeric-label policy conflict | Frozen retained outputs only |
| `BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-04` | PROMOTED | Trace finalist won 29/256 paired answers; delta 0.11328125, 95% bootstrap CI 0.046875 to 0.1796875 | Frozen third-panel deployment comparison |
| `BACKLOG-QWEN38-Q8-KV-UTILITY-04` | PROMOTED | F16 40/128 versus Q8 39/128; delta -0.0078125, CI -0.0390625 to 0.015625; 872 MiB saving | Qwen3.8 Q8 KV, frozen utility panel |
| `BACKLOG-SLX03-GDN-FUSION-BUILD-04` | PROMOTED | Retained Release build proved linked GDN objects, dispatcher call edge, SM86 cubins, identity and callability | Immutable `slop.cpp` commit `87a416bd`; no runtime/performance claim |
| `BACKLOG-SLX03-GDN-FUSION-INSTRUMENTED-01` | PROMOTED | Instrumented Release binary qualified for runtime route observation | Instrumentation only; no speed claim |
| `BACKLOG-SLX03-GDN-FUSION-RUNTIME-03` | PROMOTED | OFF/ON/ON/OFF route markers 0/2160/2160/0; 32/32 HTTP; 16/16 exact pairs | Frozen Qwen3.8 runtime and request shape |
| `BACKLOG-SLX03-GDN-FUSION-PERF-01` | PROMOTED | Decode ratio 1.051362, CI 1.041026 to 1.062275; wall ratio 1.046993, CI 1.038336 to 1.055308 | Single slot, 64 fixed output tokens; no deployment or concurrency claim |
| `BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-01` | REJECTED | Raw-completion contract emitted 40/48 empty one-token EOS responses symmetrically | Harness failure; does not establish Q8 inferiority |
| `BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-02` | PROMOTED | 48/48 correct and non-empty, 24/24 exact pairs at 8,141/16,149 prompt tokens; Q8/F16 throughput 0.958663; 872 MiB saving | Single-slot Qwen3.8 chat retrieval at 8k/16k |
| `BACKLOG-QWEN38-Q8-KV-CONCURRENCY-01` | PROMOTED | 24/24 overlapping two-request batches, 48/48 correct, 24/24 exact pairs; rate ratio 1.004780; 1,698.5 MiB saving | Two slots, at most 16k per slot; no fleet-wide or production claim |

## What worked

- Independent blind relabeling converted disputed scorer behavior into an
  immutable input for the two final aggregators.
- The trace-distilled deployment finalist retained a statistically positive
  answer-level gain on the frozen third panel.
- Q8 KV preserved exact task behavior in the broad utility panel, at 8k/16k
  single-slot context, and under physically overlapping two-slot execution.
- SLX-03 advanced through distinct build, instrumented-route and causal
  performance gates. The performance packet used 12 fresh processes, 144
  fixed-work requests, alternating crossover order and a 20,000-resample
  hierarchical bootstrap.
- Every serving experiment restored the original text route on port 8080 and
  left the embedding service on port 8081 healthy.

## What failed, and why

- SLX-03 build R1 through R3 failed because the qualification harness used
  inadequate build/identity observations. Their raw evidence remains
  preserved; R4 repaired the observations without rewriting the earlier runs.
- SLX-03 runtime R1 and R2 could not prove route activation because their
  journal capture missed or truncated the relevant process-bound evidence.
  R3 used PID-bound startup capture and produced an auditable OFF/ON contrast.
- Q8 long-context R1 used a raw completion contract that was not semantically
  suitable for the frozen prompt. Its apparent low recall was symmetric and
  dominated by empty EOS responses. R2 changed only the serving contract and
  added explicit non-empty and valid-timing gates.

These are not hidden or overwritten failures. They delimit what the successful
successors actually corrected.

## False negatives recovered

1. Numeric extraction and label-policy disagreements had understated both the
   trace deployment result and Q8 utility. Blind relabeling and immutable final
   aggregation resolved them.
2. Release debug-string absence was not evidence that SLX-03 had not been
   linked. ELF symbols, call edges, cubins and runtime markers established the
   physical route.
3. Empty EOS outputs from the raw completion endpoint were not evidence of Q8
   long-context degradation. The chat-contract successor falsified that
   interpretation with exact paired behavior.

## Independent review bindings

| Packet | `REVIEW.json` SHA-256 |
|---|---|
| `BACKLOG-SLX03-GDN-FUSION-BUILD-04` | `59bfad4ba63444b508b45908d772547846603accfeb9e0c7f3539280b79667ba` |
| `BACKLOG-SLX03-GDN-FUSION-INSTRUMENTED-01` | `008b1321b2b01f9ddd05f5807e4e7edfec308ca68c0dcae32b0178d6f64680bf` |
| `BACKLOG-SLX03-GDN-FUSION-RUNTIME-03` | `4ef6e60706cf87ef74052eebffdeafb7d3901a644cc0af6e6a3ef3a32925ceb0` |
| `BACKLOG-SLX03-GDN-FUSION-PERF-01` | `4803fa60bae51d2dc2706a3f7787675d05578813c4c94615a0fa17a8e7635393` |
| `BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-01` | `c88f586556e10d9120cf6790f6514b5406d9b4a99ffff564899fd16ee34ef393` |
| `BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-02` | `3f759898e5e58e1f5a7d305724be869974a4d75ca71e92274084c703df08a15a` |
| `BACKLOG-QWEN38-Q8-KV-CONCURRENCY-01` | `5fe459c39176049a25126e89bbf51c224107acbd0d569a02facf408404297fbc` |

## Backlog handoff

The backlog is not empty. At closeout it contains 108 records: 24 promoted, 21
rejected, 19 executed, 39 blocked, two implemented and three proposed. The next
dependency-gated phase is `P3-PROVENANCE-HOLD-RECOVERY`, with these admitted
successors:

- `BACKLOG-FLEET-CONTEXT-ENVELOPE-04` (P0, current `next`);
- `BACKLOG-FLEET-CONTEXT-INTERFERENCE-02` (P0);
- `BACKLOG-GATEWAY-ROUTE-STRESS-02` (P1).

Their predecessors already have retained physical evidence but need immutable
source/final-input binding and recomputation. The three successors explicitly
forbid new inference. Gateway R2 also separates transport and route identity
from semantic content eligibility so that empty reasoning-only responses cannot
be misreported as successful task completion.
After that, `P4-REMAINING-HOLDS` ranks the remaining blocked/rejected items by
false-negative likelihood and information per GPU-hour.

The machine-readable continuation point is
`config/research_trails/continuous_recovery_2026-08-28.json`. The watcher polls
at 300 seconds and delivers only completion, failure or audit-barrier events.
It may advance complete execution evidence to `EXECUTED`, but only independent
review can write a terminal scientific decision.

## Repository and authority boundary

The experiment authority covers execution and documentation, not remote
publication. This closeout therefore records the worktree without implying
commit or push authorization. All generated packets, failed predecessors, raw
evidence and receipts must remain together when the trail is eventually
published.
