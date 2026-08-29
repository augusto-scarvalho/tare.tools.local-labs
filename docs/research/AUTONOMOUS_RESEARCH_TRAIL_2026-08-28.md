# Continuous research trail — 2026-08-28

This trail joins the executor, persistent watcher, backlog state machine and an
independent GPT-5.6 Sol xhigh auditor. Its machine-readable source is
`config/research_trails/continuous_recovery_2026-08-28.json`.

## Operating contract

The executor may prepare and run a preregistered dependency-ready experiment,
but stops at `EXECUTED`. The watcher proves that the process was observed,
checks the complete packet and advances only `IMPLEMENTED -> EXECUTED`. A
separate auditor recomputes the result from raw evidence, searches for false
positives and false negatives, writes `REVIEW.json`, and alone decides the
terminal scientific state through `backlog_pipeline.py`.

The watcher normally polls every five minutes and persists detailed state under
`runs/autonomous/`. The controlling agent receives only completion, failure or
audit-barrier messages. Short jobs may use a five-second cadence while the
controller is foreground-bound; this changes latency, not scientific gates.

## Current trail

1. **Scorer recovery complete.** Independent review found numeric-policy false
   negatives in both R3 scorers. A blind two-rater resolution with adjudication
   was promoted, followed by final immutable aggregators: trace deployment R4
   improved 29/256 paired answers (95% bootstrap CI 0.046875 to 0.1796875), and
   Q8 KV utility R4 scored 39/128 versus F16 40/128 (delta -0.0078125; 95% CI
   -0.0390625 to 0.015625). Both final packets are promoted.
2. **Q8 physical follow-up complete.** The first long-context packet preserved a
   harness false negative: raw `/completion` produced 40/48 one-token empty EOS
   responses symmetrically in F16 and Q8, so it was rejected as not demonstrated,
   never as Q8 inferiority. The corrected chat-contract R2 produced 48/48 exact
   responses at 8,141/16,149 prompt tokens, Q8/F16 throughput `0.958663`, and
   872 MiB observed saving. The two-slot successor then produced 24/24 physically
   overlapping batches and 48/48 exact responses, with internal Q8/F16 batch rate
   `1.004780` and 1,698.5 MiB saving. Both corrected packets are promoted; claims
   remain Qwen3.8-specific, at most two slots and 16k tokens per slot.
3. **SLX-03 functional qualification complete.** Build R1/R2/R3 harness failures
   and the Release debug-string false negative remain preserved. R4 independently
   proved linked GDN objects, ELF symbols, dispatcher call edge, 140 SM86 cubins,
   dereferenced artifact identity and callability at immutable commit `87a416bd`.
   The instrumented successor and PID-bound runtime R3 then proved treatment-only
   route markers (`0/2160/2160/0`), 32/32 HTTP success and 16/16 exact pairs. The
   non-instrumented Release crossover finally measured decode `1.051362x` (95%
   hierarchical-bootstrap CI `1.041026..1.062275`) and wall throughput `1.046993x`
   (CI `1.038336..1.055308`) over 144 fixed-work requests. Every accepted claim is
   limited to the frozen Qwen3.8 artifact, binary and request shape.
4. **Provenance-hold recovery ready.** `BACKLOG-FLEET-CONTEXT-ENVELOPE-04`,
   `BACKLOG-FLEET-CONTEXT-INTERFERENCE-02` and
   `BACKLOG-GATEWAY-ROUTE-STRESS-02` are admitted as `PROPOSED`. They rebind and
   recompute retained physical evidence without new inference and may not
   silently broaden the original claims. Envelope R4 is the canonical `next`.
5. **Remaining holds.** Rank by false-negative likelihood and information per
   GPU-hour. Prefer retained-output rescoring and source/build checks before new
   inference.

## Fail-closed boundaries

- No audit is inferred from green unit tests or a complete watcher.
- No raw evidence is rewritten after execution.
- A scientific gate failure is a valid negative result, not an infrastructure
  failure.
- A provenance or identity mismatch blocks signing even when metrics look good.
- No service experiment may disturb embedding port 8081, and the original 8080
  route, PID/restart state and health must be verified after restoration.
- This trail authorizes experiments, not repository pushes.
