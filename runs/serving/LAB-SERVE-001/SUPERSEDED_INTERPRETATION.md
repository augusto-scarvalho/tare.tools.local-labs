# LAB-SERVE-001 — interpretation superseded by LAB-SERVE-001b (2026-08-10)

The LAB-SERVE-001 **raw evidence and report remain valid** as bounded 1-rep shape-discovery. This
note supersedes one *interpretation* only (§1 of the 001b packet); the original report is not edited.

## What 001 said (and what to keep)
Correct and retained: MTP wins **output throughput** at N=1,2,4,8; MTP wins **E2E median** at all N.
Also correct: server topology was unrecorded in 001.

## What is corrected
1. **Server topology (resolved in 001b):** `/props`+`/slots` on the pinned lifecycle fork show
   **4 slots, kv_unified, n_ctx 8192 shared** (the fork defaults `n_parallel` auto→4). So 001 was
   NOT a 1-slot queuing experiment: at N≤4 it exercised real 4-way batching; N=8 exceeded the 4
   slots (batching + queuing). `clientOutstandingConcurrency` (1..8) ≠ `serverSlotCount` (4) ≠
   `activeDecodeConcurrency` (min(N,4)).

2. **"MTP loses on latency at N≥4" was too broad.** Different latency dimensions move differently.
   Use **"TPOT crossover"**, not "universal latency crossover". In 001 (1 rep): E2E median favored
   MTP at all N; only TPOT median showed MTP worse at N=4/8 (+2.8/+7.1 ms).

3. **The TPOT crossover does NOT replicate.** Under 001b (pinned `--parallel 4`, fresh server per
   block, **5 paired blocks**, block-level HL delta + bootstrap CI), at N=4 MTP is **better** on
   TPOT: median on 35.1 vs off 39.7 ms, paired Δ **−5.8 ms**, CI [−7.8, −3.8] (excludes 0), 5/5 reps
   agree. At N=1, Δ −7.8 ms. **So the 001 N=4 TPOT crossover was single-rep noise**, not a real
   effect. MTP wins TPOT at both tested concurrencies (N=1, N=4).

## Net corrected verdict (dense-27B, N∈{1,4}, forced length)
MTP wins throughput, TPOT, and E2E (median and p95) at both N=1 and N=4; TTFT is mixed (MTP slightly
worse at N=1 due to draft setup, much better at N=4). No TPOT crossover in the replicated range.
N=8 remains only 1-rep descriptive (from 001) — not re-measured in 001b.

Full evidence: `runs/serving/LAB-SERVE-001b/` and `report/LAB-SERVE-001b.md`.
