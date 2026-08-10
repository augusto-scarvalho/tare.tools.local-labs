# LAB-SERVE-001c — Pre-registration (fixed BEFORE campaign results, 2026-08-10)

Per §12/§13: load points and block count are fixed here, from a capacity measurement, BEFORE running
the comparison. Not to be revised after seeing results. Owner-approved scope: **minimal, hard ≤4h**.

## Fixed model / topology (unchanged deploy family; NOT modifying deploy defaults)
- Model: `qwen36-35b-a3b-mtp` Q4_K_M (the MoE deploy candidate; reasoning model, thinking on via `--jinja`).
- Server (both arms): `-fa on --ctx-size 73728 --parallel 4 --jinja --n-cpu-moe 8 --cache-type-k q8_0 --cache-type-v q8_0`
  → 4 slots × **18432 tok/slot** (kv_unified=false, static partition; verified in Stage 1). VRAM ~20.4 GB.
- Arms differ ONLY by: OFF = (none) · ON = `--spec-type draft-mtp --spec-draft-n-max 4`.

## Fixed workload (pinned)
- `runs/serving/LAB-SERVE-001c/workload/workload_001c.jsonl` (sha256 in `workload_manifest.json`), 120 items.
- INTERACTIVE (80): prompt 532–4091 tok; per-request output cap ∈ {128,256,512,1024}.
- CODING (40): prompt 8234–9201 tok; per-request output cap ∈ {512,1024,1536}.
- Disjoint input bands ⇒ a request's class is recoverable from its measured `input_len`.
- **Output caps BIND** (reasoning model exceeds them) → outputs are cap-bounded, NOT natural-EOS. This is
  an owner-approved, documented deviation from §9: literal natural-EOS is unproducible on this deploy
  model at realistic interactive sizes. The cap DISTRIBUTION provides the §20 length strata by design.

## Capacity measurement (basis for load points)
Saturated open-loop probe (rate=inf, OFF arm, trimmed workload, 12 prompts): **request_throughput C =
0.085 req/s** (`runs/serving/LAB-SERVE-001c/capacity/captrim_off.normalized.json`; 12/12, token ratio
1.000, mean 875 out-tok/req). Low capacity is real: heavy reasoning outputs + 8k coding prefill on the
CPU-offloaded MoE.

## Pre-registered offered-load points (Poisson `--request-rate`, req/s)
| point | fraction of C | rate λ (req/s) |
|---|---|---|
| LOW        | 0.35 × C | **0.030** |
| NEAR_SAT   | 0.85 × C | **0.072** |
| OVERLOAD   | 1.30 × C | **0.110** |

## Pre-registered design
- Per load point: **2 paired blocks** (reps). Statistical unit = independent server-level block (per 001b).
- Each rep = one OFF server-start + one ON server-start, both running the SAME load point cell with the
  SAME per-rep seed ⇒ IDENTICAL Poisson arrivals + prompt identities/order across arms (§14 common schedule).
- Per-rep seeds: rep1=101, rep2=102 (independent replicate schedules across reps).
- Arm order alternated per rep to decorrelate thermal drift: rep1 = [OFF, ON], rep2 = [ON, OFF].
- Per cell: `--num-prompts 12`, `--warmup 1`, NO `--max-concurrency` (open-loop; the 4 slots queue the excess).
- Total server-starts = 3 points × 2 reps × 2 arms = **12**.

## Guards (hard ≤4h)
- Global deadline = campaign start + 4h. Before each server-start, if the estimated remaining cell time
  would cross the deadline, STOP and emit `STOPPED_ON_BUDGET=true` with whatever cells completed (partials
  are valid evidence; failed/timed-out requests are kept in denominators, never silently dropped).
- Per-cell hard timeout = 1200 s. A cell that times out is recorded as a failure, not discarded.

## Analysis plan (pre-specified)
- Pair ON−OFF by rep per (load point, metric); report median(on)/median(off), Hodges–Lehmann paired
  delta with seeded bootstrap CI + sign test (reuse `analysis/robust.py`). n=2 reps ⇒ direction +
  magnitude only; NO p<0.05 claims (consistent with minimal scope).
- Stratify by class (interactive vs coding via input_len band) and by output-length bin (§17, §20).
- Queueing: report TTFT (median/p95) growth LOW→OVERLOAD as the onset signal. Queue delay is NOT
  separately observable via this instrument, so TTFT is NOT relabeled as pure queue delay (§15); the
  TTFT inflation over the LOW baseline is reported as a clearly-labeled derived signal.
- Status outcomes per §23.
