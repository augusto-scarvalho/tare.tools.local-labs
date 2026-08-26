# BACKLOG-ADAPT-TRACE-DISTILL-03 result

Status: `EXECUTED`  
Executor: Codex executor  
Independent review: pending AGY  
Date: 2026-08-25

## Outcome

The properly controlled answer-only versus full-trace experiment passed all nine preregistered execution gates. On the frozen three-seed panel, full teacher traces improved held-out GSM8K accuracy by a mean `0.083333` (8.33 percentage points) over answer-only SFT. The trace arm was nonnegative in two of three seeds and the mean protected-QA regression, calculated according to the preregistered per-seed metric, was `0.041667`, below the `0.05` ceiling.

This is executor evidence that the predecessor's `TRACE_DISTILLATION_REJECTED` conclusion was a false negative: that predecessor did not contain a materially distinct answer-only control arm. It is not yet an authorized claim or promotion. AGY must independently verify the raw evidence before issuing either allowed claim code.

## Controlled design

- Model: a fresh `Qwen/Qwen3.5-0.8B-Base` load for every worker.
- Treatment: completion-only SFT on the complete correct teacher trace.
- Control: completion-only SFT on the final gold answer only.
- Matching: the same 128 source examples, example order, LoRA MLP configuration, optimizer, 128 training steps and decoding configuration within each seed.
- Seeds: `20260824`, `20260825`, `20260826`.
- Held-out evaluation per worker: 32 frozen GSM8K prompts plus 16 protected-QA prompts, greedy decoding.

## Per-seed results

| Seed | Answer-only math | Full-trace math | Trace math delta | Answer-only QA | Full-trace QA | Protected-QA regression |
|---|---:|---:|---:|---:|---:|---:|
| 20260824 | 11/32 | 9/32 | -2/32 (-6.25 pp) | 4/16 | 4/16 | 0.00 pp |
| 20260825 | 9/32 | 12/32 | +3/32 (+9.375 pp) | 4/16 | 2/16 | 12.50 pp |
| 20260826 | 9/32 | 16/32 | +7/32 (+21.875 pp) | 4/16 | 5/16 | 0.00 pp |

Descriptively pooled across seeds, answer-only obtained 29/96 math answers (30.21%) and full-trace obtained 37/96 (38.54%). The preregistered estimand was the mean paired per-seed difference, which is +8.33 percentage points. Two seeds were positive and one was negative. For QA, the pooled difference is -2.08 percentage points; the more conservative preregistered metric averages only per-seed regressions and is 4.17 percentage points because improvements do not offset regressions.

## Gate results

| Gate | Observed | Threshold | Result |
|---|---:|---:|---|
| `continuation_integrity` | 4 frozen partial workers verified | = 4 | PASS |
| `treatment_materiality` | distinct matched targets verified | true | PASS |
| `clean_base` | 6 fresh-base workers | = 6 | PASS |
| `paired_training` | 128 matched examples per arm and seed | = 128 | PASS |
| `seed_coverage` | 3 paired seeds | = 3 | PASS |
| `heldout_gain` | 0.083333 | > 0.0 | PASS |
| `directional_repeatability` | 2 nonnegative seeds | >= 2 | PASS |
| `protected_regression` | 0.041667 | <= 0.05 | PASS |
| `service_recovery` | serving and embedding restored | true | PASS |

## Continuation integrity

`BACKLOG-ADAPT-TRACE-DISTILL-02` stopped after four successful GPU workers because Windows decoded WSL log output with the cp1252 default and raised `UnicodeDecodeError`; the subsequent host log writer then received `None` and raised `TypeError`. That was a host-observability failure, not a model-training failure.

The successor verified and imported the exact four preregistered worker JSONs and checkpoints, then ran only the two missing seed-20260826 workers. No observed partial score changed the hypothesis, seeds, targets, hyperparameters or gates. The only implementation correction was explicit UTF-8 log decoding with replacement for undecodable diagnostic bytes.

## Evidence completeness and operational checks

- `raw/samples.jsonl`: 288 records, equal to 6 workers x 48 held-out prompts.
- Every worker contains 32 unique frozen math IDs and 16 unique protected-QA IDs.
- `raw/receipt.json` binds model, corpora, treatment, workers, checkpoints, samples, aggregates and gate observations.
- Full repository suite after execution: 97 tests passed.
- Backlog pipeline gate after execution: PASS.
- Persistent inference service restored with the same executable and arguments, `NRestarts=0`, and healthy ports 8080 and 8081.

## Claim limits

If AGY independently confirms the bound evidence and gate calculations, the admissible positive claim is `TRACE_DISTILLATION_FALSE_NEGATIVE_CONFIRMED_R3`. Otherwise the admissible disposition is `TRACE_DISTILLATION_REJECTED_R3` with the failed checks identified.

This experiment does not establish teacher-level noninferiority, token-count reduction, a production finalist, or generalization beyond the frozen small panels and recipe. The seed sensitivity is material: one of three seeds lost 6.25 math percentage points, and another incurred 12.5 percentage points of protected-QA regression. A successful independent review supports the bounded causal statement that, under this matched recipe, complete traces improved the preregistered three-seed mean over answer-only targets.
