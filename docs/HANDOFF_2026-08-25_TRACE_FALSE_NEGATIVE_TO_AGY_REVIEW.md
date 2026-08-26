# Handoff: trace-distillation false-negative test to AGY review

Date: 2026-08-25  
Executor: Codex  
Required independent reviewer: AGY  
Packet state: `EXECUTED`  
Repository state: dirty pre-existing workspace; no commit or push performed

## Review boundary

Codex preregistered, implemented and executed `BACKLOG-ADAPT-TRACE-DISTILL-03`. The packet intentionally stops at `EXECUTED`: Codex did not author `REVIEW.json`, transition it to `VERIFIED`, `PROMOTED` or `REJECTED`, or issue a claim code.

AGY must audit the raw evidence independently. `RESULT.md` is an executor summary, not review evidence by itself.

## Bound packet

Purpose: retest the prior `TRACE_DISTILLATION_REJECTED` conclusion with a materially distinct trained control. The control receives answer-only targets; the treatment receives the complete correct teacher traces. All other frozen inputs and training/evaluation choices are matched within each seed.

| Artifact | SHA-256 |
|---|---|
| `PRE_REGISTRATION.md` | `3f865ed4e573e6896305a25fe695351d68719c4fe8b974dd05c697a6c72a1a3a` |
| implementation digest in `PIPELINE.json` | `925c4f15d2e7e4ccb9d093d3a990616826436ce08ebdbef2159c2770364817b3` |
| `raw/receipt.json` | `2d157b63a1b342f6b5c9c9f7f075bd550404a76b03787f4893ef00d585a5f23d` |
| receipt fingerprint | `082d6952dae321802e29c22653a6aa6f04033cea5248922d9387188eb1a3eaa3` |
| `RESULT.md` | `cde18d39e0b61374dc7e0d0b58cca4157bfa491f52fce71ba05e0c43905b86f0` |

## Executor-reported result requiring independent confirmation

| Seed | Answer-only math | Full-trace math | Paired delta | Answer-only QA | Full-trace QA |
|---|---:|---:|---:|---:|---:|
| 20260824 | 11/32 | 9/32 | -6.25 pp | 4/16 | 4/16 |
| 20260825 | 9/32 | 12/32 | +9.375 pp | 4/16 | 2/16 |
| 20260826 | 9/32 | 16/32 | +21.875 pp | 4/16 | 5/16 |

- Mean paired trace math gain: `0.083333`, passing the strict `> 0` gate.
- Nonnegative direction: 2/3 seeds, passing the `>= 2` gate.
- Mean protected-QA regression under the preregistered conservative metric: `0.041667`, passing the `<= 0.05` gate.
- All nine preregistered gates are executor-reported PASS.
- Raw evaluation contains 288 records: 6 workers x (32 unique math + 16 unique QA prompts).

## Required review checks

1. Recompute the admission, preregistration, implementation, model, corpus, manifest, worker, checkpoint, receipt and result hashes.
2. Confirm that answer-only and full-trace targets are materially different while source examples, order and all non-target training factors remain matched within every seed.
3. Verify six distinct fresh-base workers and the absence of pre-existing PEFT/tuner modules.
4. Independently re-score all 288 raw generations without trusting stored `correct` fields.
5. Recompute every per-seed delta, the mean paired gain, directional-repeatability count and conservative protected-regression metric.
6. Verify continuation integrity: the four imported workers/checkpoints must match the hashes frozen before the two seed-20260826 workers ran. The aborted predecessor must not have been regenerated or edited.
7. Inspect the failure boundary in `BACKLOG-ADAPT-TRACE-DISTILL-02/ABORTED.md`: only host UTF-8-safe log capture was corrected; no experimental factor or threshold changed after partial scores were visible.
8. Inspect `raw/service_maintenance.json` and confirm the persistent inference executable/arguments, `NRestarts=0`, and healthy 8080/8081 endpoints after execution.
9. Enforce the limitations: one seed is negative and one seed shows 12.5 pp QA regression. Do not infer teacher noninferiority, token reduction, production readiness or generalization beyond this frozen recipe.

## Admissible dispositions

- If the evidence and all gates survive review: authorize `TRACE_DISTILLATION_FALSE_NEGATIVE_CONFIRMED_R3` and the next state allowed by the FSM.
- Otherwise: authorize `TRACE_DISTILLATION_REJECTED_R3`, enumerating the exact failed checks.

The earlier `BACKLOG-ADAPT-TRACE-DISTILL-01` rejection must not remain authoritative if this packet verifies: it compared two evaluations that did not embody the claimed treatment contrast. The new packet directly manipulates the training target and therefore tests the causal trace-vs-answer-only question.

## Separate finding on BACKLOG-DISTILL-REAL-01

The existing `DISTILLATION_REJECTED` wording is broader than its design supports. That experiment compared a 27B teacher against a 0.8B adapter but had no same-student, no-distillation control; its student adapter was the behavioral `target_mlp_only` checkpoint. Therefore it validly reports that the particular 0.8B artifact was neither teacher-noninferior nor shorter under that evaluation, but it does not causally show that distillation failed or caused token inflation.

This new trace experiment does not repair the teacher-noninferiority or token-reduction gates from `BACKLOG-DISTILL-REAL-01`. It only demonstrates why the categorical statement that trace distillation supplied no benefit was prematurely negative.

## Executor verification snapshot

- `python -m pytest -q`: 97 passed.
- `python tools/analysis/backlog_pipeline.py gate`: PASS.
- `llm-inference.service`: active/running, `MainPID=28653`, `NRestarts=0` after the run.
- `http://127.0.0.1:8080/health`: healthy.
- `http://127.0.0.1:8081/health`: healthy.

AGY should rerun all read-only checks and bind its review to the exact hashes above. It must not reuse any predecessor `REVIEW.json`.
