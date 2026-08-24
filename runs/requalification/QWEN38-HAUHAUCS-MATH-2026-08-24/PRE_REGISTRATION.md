# Qwen3.8 HauhauCS math/general-reasoning expansion — pre-registration

## Question

Does the HauhauCS candidate retain elementary multi-step mathematical
reasoning relative to vanilla Qwen3.8 and operational Fable-TC, extending the
already completed 48-item normal-question panel?

## Frozen design

- Workload: canonical local GSM8K test export, 1,319 rows.
- Sample: 200 rows selected by deterministic shuffle seed `20260820`.
- Same task IDs and order for every arm.
- Inference: greedy, `enable_thinking=false`, 512-token cap.
- Primary metric: strict numeric correctness; the final non-empty line must be
  exactly `#### <number>`.
- Diagnostics: lenient last-number correctness, format adherence, truncation,
  wall time, and per-task paired outcomes.
- Arms, in order: Fable-TC, HauhauCS aggressive, vanilla Qwen3.8 UD-Q4_K_XL.

The runner must pass its self-check before generation and must record stable
live model/server identity before and after each arm. The 8081 embedding
endpoint remains resident throughout model switches.

## Decision rule

- `NO_MEASURABLE_MATH_LOSS`: HauhauCS trails the best comparator by at most
  3/200 tasks, format adherence is at least 98%, and truncation is at most 1%.
- `POSSIBLE_SMALL_MATH_LOSS`: HauhauCS trails the best comparator by 4–7 tasks
  without a large format or truncation defect.
- `MATERIAL_MATH_LOSS`: HauhauCS trails by at least 8 tasks, format adherence
  falls below 98%, or truncation exceeds 1%.

Paired task outcomes and an exact two-sided McNemar/binomial p-value are
reported descriptively; the frozen practical thresholds remain primary.

## Operational exit contract

After all valid arms or any fatal identity/instrument failure, remove temporary
model drop-ins and restore Fable-TC on 8080, embeddings on 8081, and the locale
proxy on 8082. Fan Control remains the owner of fan curves.
