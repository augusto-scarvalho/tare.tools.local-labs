# LAB-VLM-001 — visual coding expansion result

Date: 2026-08-22  
Decision: **PASS (4/4 cases; 20/20 frozen clauses)**

## Outcome

The locally resident `gemma-4-12b-vision` profile passed the bounded visual-coding expansion. It
correctly recovered a stack trace, diagnosed an overflowing checkout button, enumerated three visual
diffs, and extracted a failing pytest record. All four completions were non-empty and each case
satisfied every pre-registered clause.

| Case | Clauses | Wall time | Key result |
|---|---:|---:|---|
| stack trace | 4/4 | 2.32 s | `NullReferenceException`, `PaymentService.cs:132`, `order` |
| UI bug | 3/3 | 2.13 s | checkout button overlaps/overflows its shipping-address container |
| visual diff | 6/6 | 1.87 s | Deploy→Delete, green→red, `3 alerts` removed |
| terminal failure | 7/7 | 2.13 s | `test_login`, 200→500, `KeyError user_id`, `auth.py:87` |

## Boundary

This is a deterministic synthetic accept suite, not a public benchmark. It expands the tested task
surface and confirms that the already-qualified Gemma VLM can handle four coding-oriented screenshot
patterns; it does not establish general visual-agent accuracy. The registered `qwen3-vl-8b` profile
could not be reused because its model/projector files are no longer resident. That was caught before
model execution; thresholds and fixtures were unchanged.

Runtime profile: Gemma-4-12B Vision Q4_0, Q8_0 projector, reasoning budget 256, MTP assistant,
temperature 0, maximum 512 tokens, port 8092 in LAB mode. The canonical 8080 service was restored and
embedding port 8081 remained healthy.

`results.json` SHA-256:
`e403a3e684cfbb13753c264a902ed1bd570b26f45ebd4e383fbe48ba02b896fa`.

Fixture SHA-256 values:

- `stack_trace.png`: `e676c8c778ac79546fd54a44a5d2a5b3ec00a1e22c397cf3546bed616fc74b4a`
- `ui_bug.png`: `b5bf6aa1c9c9efcf8010a96ef307bb03a4b2c112e074f5212e306c32c0a2c24e`
- `visual_diff.png`: `ead1d16f8f5e85b98228485eeca9ca1027c0c1a3566a70646b211597c7083000`
- `terminal_failure.png`: `4900c57541b20c980fb259fb66f5b5362920887daeedc3e851771fbb09b515e0`

