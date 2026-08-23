# LAB-VLM-001 — visual coding expansion pre-registration

Frozen before model execution: 2026-08-22.

## Model and runtime

- Existing locally resident cross-family profile: `gemma-4-12b-vision`
- Endpoint: isolated LAB server on port 8092
- Sampling: temperature 0; maximum 512 new tokens; one request at a time. The profile's frozen
  reasoning budget is 256 and its qualified MTP assistant remains enabled.
- Four newly generated, deterministic PNG fixtures; fixture text is the ground truth

## Cases and frozen clauses

1. `stack_trace.png`: recover exception, file, line and implicated identifier. Required clauses:
   `NullReferenceException`; `PaymentService.cs`; `132`; `order`.
2. `ui_bug.png`: identify that the checkout button is clipped/overflowing its card. Required clauses:
   `checkout`; one of `clip/overflow/outside/cut off`; one of `card/container/panel`.
3. `visual_diff.png`: identify label, color and badge changes. Required clauses: both `Deploy` and
   `Delete`; both `green` and `red`; `3 alerts` plus one of `missing/removed/gone/absent`.
4. `terminal_failure.png`: recover test, HTTP status, exception, identifier and source location.
   Required clauses: `test_login`; `500`; `KeyError`; `user_id`; `auth.py`; `87`.

## Gate

- `PASS`: non-empty 4/4, at least 3/4 cases satisfy every frozen clause, and aggregate clause pass
  rate is at least 85%.
- `HOLD`: server/runtime/image ingestion failure or any empty completion.
- `FAIL_QUALITY`: runtime succeeds but the quality threshold is missed.

This is a controlled visual-coding accept suite, not a public VLM benchmark or a claim of general
screenshot understanding.

Pre-execution admission note: the previously preferred `qwen3-vl-8b` profile remains registered but
its model and projector files are no longer resident. The arm was changed to the already resident
Gemma profile before any model response was generated; thresholds and fixtures were unchanged.
