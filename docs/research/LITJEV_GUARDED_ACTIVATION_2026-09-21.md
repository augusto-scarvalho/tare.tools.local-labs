# Resident decision worker and guarded activation

The existing LitJEV diagnostic remains unchanged (6/6 original, 4/6 fresh).
This delivery addresses the unknown-cost failure with deterministic OS evidence
admission and makes resource selection operational. It does not fine-tune or
claim that the model's accuracy improved.

## Serving changes

`tools/serving/decision_worker.py` uses Qwen only when already resident. A short
external text lease protects template/tokenizer preparation. Actual inference
is submitted to the gateway with `_tare_resident_backend_pid`; the gateway
rechecks PID, model and enabled coordination under its own GPU lease. A stale
request fails without loading or switching a model. Resident requests allow
one output token, wait150ms for admission and have a15-second proxy timeout.
Worker termination does not release the server's inference lease. Existing
transport-failure cleanup stops the backend under that lease.

Before-inference unavailability selects the CPU worker; uncertain submitted
inference outcomes and semantic abstention never trigger another attempt.
OpenJEV CPU requires36 GiB available RAM, an exclusive evaluator lock and a
52-second lifetime. The external OS call budget remains60 seconds. No permanent
GPU helper, additional weights or package environment was installed.

The first implementation held the inference lease in the worker. Independent
review identified cancellation/slot-observation weaknesses. The successor moved
inference ownership into the gateway. Initial deployment/canary records remain
under `DEPLOYMENT.json`/`CANARY.json`; the active successor uses `_V2` files.

## Real canaries

| Deployed successor | Wall time including worker process | Usage | Result |
| --- | ---: | --- | --- |
| Qwen resident GPU | 0.933 s | 742 input / 1 output | Expected cheaper adequate pair |
| GPU lease held; OpenJEV CPU fallback | 45.190 s | 2051 input / 0 output | Expected cheaper adequate pair |

The CPU call held a diagnostic image lease for its whole duration. The text
backend stayed unloaded and the lease remained held; no image/training was
launched or interrupted. This is a resource/transport canary, not a new model
quality sample or concurrent long-context/embedding capacity qualification.

One idle gateway restart installed the resident admission capability:
PID381697 ->916433, automatic restart count remained0. ComfyUI PID381700 and
embedding inactive state were unchanged. The versioned old release and exact
systemd override backup remain available in `SERVICE_UPDATE.json`. Initial
empty text residency was restored after both canaries; GPU lease is free.

Active worker bundle:
`/home/augus/.local/share/tare-assessments/20260921-guarded-v2`.
Active gateway release:
`/home/augus/.local/share/tare-qualified-models/releases/resident-abab824d4bda813d3178`.

## Validation and disposition

Kernel:39 assessment/readout tests passed. Windows serving tests:19 passed,
4 POSIX tests skipped; the gateway/lease tests were also run on Linux,14 passed.
OS focused guarded/strategy/observer/plan tests:68 passed. Work-dispatch/audit
regressions plus guarded/observer tests:87 passed,3 platform skips. These suites
overlap; the numbers must not be summed as distinct test cases. Serving review
found no remaining material defect after the gateway ownership change.

The operator's local default policy is guarded. Its current candidate menu has
no real complete implementation+audit+rework cohort records, so actual missing-
evidence Work requests retain baseline routing before inference. No records
were invented to force a switch. OS owns that policy and its detailed activation
receipt; Local Labs owns hardware/serving observations. Synthetic test evidence
cannot be installed as production delivery evidence.

SpecGraph index integrity passes. Local Labs reconciliation still reports the
existing BOM parse issue in `src/model_lifecycle/analysis/promotion.py`; that
unrelated failure is not a serving test failure or a new promotion.

Evidence: [successor canary](evidence/litjev-enable-20260921/CANARY_V2.json),
[deployment manifest](evidence/litjev-enable-20260921/DEPLOYMENT_V2.json),
[reversible service update](evidence/litjev-enable-20260921/SERVICE_UPDATE.json).
