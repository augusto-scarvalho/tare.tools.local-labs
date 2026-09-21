# Von documentation, SDK parity and input-contract review

## Conclusion

The original Option-Marker adapter agrees with the official implementation on all
15 original requests. Its poor routing result was reproduced. However, that screen
did not evaluate the SDK-default backend or establish a general ranking against
Laya/OpenJEV. The distinction matters: both official backends scored 10/10 on a
small public semantic-classification control set, while both struggled with our
strategy-selection inputs, even after a descriptive natural-language rewrite.

Keep Von experimental. Consider it for narrowly defined semantic classification
after task-specific qualification. Do not let these observations authorize model,
effort, cost or flow selection in production. No production worker/config changed.

## Documentation and source findings

1. **Two distinct backends.** The pinned SDK's `VonEngine` maps `von-1.0`, `von`
   and the default convenience API to `BertaBackend` (NLI). Explicit `marker` or
   `option-marker` selects `OptionMarkerBackend`. The former loads
   `model.safetensors`; the latter uses `option_marker.pt`. Our initial worker
   targets Option-Marker, as identified in its method field.
2. **Recommended criteria are descriptive.** The
   [model card](https://huggingface.co/wfzyx/von-1.0#domain-generalization--out-of-domain-tasks-eg-education-academia)
   recommends explicit operational descriptions. Its examples use a short state,
   one classification question and category descriptions. Our original structured
   routing requests combined cohort comparability, quality thresholds, cost, time
   and abstention, with repetitive evidence/provenance fields. That is a different
   demand from intent classification. JSON strings themselves are accepted unchanged
   by both official state formatters; their presence is not a protocol violation.
3. **Different input limits.** The NLI primitive silently requests truncation at
   512 tokens per premise/hypothesis pair. Six of our original requests exceed
   that limit. This audit did not feed those through truncation: the NLI arm used
   the shorter projected inputs and preflighted every pair before inference.
4. **Calibration paths differ.** Marker uses local temperature 2.2 after loading.
   NLI reads calibration metadata 1.1692, but its public choice method defaults to
   a temperature argument of 1.0. The audit retained official behavior. These
   values affect probabilities, not the argmax at a positive temperature.
5. **SDK telemetry is approximate.** Our adapter correctly retains actual token
   counts and zero generated tokens. SDK confidence is a top-two probability
   margin, not verified correctness or abstention authority.

Source: [pinned SDK](https://github.com/wfzyx/von/tree/2656a69be2ef0ebf2f3058302e52791f99cbd8de),
especially `engine.py`, `backends/option_marker_backend.py`,
`backends/berta_backend.py`, `models/option_marker.py` and `presets.py`.

## Direct implementation parity

Same model revision `aa2fdc9630ecdadef32c56073553b3a69bed38bf`, FP32 CPU, two
threads. The reference used unmodified upstream constructors/backend and
Transformers **5.17.0**, matching the checkpoint metadata. Dependencies were
installed in a separate overlay; the original 4.57.6 Laya environment was unchanged.
Both official checkpoints were verified against pinned HF hashes. Offline inference.

Compared with the existing adapter's 15 frozen observations:

- Exact input token sequences matched **15/15**.
- Selected identities matched **15/15**.
- Maximum probability difference: **0.0000470206**, below the preregistered
  0.0002 tolerance and explained by SDK four-decimal output rounding.

This supports compatibility for the tested inputs, including the explicit RoPE
translation, tokenizer loading, complete state-dictionary load and scoring head.
It does not prove equivalence for every possible input or longer context window.

## Input rewrite and results

The projected arm retained task requirements, candidate implementation/audit
identities, cohort, acceptance counts, total cost and total time. It replaced
repetitive JSON provenance with readable sentences, used one strategy-selection
question and descriptive candidate text, and explicitly described missing-evidence
abstention. The three task-demand cases, including Portuguese, stayed unchanged.
All variants were frozen before the new inference. No failed case was relabeled
or retried with another prompt. This is a post-hoc diagnostic, not held-out quality.

| Official path and arm | Correct | Order-stable routing groups | Required boundary abstentions |
| --- | ---: | ---: | ---: |
| Option-Marker, original requests | 6/15 | 0/6 | 0/4 |
| Option-Marker, descriptive rewrite | 4/15 | 1/6 | 0/4 |
| SDK-default NLI, descriptive rewrite | 6/15 | 5/6 | 2/4 |
| Option-Marker, public classification controls | 10/10 | Not tested | Not tested |
| SDK-default NLI, same public controls | 10/10 | Not tested | Not tested |

The NLI path correctly abstained on unknown costs in both orders and chose faster
equal-cost/equal-quality delivery in both orders. It failed cohort-mismatch
abstention and most implementation/effort choices. Both paths misclassified the
Portuguese concurrency task as insufficient. Stable wrong answers do not count
as accuracy.

Controls were five support-department and five email-intent cases, selected by
fixed indices from the upstream benchmark file before execution. They cover clear
category assignments, rather than cost optimization. These public examples may
overlap training; 10/10 is a sanity check, not an independent qualification.

Exactly **65** public-primitive calls completed: 40 marker, 25 NLI. No refusals,
truncation, model errors or surviving processes. Median projected-request latency
was 2.683 s for marker and 5.253 s for NLI. Public-control medians were 0.921 s and
2.176 s. Sampled process-tree RSS peaks were 3.27 GiB and 1.86 GiB respectively.
Cold constructors took 20.138 s and 13.577 s; these reference loads exclude the
adapter's separate artifact-hashing step. CPU only; GPU services were untouched.

## Interpreting the published comparisons

The [linked independent v1/v2 benchmark](https://github.com/jabr/classifier-benchmark/blob/627f3013f8a9a8cba94a3954d1ad061fe2cac0a6/results/v1v2-summary.md)
does show Von ahead of Laya on aggregate: v2 macro 66.7% versus 58.3%. It also
reports substantial domain-shift failures and strong results for clear taxonomy
routing. Its adapter uses NLI, not the Option-Marker path. It does not include
OpenJEV in that comparison. The model card's Option-Marker/Qwen table is a separate
claim with a different setup; these rows cannot be treated as one matched study.

The upstream `run_comparison.py` also prints an OpenJEV row as literal constants,
without invoking OpenJEV in that script. That row alone is not a reproducible
head-to-head measurement. Our small local screen likewise cannot overturn or
confirm broad public rankings. It only supports the bounded deployment decision.

For our ecosystem, clear intent/category classification is a plausible future
scope. Numeric quality floors, cost comparability, quotas and GPU eligibility
should remain deterministic admission rules; winning a semantic score is not
evidence that all those constraints were satisfied.

## Evidence and reproduction

Use `tools/analysis/audit_von_contract.py --help` with the pinned upstream checkout,
local checkpoint directory, frozen protocol and a fresh output path. Run each
backend separately under the retained supervisor: 600-second deadline, two CPU
threads, RAM admission and descendant cleanup. No dependency upgrades to production.

[Preregistration](evidence/von-review-20260921/PRE_REGISTRATION.md),
[full inputs](evidence/von-review-20260921/PROTOCOL.json),
[parity](evidence/von-review-20260921/PARITY.json),
[summaries](evidence/von-review-20260921/SUMMARY.json),
[marker observations](evidence/von-review-20260921/marker-RESULT.json),
[NLI observations](evidence/von-review-20260921/nli-RESULT.json).
Original receipts remain unchanged. Source snapshots, dependency installation log
and console logs are retained under
`.reconciliation/evidence-store/reports/von-review-20260921`.
