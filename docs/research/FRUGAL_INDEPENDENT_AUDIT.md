# Frugal independent audit

## Purpose

Independent audit exists to test whether an experiment supports the bounded
decision it promised. It is not a search for implementation perfection and it
must not turn every technical imperfection into a new experiment.

The auditor carries the burden of blocking. Before returning work for repair,
the auditor must identify a falsifiable defect, prove from evidence that it can
reverse the claimed result, and explain which product or research decision
changes. If that chain is absent, the finding narrows the claim or becomes
follow-up work; it does not block the result.

## Review order

Use this order for every new packet:

1. **State the promise.** What bounded outcome did the experiment promise, and
   which tare.tools decision would change if it were true?
2. **Recompute the decisive evidence.** Reopen retained samples and independently
   reproduce the gates that carry the claim. Do not spend the audit budget
   reproducing every incidental field first.
3. **Try to falsify it.** Attack the most decision-relevant assumption, scorer,
   control, identity or aggregation. Record evidence and outcome.
4. **Search for a false negative.** Check whether a stricter or alternative
   valid scorer, retained subgroup, control or recomputation rescues a result
   that looks negative. Auditor-induced false negatives are audit failures.
5. **Classify materiality.** Separate result-reversing defects from claim limits
   and non-blocking technical debt.
6. **Choose the smallest sufficient remedy.** Reuse retained evidence before
   requesting any new inference or training.

One strong falsification attempt is better than a long generic checklist. Add
more only when the decision remains genuinely uncertain.

## Materiality rule

| Level | Meaning | Disposition |
| --- | --- | --- |
| `RESULT_REVERSING` | The defect proves that the bounded promise is not supported and changes the business/research decision. | `BLOCKED` or `REJECTED`, with the smallest sufficient remedy. |
| `CLAIM_NARROWING` | The decisive result survives, but its scope must be smaller. | `APPROVED` with an explicit boundary; no rerun. |
| `NON_BLOCKING` | Editorial, packaging, telemetry or technical debt that cannot change the decision. | `APPROVED`; record optional follow-up. |

Examples of findings that do not block by themselves:

- a missing ephemeral lock file after a clean terminal;
- an unbound wrapper whose decisive samples, implementation and gates are
  independently reconstructable;
- a formatting or naming defect with no effect on treatment, score or claim;
- absent evidence for a broader claim that the packet never promised.

Examples that can block when proved:

- treatment telemetry shows that ON and OFF executed the same physical path;
- an independently recomputed decisive gate changes pass to fail;
- the scorer systematically labels incorrect outputs as correct and reverses
  the accepted conclusion;
- the measured artifact is not the promised model, adapter or runtime;
- the experiment omits a cost that its promised business benefit explicitly
  includes.

## Remedy ladder

Choose the first sufficient action:

1. accept the bounded result;
2. narrow the claim;
3. rescore retained evidence;
4. bind missing metadata or patch the auditor/scorer and recheck;
5. rerun only the missing cell or minimal discriminating panel;
6. run a full experiment again only when retained evidence is unusable and the
   result-reversing uncertainty cannot be resolved more cheaply.

`RERUN_FULL` therefore requires both an explicit justification and
`retained_evidence_reusable=false`. Convenience, aesthetic completeness and
uniformity with another packet are not sufficient reasons.

## Executable contract

Newly scaffolded packets use `local-labs-independent-review-v2`. The generated
`REVIEW.template.json` asks for only the decision-bearing fields:

- `promise`, `value` and `promise_met`: the promised outcome and why it matters;
- `falsification_*`, `false_negative_check` and `result_reversing`: one concise
  attack, its evidence and whether it actually overturns the result;
- `materiality`: `RESULT_REVERSING`, `CLAIM_NARROWING` or `NON_BLOCKING`;
- `remedy`, `smallest_remedy` and `retained_evidence_reusable`: the narrowest
  sufficient action and whether a rerun can reuse existing evidence;
- receipt and implementation bindings plus concise findings.

For v2 packets, independent review is mandatory not only for verification and
promotion, but also for `EXECUTED -> REJECTED` and `EXECUTED -> BLOCKED`.
Negative dispositions fail closed unless the review proves a result-reversing
failure with business impact. Historical v1 reviews remain valid and immutable.

## Compact reviewer prompt

```text
Audit the bounded promise, not implementation perfection. First state the
product/research decision this result is supposed to change. Recompute the
decisive gates, make at least one evidence-backed attempt to falsify the result,
and actively search for a scorer/control-induced false negative. Block or reject
only if you prove a result-reversing defect that changes that decision. Otherwise
approve with a narrower claim or record non-blocking debt. Reuse retained
evidence and request the smallest sufficient repair; a full rerun is the last
resort. Fill REVIEW.json using the v2 template and keep findings concise.
```

## Non-goals

The contract does not weaken frozen gates, permit executor self-review, excuse
fabricated evidence or promote proxy results as physical evidence. Frugality
reduces low-value audit work; it does not reduce independence or truthfulness.
