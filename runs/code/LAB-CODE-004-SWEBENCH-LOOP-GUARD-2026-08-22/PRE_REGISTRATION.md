# LAB-CODE-004 SWE-bench loop-guard preregistration

Status: **FROZEN BEFORE GENERATION**  
Date: 2026-08-22

## Hypothesis

LAB-CODE-003 resolved every submitted patch (5/5), while the other five trajectories repeated a
command or materially equivalent search until the 40-call limit. A minimal anti-loop system
instruction should increase nonempty submissions without reducing resolved patches on the same
ten-instance bounded panel.

## Single treatment lever

Only the mini-SWE-agent `system_template` changes. It forbids more than two successive identical
commands and directs the model to choose a substantively different action, implement, or submit after
two repeated results. Model artifact, runtime, dataset and ordered IDs, base task prompt, tools,
temperature 0, seed 42, thinking-off mode, 2,048 output tokens, 40-call limit, timeouts, one trajectory,
Docker images and official evaluator remain identical to LAB-CODE-003.

## Dependency-gated execution

1. Run the first three frozen IDs in order. They contain two prior resolved controls and the first prior
   loop failure (`django__django-13401`).
2. Open the remaining seven only if all first three submit nonempty patches within 40 calls.
3. Evaluate all ten nonempty/empty predictions with the official SWE-bench harness. Empty predictions
   remain model failures in the denominator; no selective retry and no leaderboard submission.

## Frozen decision rule

- **PROMOTE LOOP GUARD:** at least 6/10 resolved, at least 6/10 nonempty submissions, and no regression
  below 4/5 resolved among the five LAB-CODE-003 submitters.
- **HOLD:** submission rate improves but the promotion rule is not met.
- **REJECT:** Stage A fails, submission rate does not improve, or prior submitter resolution regresses
  below 4/5.

Hashes and exact execution receipts are recorded in `FROZEN_RECEIPTS.json` before generation.
