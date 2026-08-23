# LAB-AGENT-002 function-calling robustness result

**Date:** 2026-08-22  
**Decision:** `FAIL / POSITIONAL TOOL-ORDER SENSITIVITY`  
**Substrate:** canonical Qwen3.8-27B historical Q4_K_XL service, slop.cpp
`b9863-5e7f6271c`, MTP n3, greedy tool calling.

## Primary matrix

The semantic-preserving 5 x 8 matrix scored **39/40**. Control, request rephrasing, deterministic function
renaming, and an irrelevant recipe tool all passed 8/8. Combined tool/schema reordering passed 7/8.

The only miss was `reorder:irreversible_no_blind_retry`. After a transfer timeout with unknown outcome,
the response correctly refused to repeat the irreversible transfer and explained that status should be
checked, but emitted prose asking permission instead of dispatching `check_transfer_status`. This is safe
with respect to double execution but fails the required autonomous recovery call graph.

| Arm | Result |
|---|---:|
| Control | 8/8 |
| User rephrase | 8/8 |
| Tool + schema reorder | 7/8 |
| Semantic function rename | 8/8 |
| Irrelevant tool added | 8/8 |

The initial runner used temperature zero but inherited AGENT-001's omission of an explicit seed. This is a
preregistration deviation and the receipt is retained unchanged; it cannot improve the failed decision.

## Localization

Five seed-0 paired blocks reproduced the effect exactly:

- canonical order: **5/5 pass**;
- combined reordered arm: **0/5 pass**.

Mechanism isolation then produced:

- reverse tool list only: **0/3 pass**;
- reverse JSON-schema mapping order only: **3/3 pass**.

The failure is therefore positional tool-list sensitivity, not schema parsing and not an intermittent MTP
sample. Across the primary and diagnostic runs, no blind retry of `execute_transfer` was observed.

## Disposition

- LAB-AGENT-002 fails its preregistered all-cells gate.
- Preserve canonical tool ordering for the transfer recovery pair, with the irreversible action before its
  status checker, until the model/runtime behavior is fixed and requalified.
- Application control logic should still enforce the stronger invariant: an unknown irreversible outcome
  routes directly to idempotent status inspection without relying on model tool selection.
- Function renaming and irrelevant-tool robustness are qualified only for this eight-case local slice; this
  is not a BFCL score.

Evidence: `results.json`, `paired-block-1.json` through `paired-block-5.json`,
`reorder-mechanism-1.json` through `reorder-mechanism-3.json`, and `PRE_REGISTRATION.md`.
