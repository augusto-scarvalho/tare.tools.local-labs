# BACKLOG-SLX08-REAL-FIDELITY-01 result

Status: `EXECUTED`  
Executor: Codex executor  
Independent review: pending AGY  
Date: 2026-08-25

## Outcome

The rejected SLX-08 fidelity gate is a false-negative candidate. Across 12 real Qwen3.5-0.8B QKV cells, gathering the top 50% blocks identified by the computed scores achieved median last-token attention-context cosine `0.995449`, passing the frozen `>=0.95` gate. All six evidence and operational gates passed.

The old probe computed `selected_indices` but did not use them. It sliced the first half of K and V instead. It also based the decisive fidelity value on random QKV tensors. On the real cells, even the legacy first-half control reached median cosine `0.991834`; the corrected selected-block treatment was higher at `0.995449`.

## Scope

- Two deterministic 8,192-token frozen GSM8K contexts.
- Six full-attention layers per context: 12 cells total.
- Actual model Q/K/V projections with the installed Q/K normalization contract.
- Corrected and legacy arms share exactly the same real tensors and dense reference.
- Computed selected sets differed materially from the legacy prefix sets in every cell.
- Independent aggregation reproduced the worker metrics exactly.

## Claim limit

This reverses only the fidelity basis for the historical rejection. The successor did not integrate speculative prefill into a serving runtime and did not measure TTFT, so it cannot claim the original `>=1.40x` speed gate or production qualification. If AGY independently verifies the evidence, the admissible claim is `SLX08_FIDELITY_FALSE_NEGATIVE_CANDIDATE_R1`; a separate physical runtime successor is required to decide the complete SLX-08 hypothesis.
