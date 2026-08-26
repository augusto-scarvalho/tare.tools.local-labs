# BACKLOG-NEGATIVE-KV-REAL-SCREEN-02 result

Status: `EXECUTED`  
Executor: Codex executor  
Independent review: pending AGY  
Date: 2026-08-25

## Outcome

None of the five AGY negative conclusions became a complete false-negative candidate under its original frozen conjunction of success rules when retested on actual Qwen3.5-0.8B weights and forward-pass activations.

The evidence-validity gates all passed: the aborted predecessor was frozen, 18 activation cells and 12 weight matrices were captured, all decisive tensors came from the frozen model, five candidates completed, independent aggregation matched and the serving baseline was restored. Candidate success gates produced mixed partial results, but every candidate failed at least one mandatory rule.

| Candidate | Real-model observations (median) | Frozen disposition |
|---|---|---|
| RSH-01 Fibonacci INT4 | MSE ratio 1.57094; SQNR gain -1.9616 dB; cosine 0.992141 | Negative retained; all three rules fail |
| REP-03 Hadamard INT4 | MSE reduction 27.775%; attention cosine 0.999456 | Negative retained; fidelity passes but 50% MSE rule fails |
| RSH-03 rank-4 residual | MSE recovery 3.179%; output cosine 0.999200; overhead 0.78125% | Negative retained; cosine/overhead pass but recovery rule fails |
| RSH-04 binary retrieval | Top-block recall 50.781%; retained fraction 25% | Negative retained; memory fraction passes but recall rule fails |
| REP-06 entropy precision | 7.796875 bits; attention cosine 0.972286; did not beat static INT4 in every cell | Negative retained; all three rules fail |

## Evidence boundary

- The screen contains 78 individual rows: 18 each for REP-03, RSH-04 and REP-06, and 12 each for RSH-01 and RSH-03.
- Inputs are three deterministic 4,096-token contexts assembled from the frozen GSM8K corpus.
- Full-attention layers are 3, 7, 11, 15, 19 and 23; actual Qwen dimensions replace the synthetic shapes assumed by the old probes.
- The sole randomized operation is the preregistered RSH-04 projection treatment, evaluated with three fixed seeds. It does not substitute for model input.
- The first attempt stopped before measurement on an object-binding error. Its logs and restoration receipt are frozen; the successor changed only that binding and task metadata.

## Claim limits

This result does not measure packed bytes, realized VRAM savings, native-kernel latency or serving throughput. It supports only the bounded conclusion that the five original negative mechanism decisions were not reversed on these real-model tensor cells. AGY must independently recompute hashes, cell aggregates and decision rules before authorizing `NEGATIVE_KV_REAL_SCREEN_VERIFIED_R2`.
