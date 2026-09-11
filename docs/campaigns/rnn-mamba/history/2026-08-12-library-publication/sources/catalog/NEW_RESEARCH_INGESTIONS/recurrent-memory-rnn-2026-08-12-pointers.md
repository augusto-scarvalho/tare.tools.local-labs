# Research Pointers — Recurrent Memory / RNN — 2026-08-12

**Authority:** `RESEARCH_ONLY_NO_IMPLEMENTATION_AUTHORITY`

These are continuity pointers, not tare.tools product backlog.

## OPEN / authorized research frontier after independent RNN-06T2 audit

1. **Realistic high-interference long-context operating point**  
   Authorized for discovery on the same exact official Mamba-2 checkpoint. Establish task competence with a target-agnostic compressed/RAG control before interpreting full-context failure. A discovery cell is not confirmatory evidence; any positive operating point must be frozen and tested on a fresh RNN-07B set.

2. **Economics semantic closure**  
   Repair the RNN-06T2 economics output-domain mismatch: FINAL arms return scored token IDs while RECOVERY returned scored-vocabulary column indices. Re-run economics only with identical answer domains and randomized/interleaved timing cycles. Keep marginal recovery overhead conditional on the step path separate from full deployment cost vs the fused baseline.

3. **Adaptive temporal-state selection mechanism**  
   RNN-06T2 qualifies the frozen `MAX_CONFIDENCE` selector in the exact wide synthetic regime. Test whether the `NOT_YET_WRITTEN ↔ SEEN_AND_RETAINED ↔ ALREADY_FORGOTTEN` interpretation transfers to realistic workloads without target/gold-aware snapshot selection.

4. **Batch-shape numerical portability**  
   Investigate why official Mamba state differs across batch shapes (`max_abs_diff=0.5` observed historically). Fixed-batch request isolation is qualified; batch-shape portability is not. Determine future serving relevance before making it a runtime requirement.

5. **Historical-state memory vs current-state expansion**  
   Compare historical recovery with StateX-like current-state expansion only if the historical path remains semantically useful but end-to-end economics are unattractive.

6. **GDN/hybrid transfer**  
   Qwen3.5-class GDN/hybrid transfer remains deferred until realistic Mamba evidence clarifies whether the recovered phenomenon matters outside controlled synthetic forgetting.

7. **Trained state-memory attention**  
   DART-like trained retrieval remains conditional on a persistent oracle/recovery gap that a simple frozen target-agnostic selector cannot close on a useful workload.

## CONFIRMED WITHIN EXACT EXPERIMENTAL SCOPE

- controlled unique-binding-load forgetting on the frozen synthetic Mamba regime;
- historical-state information presence;
- strong parameter-free historical recovery;
- official Mamba fixed-batch lifecycle;
- actual single-pass in-run historical capture;
- wide-target synthetic adaptive selection (`MAX_CONFIDENCE` vs prospectively selected fixed slot153: `+0.3125`, 95% CI `[0.2396, 0.3751]`, positive 3/4 strata).

These findings do not establish natural-workload generalization or production utility.

## PARK / do not reopen by tuning

- tested TPTT configuration;
- old synthetic dense post-hoc Memory Caching recipe;
- EXT3/B4-style repeated threshold hunting.

## DEFER

- Qwen3.6-35B-A3B deployment-family transplant;
- Qwen GDN/hybrid transfer until realistic Mamba evidence;
- INT8 historical archives;
- ReplaySSM;
- serving integration.
