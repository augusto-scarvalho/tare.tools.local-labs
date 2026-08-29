# Independent audit handoff: 2026-08-29 continuous wave

This queue contains five `EXECUTED` packets produced by the Codex executor
lineage. None is independently reviewed or promoted. The canonical state is
`config/research_backlog.json`; the machine-readable order is
`config/research_audit_queue_2026-08-29_continuous_wave.json`.

The reviewer must be a genuinely independent actor. Recompute gates from raw
evidence, verify receipt fingerprints and provenance inputs, and actively look
for both false positives and false negatives. Review one packet at a time. Do
not batch-sign or trust executor summaries.

## Review order

1. `BACKLOG-GATEWAY-ROUTE-STRESS-02`: verify 30 switches and 120 responses;
   distinguish HTTP/route success from eligible text content. Vision chat
   outputs are not vision-task evidence.
2. `BACKLOG-FLEET-REGRESSION-SCREEN-02`: recompute 448 rows, 224 greedy pairs,
   payload retention, service identity/restoration and final-state provenance.
3. `BACKLOG-FLEET-SEEDED-STABILITY-02`: recompute 288 rows and 192 repeat pairs;
   challenge whether reconstructed requests exactly match the executed
   temperature/top-p/seed contract and whether live artifact identity is strong
   enough for the allowed claim.
4. `BACKLOG-FLEET-CONTEXT-ENVELOPE-05`: inspect failed R4 first. Confirm that R5
   changes only the prompt digest contract, joins and reconstructs all 72 rows,
   and makes no new-inference claim.
5. `BACKLOG-FLEET-CONTEXT-INTERFERENCE-03`: inspect the aborted R2 terminal and
   traceback first. Confirm the non-recursive wrapper, 31-decoy construct and
   all 72 retained joins.

## Claim boundary

All five packets stop at `EXECUTED`. Passing executor gates is not independent
approval. The reviewer alone may write `REVIEW.json` and use legal pipeline
transitions. A digest mismatch, irrecomputable metric or ambiguous artifact
binding is a hold or rejection according to the frozen gate/claim contract,
not permission to repair evidence in place.

## Operational checks

```powershell
python tools/analysis/backlog_pipeline.py gate
python -m pytest -q
git status --short
```

At executor handoff, the watched repair wave ended `complete`, both corrected
packets were audit-ready, the backlog gate passed, and `next --json` returned
`null`. These are executor observations and must be refreshed independently.
