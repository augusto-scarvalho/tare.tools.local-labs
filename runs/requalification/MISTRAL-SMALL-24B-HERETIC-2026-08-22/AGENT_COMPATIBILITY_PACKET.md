# Agent harness compatibility correction

Status: **FROZEN BEFORE CORRECTIVE RERUN**  
Date: 2026-08-22

The initial agent matrix returned HTTP 400 before generation for `sequential` because the Mistral
chat template requires tool-call IDs of length at least nine while the fixture uses `doc_1`. That cell
is invalid infrastructure evidence, not a model failure.

The corrective runner repeats all eight cases and changes only fixture identifiers by prefixing
`fixture_` consistently in assistant calls and matching tool results. Tool names, schemas, content,
order, validators, endpoint, temperature and seed remain unchanged. The original 7/8 threshold and
no-blind-retry requirement remain binding; no system mitigation is added.
