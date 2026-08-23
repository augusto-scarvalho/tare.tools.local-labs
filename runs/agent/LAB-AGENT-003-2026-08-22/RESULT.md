# LAB-AGENT-003 bounded stress and scale result

**Date:** 2026-08-22  
**Decision:** `PASS (BOUNDED LOCAL SLICE)`  
**Substrate:** canonical Qwen3.8-27B historical Q4_K_XL, slop.cpp `b9863-5e7f6271c`,
MTP n3, temperature 0, seed 0.

The corrected preregistered matrix passed **16/16**:

| Axis | Qualified levels | Result |
|---|---|---:|
| Distractor tool count | 2 / 8 / 16 / 32 | 4/4 |
| Exact parallel fan-out | 2 / 4 / 8 / 12 | 4/4 |
| Completed sequential depth | 0 / 2 / 4 / 8 | 4/4 |
| Irrelevant dialogue history | 0 / 4 / 8 / 16 turns | 4/4 |

`results.json` is preserved but **INVALID / SUPERSEDED AS A GATE**. Its only miss was sequential depth 0,
where the validator required starting token `root` but the prompt did not supply it. The model explicitly
identified the missing required value and called with an empty token; deeper cells passed because tool
history supplied tokens. The frozen correction added only the intended starting token to the shared prompt
and reran all 16 cells as `results.corrected.json`, which passed fully.

This qualifies the tested bounds, not arbitrary scale and not BFCL comparability. It also does not override
AGENT-002's positional tool-order failure for irreversible recovery.

Evidence: `PRE_REGISTRATION.md`, `results.json` (invalid retained), and `results.corrected.json` (active).
