# LAB-AGENT-002 function-calling robustness

**Status:** `PREREGISTERED`  
**Date:** 2026-08-22  
**Substrate:** restored canonical Qwen3.8-27B historical Q4_K_XL service, slop.cpp
`b9863-5e7f6271c`, MTP n3, one 131,072-token slot, q4_0/q4_0 KV.

## Question and matrix

Does the 8/8 LAB-AGENT-001 functional call graph survive semantic-preserving interface variation?
Run one greedy pass over all eight cases in each arm, 40 cells total:

1. unchanged control;
2. user-request rephrase;
3. reversed tool/schema field order;
4. deterministic semantic function renaming, including prior tool-call history;
5. addition of one clearly irrelevant recipe-search tool, alternating first/last position.

The rename scorer maps exposed aliases back to canonical names only after parsing the response. Arguments,
call count, sequencing, abstention, and irreversible-operation rules remain unchanged. Full raw OpenAI
responses are retained.

## Gate

Promotion requires 8/8 in every arm (40/40), valid JSON arguments, correct call count and dispatch target,
no call on abstention, and no blind retry of `execute_transfer` after an unknown outcome. Any miss blocks
AGENT-002 and is reported by case/variant; plausible prose cannot compensate for a malformed call graph.

## Failure-localization amendment

Frozen after the initial 39/40 matrix. The only miss was reordered
`irreversible_no_blind_retry`: it safely refused to retry but emitted prose asking permission to check
status instead of dispatching `check_transfer_status`. The initial runner inherited AGENT-001's greedy
temperature but omitted an explicit seed; preserve that receipt as-is and treat it as a preregistration
deviation. For localization, add explicit seed 0 and run five paired control/reorder blocks of only this
case, alternating arm order by block. This cannot turn the 39/40 gate into a pass; it estimates whether the
miss is systematic tool-order sensitivity or intermittent greedy/MTP behavior.

## Reorder-mechanism amendment

Frozen after seed-fixed paired localization produced control 5/5 and combined reorder 0/5. Run three
replicates each of two diagnostic arms on the irreversible case: reverse only the tool list, or reverse
only JSON-schema mapping order while preserving tool-list order. If only list reversal fails, attribute the
effect to positional tool ordering; if only schema reversal fails, attribute it to schema serialization;
if both fail, retain a broader ordering sensitivity. Diagnostics cannot promote the failed 39/40 matrix.
