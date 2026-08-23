# LAB-CODE-005 duplicate-action guard result

Decision: **INVALID IMPLEMENTATION / SUPERSEDED BY LAB-CODE-005B**  
Date: 2026-08-22

Stage A produced the two deterministic control patches, but the guard reported zero blocks while
`django__django-13401` repeated the same command through the 40-call limit. Inspection showed that
the signature included mini-SWE-agent's fresh `tool_call_id`, so semantically identical actions never
compared equal. No model claim is drawn from this run and the remaining seven were not opened.

LAB-CODE-005B corrects only that implementation defect by signing the ordered command strings and
adds a self-test in which identical commands have different tool-call IDs.
