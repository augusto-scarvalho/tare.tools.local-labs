# Qwen3.8 agent/tool regression — result

## Bottom line

`NO_MEASURABLE_AGENT_LOSS` for HauhauCS on the frozen core panel.

HauhauCS and vanilla each passed all 8/8 dispatch cases. Fable-TC passed 7/8.
The exclusive Fable failure was the unknown-outcome transfer case: it correctly
refused to retry the irreversible transfer, but answered in prose instead of
calling `check_transfer_status`. HauhauCS and vanilla made the required status
call with `transfer_id=tx-9001`.

This is a bounded local result, not a BFCL-comparable score.

## Frozen core result

| Arm | Pass | Elapsed | Blind retry observed |
|---|---:|---:|---:|
| Fable-TC | 7/8 | 9.609 s | no |
| HauhauCS aggressive | 8/8 | 10.765 s | no |
| Vanilla Qwen3.8 UD-Q4_K_XL | 8/8 | 9.688 s | no |

All seven remaining cases passed on all three arms: tool selection, nested
arguments, abstention, parallel dispatch, sequential dispatch, multi-turn
dispatch, and error recovery.

## Dependency-gate outcome

The pre-registration required all three arms to pass 8/8 before spending the
larger budget. Because Fable scored 7/8, LAB-AGENT-002-v2 (40 cells per arm) and
LAB-AGENT-003-v2 (16 cells per arm) were not launched. This avoids converting a
failed primary gate into an unplanned exploratory campaign.

## Evidence identity

- `fable-core.json`: SHA-256
  `f092064ddfc7e03348ff138cc642001e035c209d41b44472927e2a0f94eca27f`
- `hauhaucs-core.json`: SHA-256
  `dad5ca6337e6b86ae3aa50789729a1654a0fec0384e3d5ad2a8f6dc0d08cb4f2`
- `vanilla-core.json`: SHA-256
  `fb8e305226e7e6ba6c18e27f66f351a796f74b91f06e9d53c28eb86e12a6c8d5`
- HauhauCS and vanilla: engine b10165 commit `71676e46c`, 32,768 context,
  Q4 KV, MTP n3, parallel 1.
- Fable-TC: operational engine b10159 commit `068764d92`, 8,192 context,
  MTP n4.

## Final operational state

Verified after the experiment:

- `llm-inference.service`, `llm-embedding.service`, and
  `llm-locale-proxy.service`: active;
- 8080: Fable-TC real response `agent-restored-ok`;
- 8081: real embedding response with 768 dimensions;
- 8082: proxied real response `agent-restored-ok`;
- both experimental systemd drop-ins absent;
- Fan Control and MSI Afterburner processes active. Fan Control remains the
  owner of fan curves; Afterburner remains limited to the established GPU
  clock/voltage profile.

No reboot, commit, or push was performed.
