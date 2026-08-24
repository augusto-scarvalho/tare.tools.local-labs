# Remaining experiment register — 2026-08-24

This register supersedes the 2026-08-22 queue for current execution state.

## Ready for execution

None. All dependency-free non-soak tasks have either run or stopped at their
frozen dependency gate. Reliability soaks remain explicitly excluded.

## Blocked on a concrete trigger

| Priority | Item | Current status | Required trigger |
|---:|---|---|---|
| 1 | MTP persistence root cause | `BLOCKED_MECHANISM` | New falsifiable cache-lifecycle hypothesis with invariant controls |
| 2 | ThinkingCap Qwen3.8 | `BLOCKED_UPSTREAM` | Official weights plus a 3090-fit artifact |
| 3 | ThinkingCap legacy MTP identity | `BLOCKED_IDENTITY` | Receipt identifying the exact local digest |
| 4 | Third-party quantizer builds | `UNKNOWN_BUILD` | Exact publisher build receipts |
| 5 | Human-judge calibration | `BLOCKED_HUMAN_INPUT` | 50–100 frozen blind human labels |
| 6 | RetNet official checkpoint | `BLOCKED_UPSTREAM` | Official Microsoft/TorchScale checkpoint |

## Newly closed

- Driver 591.86 post-reboot qualification: closed, stable, baseline restored.
- HauhauCS agent/tool core: 8/8; vanilla 8/8; Fable 7/8 conservative
  no-status-call failure; larger matrix stopped by gate.
- HauhauCS GSM8K-200: 191/200, but 8 truncations; `MATERIAL_MATH_LOSS`
  under the frozen bounded-response contract. Fable remains the broad default.
- FastMTP: `NO-GO BEFORE INSTALL`; prerequisite termination/default gate failed.
- RWKV7: license unblocked, then `HOLD_QUALITY` at 13/48; no serving stack added.

See `BLOCKER_REVALIDATION_2026-08-24.md` for source-backed trigger evidence.
