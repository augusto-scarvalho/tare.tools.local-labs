# Remaining experiment register — 2026-08-24

This register supersedes the 2026-08-22 queue for current execution state.
The material execution history, persistent artifacts, service restoration, and
host-side effects are consolidated in
[`../EXECUTION_CLOSEOUT_2026-08-24_25.md`](../EXECUTION_CLOSEOUT_2026-08-24_25.md).

## Ready for execution

No non-soak transcript item remains ready. ADAPT-00C admitted no behavioral
finalist, so ADAPT-01 cannot start. Reliability soaks remain explicitly excluded.

Completed intake gates:

- `BEE-L0-SOURCE-ARCHAEOLOGY`: `COMPLETE`; whole-fork transfer rejected.
- `SLX-01A-GAP-AUDIT`: `GAP_CONFIRMED`; open a receipt-first shadow packet,
  not a broad fork implementation.
- `SLX-02A-APEX4-PREFLIGHT`: `BLOCKED_PUBLISHED_CHECKPOINT`; official kernel
  build/correctness passed, but the pinned public checkpoint shards are
  internally truncated and the separate end-to-end package is unavailable.
- `ADAPT-00A-MECHANICS-PREFLIGHT`: `PASS`; target held-out loss improved
  38.62%, protected loss regressed 0.53%, clean reload matched exactly, and peak
  allocation was 4.88 GiB.
- `BEE-L2-KV-QUALIFICATION-DESIGN`: `DESIGN_COMPLETE`; the codec-independent
  staged pack and full-distribution scorer are ready, but execution requires a
  candidate with immutable physical format and backend-route receipts.
- `ADAPT-00B-GEOMETRY-MATRIX`: `SCREEN_COMPLETE`; six of seven arms passed.
  LoKr led target-loss improvement with 359,040 trainable parameters, IA3 was
  the smallest arm, and DoRA failed non-finite at step 0 under the frozen arm.
- `ADAPT-00C-BEHAVIORAL-FINALIST-PANEL`: `NO_ARM_PROMOTED`; LoKr led at
  15/32 exact target answers versus 4/32 base, but missed the 16/32 floor and
  natural-EOS gate. LoRA reached 10/32 and IA3 4/32.

See `BEELLAMA_SLOP_PEFT_ANALYSIS_2026-08-24.md` for reconciliation and ordering.

## Blocked on a concrete trigger

| Priority | Item | Current status | Required trigger |
|---:|---|---|---|
| 1 | `ADAPT-01A-TRACE-DISTILLATION` | `BLOCKED_BEHAVIORAL` | A new preregistered budget or scale hypothesis that produces an ADAPT-00C finalist |
| 2 | MTP persistence root cause | `BLOCKED_MECHANISM` | New falsifiable cache-lifecycle hypothesis with invariant controls |
| 3 | ThinkingCap Qwen3.8 | `BLOCKED_UPSTREAM` | Official weights plus a 3090-fit artifact |
| 4 | ThinkingCap legacy MTP identity | `BLOCKED_IDENTITY` | Receipt identifying the exact local digest |
| 5 | Third-party quantizer builds | `UNKNOWN_BUILD` | Exact publisher build receipts |
| 6 | Human-judge calibration | `BLOCKED_HUMAN_INPUT` | 50–100 frozen blind human labels |
| 7 | RetNet official checkpoint | `BLOCKED_UPSTREAM` | Official Microsoft/TorchScale checkpoint |

## Newly closed

- Driver 591.86 post-reboot qualification: closed, stable, baseline restored.
- HauhauCS agent/tool core: 8/8; vanilla 8/8; Fable 7/8 conservative
  no-status-call failure; larger matrix stopped by gate.
- HauhauCS GSM8K-200: 191/200, but 8 truncations; `MATERIAL_MATH_LOSS`
  under the frozen bounded-response contract. Fable remains the broad default.
- FastMTP: `NO-GO BEFORE INSTALL`; prerequisite termination/default gate failed.
- RWKV7: license unblocked, then `HOLD_QUALITY` at 13/48; no serving stack added.

See `BLOCKER_REVALIDATION_2026-08-24.md` for source-backed trigger evidence.
