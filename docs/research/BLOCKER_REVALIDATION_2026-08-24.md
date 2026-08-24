# Residual blocker revalidation — 2026-08-24

## Bottom line

Seven residual items were rechecked against their exact unblock triggers. One
external trigger changed: the official RWKV7 publisher now explicitly licenses
the weights under Apache-2.0. Its newly authorized quality gate was executed and
failed 13/48, closing that item as `HOLD_QUALITY` without installing a server.

The other six triggers remain absent. There is no dependency-free non-soak GPU
experiment left to launch.

## Trigger matrix

| Item | Evidence checked | Current status | Next trigger |
|---|---|---|---|
| MTP persistence root cause | Repository-wide search found only the original intermittent `!`-oracle failure and clean reruns; no new mechanism packet or invariant set exists | `BLOCKED_MECHANISM`, unchanged | New falsifiable cache-lifecycle hypothesis with controls and invalidation rules |
| ThinkingCap Qwen3.8 | BottleCapAI's current official organization exposes four Qwen3.6 ThinkingCap repositories and no Qwen3.8 weights | `BLOCKED_UPSTREAM`, unchanged | Official Qwen3.8 ThinkingCap weights plus a 3090-fit artifact |
| RWKV7 deployment | Current official card/manifest assert Apache-2.0 and bind the same local weight hash; frozen 48-item gate then scored 13/48 | `LICENSE_UNBLOCKED / HOLD_QUALITY`, closed | A materially stronger official/post-trained RWKV7 checkpoint with a new role packet |
| ThinkingCap MTP identity | Local 17,221,641,152-byte SHA `b0987c4e...` still differs from pinned revision `f015d8b`; the publisher now offers integrated-MTP GGUFs but does not identify that old local content | `BLOCKED_IDENTITY`, unchanged; old artifact is non-promotion-relevant | Publisher/download receipt for the exact local content, or delete only under a separate cleanup authorization |
| Third-party quantizer provenance | No new local provenance receipt exists. Current model cards may describe quant type and MTP validation but still do not provide exact quantizer/llama.cpp build commits for the 31 historical artifacts | `UNKNOWN_BUILD`, unchanged | Publisher build receipts; collect hashes only for newly admitted or promotion-relevant artifacts |
| Human-judge calibration | No frozen blind human-label artifact exists locally | `BLOCKED_HUMAN_INPUT`, unchanged | 50–100 genuine blind human preference labels |
| RetNet official checkpoint | Microsoft TorchScale issue #99 remains open with zero comments and no development link; official repository still exposes construction code, not a pretrained checkpoint | `BLOCKED_UPSTREAM`, unchanged | Official Microsoft/TorchScale pretrained checkpoint |

## Current primary sources

- RWKV7 official model and license:
  <https://huggingface.co/RWKV/RWKV7-1.5B-20260805>
- BottleCapAI current model inventory:
  <https://huggingface.co/bottlecapai/models>
- BottleCapAI current integrated-MTP GGUF card:
  <https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF>
- Microsoft TorchScale RetNet checkpoint issue:
  <https://github.com/microsoft/torchscale/issues/99>
- Microsoft TorchScale implementation:
  <https://github.com/microsoft/torchscale>

## Exclusions preserved

The 24/48/72-hour reliability soaks remain excluded and incomplete. Nothing in
this revalidation reopens them. No community RetNet weight may be substituted
for the official-checkpoint trigger, and no model-generated preference verdict
may be relabeled as human calibration.
