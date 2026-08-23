# RNN-09 — official RetNet checkpoint reproduction blocked

Date: 2026-08-22

## Decision

The requested next step — reproduce a small **official** RetNet checkpoint — is dependency-blocked.
This is not a negative result for RetNet and does not change the 7/7 mechanism qualification.

## Evidence checked

- The official Microsoft RetNet directory documents architecture/configuration and points to the
  TorchScale implementation, but does not publish a pretrained RetNet checkpoint:
  <https://github.com/microsoft/unilm/tree/master/retnet>
- The official TorchScale repository exposes RetNet construction from configuration, but its model
  list/download material does not provide a RetNet pretrained checkpoint:
  <https://github.com/microsoft/torchscale>
- The still-open upstream request for a RetNet checkpoint has no published resolution:
  <https://github.com/microsoft/torchscale/issues/99>

## Boundary and next action

Do not substitute community weights under the official-reproduction label. Preserve the synthetic
mechanism result as `COMPLETE`, mark only the checkpoint stage `BLOCKED_UPSTREAM`, and move the
single-GPU recurrent-architecture lane to an official RWKV7 checkpoint that fits the RTX 3090.

