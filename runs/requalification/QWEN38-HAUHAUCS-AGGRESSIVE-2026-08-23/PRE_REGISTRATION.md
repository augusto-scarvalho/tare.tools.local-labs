# Qwen3.8 HauhauCS Aggressive requalification

## Objective

Determine whether the HauhauCS Aggressive Qwen3.8-27B variant is materially more
capable than the currently served Fable-TC while refusing or hedging materially
less often than vanilla/current Unsloth Qwen3.8. This is a candidate evaluation,
not an automatic promotion.

## Frozen artifact

- Repository: `HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF`
- Revision: `993a5971fda8f30dd1b7eb2654792ba4415c7460`
- Quant: `Q4_K_P`, chosen over IQ4_XS to prioritize intelligence.
- File: `Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf`
- Expected bytes: `17923393664`
- Expected SHA-256: `ba36dc3c2b2ff5e0aa5d71092a8894546996a6a119ae391803dda07cdc08516d`
- Release-manifest signature: Ed25519, verified before the large download.
- License asserted by the model card: Apache-2.0.

## Isolation

- Preserve Fable-TC's model, versioned engine, systemd drop-in, and receipts as the
  rollback baseline.
- Keep `llm-embedding.service` on port 8081 untouched.
- Start with the stock-compatible embedded MTP head on canonical engine
  `b10165 (71676e46c)`, MTP n3 and ubatch 512. Do not apply the third-party
  FastMTP patch or download its sidecar unless the target model first passes the
  quality and alignment gates.
- Use q4_0 K/V, one slot, FlashAttention, all GPU layers, Jinja, metrics, and the
  same sampler contracts used for Qwen3.8 comparisons.

## Dependency-gated ladder

1. **Identity:** signed manifest, exact bytes/hash, GGUF metadata and MTP tensors.
2. **Fit/runtime:** 32k canary first; health, non-empty final content, MTP activity,
   embedding health, no CUDA/kernel faults, and at least 2 GiB free VRAM.
3. **Qwen profile envelope:** attempt the familiar 131072-token profile only after
   the 32k canary; keep it only if it loads with at least 1 GiB free and both
   endpoints remain healthy. Otherwise search downward without changing the model.
4. **Capability:** deterministic compact code/reasoning gate, followed by the exact
   historical 60-case HumanEval+ slice only if the compact gate is non-inferior.
   Primary requirement: materially exceed Fable-TC's recorded 53/60 (88.3%) and
   target vanilla Qwen3.8's recorded 57/60 (95.0%).
5. **Alignment/persona:** reuse the A2 refusal/hedging panel and compare against
   recorded Fable-TC and aligned Qwen anchors. Desired region is lower balk/hedge
   than vanilla Qwen without incoherence, degeneration, or unconditional compliance
   on control prompts.
6. **Serving:** compare decode, prompt throughput, MTP acceptance, termination and
   output equivalence. Evaluate the third-party FastMTP patch only after all earlier
   gates pass, in an isolated source clone with patch review.

## Decision

- **Promote candidate:** capability beats Fable-TC, alignment is below vanilla Qwen,
  outputs terminate correctly, and runtime/VRAM are operationally acceptable.
- **Hold:** desired behavioral region is reached but context/VRAM or statistical
  evidence is insufficient.
- **Reject:** capability is not better than Fable-TC, behavior degenerates, or any
  integrity/runtime gate fails.

At the end, restore Fable-TC unless the evidence supports promotion and the user
explicitly chooses to keep the candidate serving.
