# A2 Stage-2 D1am - refusal-direction extraction

Explicit backlog authorization on 2026-08-22 opens the previously optional
purism leg described in `docs/campaigns/a2-ablation-merging/A2_STAGE2_PLAN.md`.

This first execution is dependency-gated:

1. Extract base-model residual activations for the frozen 128/128 train and
   32/32 disjoint validation pools.
2. Select a refusal direction/layer using bypass, induction and KL gates.
3. Measure base-to-ThinkingCap direction transfer.
4. Stop before editing weights if there is no eligible layer, if the direction
   fails the G0 behavior requirements, or if transfer is red.

Frozen model directories are `/home/augus/models/fp16/{base,tc,fable}`. Runtime
is the isolated `sglang-venv` PyTorch stack on one RTX 3090, with only auxiliary
embedding port 8081 resident. The authoritative thresholds, artifact formulas,
kill criteria and later-arm order remain those in `A2_STAGE2_PLAN.md`; this
packet does not weaken them.
