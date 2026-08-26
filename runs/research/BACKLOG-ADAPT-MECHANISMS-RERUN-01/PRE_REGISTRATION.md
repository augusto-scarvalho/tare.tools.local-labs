# BACKLOG-ADAPT-MECHANISMS-RERUN-01 preregistration

Task: Fresh causal rerun of ADAPT-01 through ADAPT-05 mechanisms
Evidence class: `model_training`

## Hypothesis

At fresh seed 20260827, all five historical mechanisms can be reproduced as physical Qwen3.5-0.8B training/evaluation runs: four LoKr scale arms, four matched module-targeting arms, one soft-prompt arm, three prior-preservation arms, and a composite built only from the fresh disjoint MLP/attention adapters. Each historical verdict is retained only if the fresh matched behavioral result supports it; otherwise it is classified as a false positive or false negative.

## Frozen inputs

- `runs/research/ADAPT-01A-LOKR-SCALE-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/ADAPT-03-SOFT-PROMPTS-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/ADAPT-04-PRIOR-PRESERVATION-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/ADAPT-05-MODULAR-MERGING-2026-08-25/PRE_REGISTRATION.md`
- `workloads/gsm8k.jsonl`
- `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl`

- Official local checkpoint: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe`; config and tensor hashes are captured before training.
- Teacher traces: `runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json`.
- The original 32 held-out math and 16 protected-QA panels are used by every mechanism with seed 20260827.
- Historical inputs listed above are hashed before execution.

## Command

```powershell
python tools/research/run_adapt_mechanisms_rerun.py --outdir runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01
```

## Factors

- ADAPT-01: LoKr at 128, 384 and 640 steps plus the 384-step lower-learning-rate arm and base.
- ADAPT-02: all-linear, attention-only, MLP-only and QV-gate LoKr arms plus base, all trained for 384 steps.
- ADAPT-03: eight-token learned soft prompt for 384 steps.
- ADAPT-04: lambda 0.0, 0.2 and 0.5 prior-preservation arms for 640 steps plus base.
- ADAPT-05: disjoint static composite constructed only from the fresh ADAPT-02 MLP and attention adapters.
- All 16 evaluated arms/controls use the same 32 math and 16 QA records, yielding 768 scored generations. Training uses bfloat16 on RTX 3090 and seed 20260827.

## Acceptance gates

- `mechanism_coverage`: `fresh_mechanisms_completed eq 5`
- `training_coverage`: `fresh_training_arms eq 12`
- `evaluation_coverage`: `fresh_scored_generations eq 768`
- `seed_control`: `fresh_seed_verified eq 20260827`
- `independent_aggregate`: `independent_score_match eq 1`
- `artifact_identity`: `hashed_adapter_artifacts ge 13`
- `service_restore`: `original_service_restored eq 1`
- `embedding_integrity`: `embedding_health eq 200`

## Abort conditions

- Stop the inference service through root systemd before training, keep embedding 8081 healthy, and restore the exact original unit afterward.
- Abort on input drift, missing fresh adapter/checkpoint, fewer than 768 scored generations, seed mismatch, scorer mismatch, CUDA OOM without a preregistered retry, or failed service restoration.
- A probe's scientific rejection exit code is accepted only when its complete result JSON exists; crashes or incomplete outputs abort the packet.
- No arm, threshold, seed, panel, learning rate or step budget may change after observation.

## Allowed claims

- `ADAPT01_05_MECHANISMS_REPRODUCED_R1`
- `ADAPT01_05_MECHANISMS_MIXED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.

Training loss alone, cross-seed repeatability, production promotion, general model capability and unpaired causal claims are forbidden.
