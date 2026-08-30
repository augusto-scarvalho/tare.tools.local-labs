# BACKLOG-HYPER01-FACTORIZED-02 preregistration

Task: Retest HYPER-01 resource failure with a frozen factorized output head
Evidence class: `proxy_realization`

## Hypothesis

A rank-32 factorization of the oversized output projection will preserve a
worst-seed mean physical-delta cosine of at least 0.95 across five deterministic
training seeds while reducing exact FP32 generator parameter storage from
72.706 MiB to at most 20 MiB and keeping isolated synthesis latency at most
5 ms. Failure of any frozen gate retains the bounded HYPER-01 rejection.

## Frozen inputs

- `runs/research/BACKLOG-HYPER01-REAL-ADAPTER-01/PRE_REGISTRATION.md`
- `runs/research/BACKLOG-HYPER01-REAL-ADAPTER-01/RESULT.md`
- `runs/research/BACKLOG-HYPER01-REAL-ADAPTER-01/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints`

- Admission: `4773ccda02ad27395bb30caf75333d4ae6afe99081ab6ac263dd10a1109a4752`.
- R1 preregistration, result, receipt and independent review:
  `6969ca78bfcfa3e832895904c7dcea0f920f2e428a459051f2230b3b82da20f9`,
  `f88b3e2934caa5726eb80529bdfb225735e0a54241a5dd1320c638ebed53249d`,
  `60e9d83a46a06c65b28067c141b06d8bfdb9ba5009fbfbcdb1dce37afe2f2fa3`,
  `09073f38bb1d21e4ad2dc59f060ffb4a32c68288cda0d54848d3ac632a80ace5`.
- Physical checkpoint weights, in frozen target order: seed-20260824 answer
  `ef5bec8822e856883eaec930d2b851892bb6b681bde1fda5f76005667adbf1a2`;
  seed-20260824 trace
  `174832aa1bd25cbc5ed7f0ff717ad253ec94e2c23edc82e6f828ceadeed566b7`;
  seed-20260825 answer
  `56ff9be8c5ac0876389cf12fe23a2ac301eac7c99cef977fa455b76f5817a2e6`;
  seed-20260825 trace
  `dc696b7553cf8e4d920f8554ec4e3dee484a04da374ef0d54bcb48160044050a`.

## Command

```powershell
python tools/research/run_hyper01_factorized_r2.py --outdir runs/research/BACKLOG-HYPER01-FACTORIZED-02
```

## Factors

- Four immutable physical layer-0 `gate_proj` LoRA A/B targets are identical to
  R1. No random target or new checkpoint is permitted.
- Architecture is fixed to 64 -> 256 -> 512 -> 32 -> 36,864 with GELU after
  every layer except the output. Exact FP32 parameter bytes include biases.
- Five independent deterministic seeds (`20260824` through `20260828`) train
  for 1,200 cyclic steps each with AdamW, learning rate 0.005 and the same four
  one-hot task codes. The worst seed, not the best seed, gates fidelity.
- Every trained state and every generated A/B tensor is retained in safetensors.
  A separate scorer process reopens those bytes and all physical checkpoints
  to recompute all 20 B@A cosines. All 1,000 CUDA-event latency observations
  are retained, alongside peak worker allocation and reservation.
- The direct four-target storage baseline is 0.5625 MiB and must be reported.
  Passing the 20 MiB historical gate does not establish compression relative
  to directly retaining the four targets or generalization to unseen tasks.
- The worker shares the RTX 3090 with the qualified gateway without stopping
  services. MainPID, restart count and 8080/8081 health must remain unchanged.

## Acceptance gates

- `physical_targets`: `physical_adapter_targets eq 4`
- `target_distinctness`: `distinct_target_deltas eq 4`
- `completed_seeds`: `completed_seeds eq 5`
- `latency`: `median_synthesis_latency_ms le 5.0`
- `worst_seed_fidelity`: `worst_seed_mean_cosine ge 0.95`
- `overhead`: `generator_fp32_storage_mb le 20.0`
- `retained_states`: `retained_seed_states eq 5`
- `independent_recompute`: `independent_metric_recompute_match eq True`
- `service_recovery`: `service_and_embedding_unchanged eq True`

## Abort conditions

Abort before training on any source hash mismatch, missing/nonfinite target,
non-distinct physical deltas, less than 2 GiB free GPU memory, or unhealthy
8080/8081 baseline. Abort after launch on any missing seed/state/generated
tensor/timing, scorer disagreement above `1e-7`, nonfinite metric, service
restart, endpoint health loss, worker error or incomplete provenance. No rank,
step count, seed, threshold or optimizer change is allowed after observation.

## Allowed claims

- `HYPER01_COMPACT_PHYSICAL_FIT_R2`
- `HYPER01_COMPACT_NEGATIVE_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
No whole-adapter, unseen-task, end-to-end serving, production or target-storage
compression claim is permitted.
