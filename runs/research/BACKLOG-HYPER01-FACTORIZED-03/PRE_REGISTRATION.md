# BACKLOG-HYPER01-FACTORIZED-03 preregistration

Task: Close the HYPER-01 resource-fidelity frontier at the maximum admissible factor rank
Evidence class: `proxy_realization`

## Hypothesis

Rank 135, the largest integer factor rank whose exact FP32 parameter storage
does not exceed 20 MiB, will close the gap observed at rank 32 and achieve
worst-seed mean physical-delta cosine at least 0.95 across five seeds. Failure
of fidelity or any other frozen gate closes this factorized HYPER-01 family.

## Frozen inputs

- `runs/research/BACKLOG-HYPER01-FACTORIZED-02/raw/receipt.json`
- `runs/research/BACKLOG-HYPER01-FACTORIZED-02/REVIEW.json`
- `runs/research/BACKLOG-HYPER01-FACTORIZED-02/raw/generator_states.safetensors`
- `runs/research/BACKLOG-HYPER01-FACTORIZED-02/raw/generated_tensors.safetensors`

- Admission: `be40adf342d0f311b0115a0c93be89f4ef1ee20a7384c8c80bbec7f763af91fa`.
- R2 receipt and independent review:
  `0662336e28e49661f56920470bf0c5a5fb06572362d5890f23659f6359d54028`,
  `3b4e7ff3b6bb4fc3eb89cbbd2f6da9f3365e589c9ffe3776616214843a2272ef`.
- R2 retained states and generated tensors:
  `efdf974bdc05aa6ffcdfbd074c5300f1c7e338314195110ca342558f48b71268`,
  `4121cb87fbed81aeb3702ac9e547802c970a603d94688b337fc034843a15566c`.
- The four physical checkpoint hashes remain the R2/R1 frozen identities:
  `ef5bec8822e856883eaec930d2b851892bb6b681bde1fda5f76005667adbf1a2`,
  `174832aa1bd25cbc5ed7f0ff717ad253ec94e2c23edc82e6f828ceadeed566b7`,
  `56ff9be8c5ac0876389cf12fe23a2ac301eac7c99cef977fa455b76f5817a2e6`,
  `dc696b7553cf8e4d920f8554ec4e3dee484a04da374ef0d54bcb48160044050a`.

## Command

```powershell
python tools/research/run_hyper01_factorized_r3.py --outdir runs/research/BACKLOG-HYPER01-FACTORIZED-03
```

## Factors

- Frozen architecture: 64 -> 256 -> 512 -> 135 -> 36,864, with exact FP32
  storage `19.95461654663086 MiB`; rank 136 would require
  `20.097198486328125 MiB`, so rank 135 is the discrete capacity frontier.
- Five deterministic seeds `20260824..20260828`, 3,000 cyclic AdamW steps per
  seed, learning rate 0.005, four physical targets and the same one-hot codes.
- All five states, 20 generated A/B pairs and 1,000 CUDA-event timings are
  retained. A separate process reopens the tensors and produces the 20
  acceptance cossines. GPU/CPU float32 disagreement is diagnostic only; the
  independent retained-byte scorer supplies the frozen acceptance values.
- MainPID, restart count and HTTP 8080/8081 must remain unchanged. No service
  stop is required and the worker must fit beside the qualified gateway.
- This is the terminal rank under the historical resource ceiling. No further
  rank/step successor is justified regardless of outcome.

## Acceptance gates

- `physical_targets`: `physical_adapter_targets eq 4`
- `target_distinctness`: `distinct_target_deltas eq 4`
- `completed_seeds`: `completed_seeds eq 5`
- `latency`: `median_synthesis_latency_ms le 5.0`
- `worst_seed_fidelity`: `worst_seed_mean_cosine ge 0.95`
- `overhead`: `generator_fp32_storage_mb le 20.0`
- `retained_states`: `retained_seed_states eq 5`
- `independent_rows`: `independently_recomputed_cosines eq 20`
- `service_recovery`: `service_and_embedding_unchanged eq True`

## Abort conditions

Abort on any frozen hash mismatch, missing/nonfinite or nondistinct target,
less than 2 GiB free GPU memory, missing seed/state/generated tensor/timing,
independent scorer failure, nonfinite metric, service restart, endpoint health
loss, worker failure, incomplete harness seal or incomplete provenance. No
rank, steps, seed, optimizer or threshold changes are allowed after launch.

## Allowed claims

- `HYPER01_MAX_RANK_PHYSICAL_FIT_R3`
- `HYPER01_RESOURCE_FIDELITY_FRONTIER_CLOSED_R3`

Claims outside these codes are forbidden even if a metric looks favorable.
No unseen-task, whole-adapter, end-to-end serving, production or direct-target
compression claim is allowed.
