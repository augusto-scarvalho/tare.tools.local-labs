# ADAPT-00B seven-geometry matrix - pre-registration

## Question

Under one frozen data and optimization budget, which PEFT geometry can learn
the ThinkingCap-derived target while preserving unrelated base-model text, and
which methods are mechanically incompatible or dominated on this 0.8B hybrid
architecture?

This is a loss/retention efficiency screen. It is not a generation-quality,
concision, or production-transfer qualification.

## Common frozen budget

- Same base/revision, environment, teacher/prompt/protected hashes, seed, and
  128/32 split as ADAPT-00A.
- 24 optimizer steps, batch size 1, BF16, maximum length 384, AdamW 2e-4.
- No per-method tuning after results, no quantization, no retries with a more
  favorable seed.
- LoRA reuses the already-completed ADAPT-00A receipt because its configuration
  is byte-for-byte the common matrix configuration.

"Equal budget" means identical examples, optimizer steps, sequence ceiling,
precision, and single-GPU resource boundary. Intrinsic trainable parameter
counts are reported rather than rank-tuned into artificial equality; parameter
efficiency is part of the comparison.

## Frozen arms

| Arm | Configuration | Preflight trainable parameters |
|---|---|---:|
| LoRA | rank 8, alpha 16, all linear | 5,411,328 |
| DoRA | LoRA geometry plus magnitude direction | 5,811,264 |
| LoHa | rank 8, alpha 16, all linear | 10,822,656 |
| LoKr | rank 8, alpha 16, all linear | 359,040 |
| BOFT | block size 4, one butterfly factor, all linear | 1,505,856 |
| IA3 | K/V, linear-attention input/output, and MLP down routes | 239,616 |
| Trainable tokens | most frequent unique tokens in frozen train split, cap 4096 | 2,026,496 (1,979 realized token rows) |

The optional PEFT BOFT FBD CUDA extension failed to compile against this frozen
PyTorch/CUDA header combination. BOFT is therefore preregistered on PEFT's torch
fallback, with that physical route recorded; it cannot be compared on training
speed as though it used the extension.

## Per-arm gates

The same ADAPT-00A gates apply independently:

- finite losses and nonzero gradient;
- target held-out loss improves at least 1%;
- protected loss regression is at most 15%;
- clean reload differs from pre-save target loss by at most 0.5%;
- peak allocated VRAM is below 23 GiB.

Unsupported construction, OOM, non-finite loss, or reload failure is retained
as that arm's result. A failed arm does not prevent later arms from running.

## Interpretation

Among passing arms, report the Pareto frontier over target improvement,
protected regression, trainable parameters, and peak VRAM. Do not declare a
production winner from this screen. A follow-on generation panel is permitted
only for non-dominated arms and must measure exact GSM8K answer, termination,
concision, and an unrelated protected task.

## Operations

Run arms sequentially. Stop only `llm-inference.service`; keep embeddings
active. Restore and verify Fable after the full matrix even when one or more
arms fail. Do not touch Fan Control, MSI Afterburner, or production config.
