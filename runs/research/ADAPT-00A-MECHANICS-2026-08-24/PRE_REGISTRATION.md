# ADAPT-00A adapter-mechanics preflight - pre-registration

## Purpose and scope

This is a mechanics and retention gate, not the seven-geometry bakeoff. It asks
whether a small official base, the already-recorded ThinkingCap teacher corpus,
and the local RTX 3090 can produce a reloadable adapter with measurable target
learning and bounded protected-text regression.

Passing this smoke authorizes the equal-budget LoRA/DoRA/LoHa/LoKr/BOFT/IA3/
trainable-token comparison. It does not select a production adapter or claim
that the 0.8B result transfers to Fable.

## Frozen inputs

- Base: `Qwen/Qwen3.5-0.8B-Base`, immutable Hub revision recorded in `RESULT.md`
- Teacher receipts: `runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json`
- Teacher receipt SHA-256:
  `dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e`
- Prompt snapshot: `workloads/gsm8k.jsonl`
- Prompt snapshot SHA-256:
  `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`
- Protected text: repository `README.md`, tokenized into fixed next-token-loss
  blocks and hashed into the result
- Seed: 20260824
- Initial arm: LoRA, rank 8, alpha 16, dropout 0, all linear modules
- Budget: 24 optimizer steps, batch size 1, maximum sequence length 384
- Precision: BF16 parameters/activations; no quantization

## Frozen split

Teacher records are joined to questions by `task_id`, sorted, then shuffled by
the frozen seed:

- 128 records for training
- 32 disjoint records for target held-out loss
- remaining teacher records unused by this smoke

The protected panel is never used as an optimization target.

## Metrics and gates

The run records initial/final held-out target cross-entropy, initial/final
protected-text cross-entropy, finite training losses, trainable parameter count,
peak allocated VRAM, adapter files, and a clean adapter reload check.

The smoke passes only when all are true:

1. all losses are finite and at least one trainable gradient is nonzero;
2. target held-out loss improves by at least 1%;
3. protected-text loss regression is no greater than 15%;
4. the saved adapter reloads into a fresh base instance and reproduces finite
   target loss within 0.5% of the pre-save adapter;
5. peak allocated VRAM stays below 23 GiB.

Fail closed on OOM, incompatible model/package code, missing teacher/prompt
identity, non-finite output, or reload mismatch. Do not tune the thresholds
after seeing the result.

## Operational boundary

Download and environment setup may occur with the canonical service active.
Training stops only `llm-inference.service`, leaves the embedding service
active, and restores/verifies the Fable baseline immediately afterward. No
production config, Fan Control setting, or MSI Afterburner setting is changed.
