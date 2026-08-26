# BACKLOG-HYPER01-REAL-ADAPTER-01 preregistration

Task: Retest HYPER-01 rejection against four physical LoRA checkpoints
Evidence class: `proxy_realization`

## Hypothesis

The HYPER-01 rejection may have been an artifact of randomly generated target adapters. Training the same two-hidden-layer hypernetwork recipe against four physical LoRA checkpoints will synthesize their matched layer-0 `gate_proj` deltas with mean cosine at least 0.95, median latency at most 5 ms and generator FP32 parameter storage at most 20 MB.

## Frozen inputs

- `runs/research/HYPER-01-CAPSULES-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/HYPER-01-CAPSULES-2026-08-25/RESULT.md`
- `runs/research/HYPER-01-CAPSULES-2026-08-25/raw/receipt.json`
- `tools/probes/hyper01_capsule_generator.py`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints`

- Admission SHA-256: `0884c22bd3b6ead236543cd9d97db708b608790409c3a458981b6fb086f51a9d`.
- Original preregistration/result/receipt/probe SHA-256: `776cf15eaa10e3ebf0a343b99afd5bbab6c96d31f036ded1fafa0bae98d42bc0`, `87a9d4a2768e961f5a3c9cba2b691c46cd1a49687854d375d61808e2ead57ea8`, `1cddbc6b59a3148803fb51cac537316ce9039cfb7f7c0ce872b1e2b150f1b342`, `a35bcd0877a35788e635305df0313d9e35e69f00acca36cfecbb3d49c09eb0e8`.
- Physical checkpoint weight SHA-256 values, in frozen target order: seed-20260824 answer `ef5bec8822e856883eaec930d2b851892bb6b681bde1fda5f76005667adbf1a2`; seed-20260824 trace `174832aa1bd25cbc5ed7f0ff717ad253ec94e2c23edc82e6f828ceadeed566b7`; seed-20260825 answer `56ff9be8c5ac0876389cf12fe23a2ac301eac7c99cef977fa455b76f5817a2e6`; seed-20260825 trace `dc696b7553cf8e4d920f8554ec4e3dee484a04da374ef0d54bcb48160044050a`.

## Command

```powershell
python tools/research/run_hyper01_real_adapter.py --outdir runs/research/BACKLOG-HYPER01-REAL-ADAPTER-01
```

## Factors

- Target module in every checkpoint: `base_model.model.model.layers.0.mlp.gate_proj`, with physical A shape 8 x 1024 and B shape 3584 x 8. The evaluated delta is `B @ A`.
- Four deterministic 64-dimensional task codes: one-hot indices 0-3 with remaining coordinates zero. Codes are metadata, not synthetic target evidence.
- Hypernetwork: Linear 64->256, GELU, Linear 256->512, GELU, Linear 512->36,864; split output into A and B with the frozen physical shapes.
- Training: seed 20260824, AdamW learning rate 0.005, 150 cyclic steps over the four targets, direct A/B MSE as in the original probe.
- Fidelity: cosine between generated and physical `B @ A`, mean over four targets. Latency: median per-target CUDA-event latency over 100 x four syntheses after warmup.
- Overhead: exact generator parameter count times four bytes, excluding optimizer, targets and CUDA allocator overhead, matching the original metric definition.
- This tests one matched physical module, not generation of the complete 24-layer adapter or generalization to unseen tasks.
- The RTX 3090 service is isolated/restored using the existing systemd contract; embedding remains healthy.

## Acceptance gates

- `physical_targets`: `physical_adapter_targets eq 4`
- `target_distinctness`: `distinct_target_deltas eq 4`
- `latency`: `median_synthesis_latency_ms le 5.0`
- `fidelity`: `mean_weight_delta_cosine ge 0.95`
- `overhead`: `generator_vram_overhead_mb le 20.0`
- `independent_recompute`: `independent_metric_recompute_match eq True`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

- Any frozen checkpoint or predecessor hash differs, any target key is absent, or the four target deltas are not distinct.
- Any random matrix substitutes for a target adapter.
- Architecture, training steps, learning rate, task order or thresholds change after measurement.
- CUDA execution, independent recomputation, provenance or service restoration is incomplete.

## Allowed claims

- `HYPER01_FALSE_NEGATIVE_CANDIDATE_R1`
- `HYPER01_NEGATIVE_RETAINED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
