# BACKLOG-SLX11-OFFICIAL-HYBRID-01 preregistration

Task: Qualify the official Qwen3.5 hybrid recurrent-attention artifact for SLX-11
Evidence class: `artifact_requalification`

## Hypothesis

The locally pinned official checkpoint declares and physically instantiates exactly 24 language layers: 18 learned linear-attention/GatedDeltaNet layers and 6 full-attention layers, and completes 24 fresh finite forwards on the frozen GSM8K panel while the serving baseline remains unchanged. This qualifies only the existence and executable hybrid topology. It does not validate the historical synthetic 4.49x or 100% recall claims.

## Frozen inputs

- `runs/research/SLX-11-GRANITE-HYBRID-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/SLX-11-GRANITE-HYBRID-2026-08-25/RESULT.md`
- `runs/research/SLX-11-GRANITE-HYBRID-2026-08-25/raw/receipt.json`
- `runs/research/BACKLOG-GDN02-LEARNED-STATE-01/raw/receipt.json`

- Official local checkpoint: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe`; its config and tensor identities are frozen by the runner before execution.
- Frozen prompts: first 24 unique records in `workloads/gsm8k.jsonl`.

## Command

```powershell
python tools/research/run_slx11_official_hybrid.py --outdir runs/research/BACKLOG-SLX11-OFFICIAL-HYBRID-01
```

## Factors

- Parse `text_config.layer_types` without mutation and compare it to the physical PyTorch module classes.
- Run one deterministic no-grad next-token forward for each of 24 frozen prompts; record prompt token count, argmax token and logits hash.
- Capture checkpoint config/tensor SHA-256, package/GPU identity and before/after service state.

## Acceptance gates

- `official_artifact`: `official_checkpoint_identified eq 1`
- `hybrid_topology`: `hybrid_layer_types_verified eq 24`
- `recurrent_layers`: `physical_recurrent_layers eq 18`
- `attention_layers`: `physical_full_attention_layers eq 6`
- `live_forward`: `successful_live_forwards eq 24`
- `finite_outputs`: `finite_output_rate eq 1.0`
- `runtime_unchanged`: `serving_process_unchanged eq 1`

## Abort conditions

- Abort on artifact drift, a declared/physical topology mismatch, non-finite logits, panel drift, or serving-process change.
- The embedding service and active inference service must remain healthy; this experiment does not stop or mutate them.
- No threshold, prompt, topology classifier or scorer may change after observation.

## Allowed claims

- `SLX11_OFFICIAL_HYBRID_ARTIFACT_QUALIFIED_R1`
- `SLX11_OFFICIAL_HYBRID_ARTIFACT_REJECTED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.

Historical 4.49x speedup, 100% recall, dense-model superiority, production promotion and out-of-panel claims remain forbidden.
