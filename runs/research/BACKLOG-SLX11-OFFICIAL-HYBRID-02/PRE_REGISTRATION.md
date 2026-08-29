# BACKLOG-SLX11-OFFICIAL-HYBRID-02 preregistration

Task: Qualify official Qwen3.5 hybrid topology with retained logits
Evidence class: `artifact_requalification`

## Hypothesis

The pinned official Qwen3.5 0.8B checkpoint declares and physically
instantiates exactly 18 GatedDeltaNet and six full-attention layers, and all 24
frozen next-token forwards produce finite logits. Retaining every logits vector
in a digest-bound safetensors bundle will let a separate frozen scorer reproduce
tensor coverage, finiteness, argmax and SHA-256 for 24/24 rows. Failure to
recompute any projection rejects the artifact-qualification claim even when the
worker reports success.

## Frozen inputs

- `runs/research/BACKLOG-SLX11-OFFICIAL-HYBRID-01/raw/receipt.json`
- `docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md`
- `tools/research/run_slx11_official_hybrid.py`
- `tools/research/slx11_official_hybrid_worker.py`
- `workloads/gsm8k.jsonl`

- admission: `7e199fdc5df92f333fa4e50281d789e770412924a59709409dd00c599ad24225`
- R1 receipt: `d539dda7fb419ae390be32eb82e1d2c34ac44a44845c5992d505da35cdc28caf`
- independent audit ledger: `a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04`
- R1 host runner: `43e5259b7fbbf6fdb733d40d696fe65ba44fb756e1ad5e4b2816fab48b25e049`
- R1 worker: `94e6b0fbae2aed4c97bbea0072a182b9dbf4df691caf80c08776c2de6ceb36ae`
- frozen GSM8K source: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`
- official checkpoint: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe`; config, tensor, index and tokenizer hashes must reproduce the R1 receipt before forwards.

## Command

```powershell
python tools/research/run_slx11_official_hybrid_r2.py --outdir runs/research/BACKLOG-SLX11-OFFICIAL-HYBRID-02
```

## Factors

One immutable official checkpoint; its declared 24-layer topology versus the
24 instantiated PyTorch modules; the first 24 unique frozen GSM8K prompts; one
deterministic no-grad next-token forward per prompt in bfloat16 on the RTX 3090.
The worker writes each full next-token logits vector to a safetensors bundle.
A separately frozen scorer reloads that file and recomputes per-key shape,
nonfinite count, min, max, argmax and canonical tensor SHA-256. The host runner
joins scorer projections to worker rows and derives decisive gates only from
the retained bundle. Gateway and embedding processes are read-only controls and
must preserve PID, restart count and health.

## Acceptance gates

- `official_artifact`: `official_checkpoint_identified eq 1`
- `hybrid_topology`: `hybrid_layer_types_verified eq 24`
- `recurrent_layers`: `physical_recurrent_layers eq 18`
- `attention_layers`: `physical_full_attention_layers eq 6`
- `live_forward`: `successful_live_forwards eq 24`
- `logits_bundle_coverage`: `retained_logits_tensors eq 24`
- `finite_outputs`: `recomputed_finite_output_rate eq 1.0`
- `logits_projection_match`: `recomputed_projection_match_rate eq 1.0`
- `runtime_unchanged`: `serving_process_unchanged eq 1`

## Abort conditions

Abort on source, checkpoint or panel drift; missing/extra/duplicate logits key;
declared/physical topology mismatch; scorer disagreement on shape, argmax or
tensor hash; any nonfinite count; fewer than 24 forwards; worker/scorer failure;
GPU OOM; serving PID/restart/health change; incomplete provenance; or missing
bundle binding in the receipt. Raw R1 evidence remains immutable. Historical
speed, recall, quality and production claims are never inferred.

## Allowed claims

- `SLX11_OFFICIAL_HYBRID_ARTIFACT_QUALIFIED_WITH_LOGITS_R2`
- `SLX11_OFFICIAL_HYBRID_ARTIFACT_REJECTED_WITH_LOGITS_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
