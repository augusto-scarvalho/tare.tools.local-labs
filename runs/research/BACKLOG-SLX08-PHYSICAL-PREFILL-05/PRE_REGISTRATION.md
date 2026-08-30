# BACKLOG-SLX08-PHYSICAL-PREFILL-05 preregistration

Task: Repeat physical selected-block prefill with real TTFT and absolute semantic floors
Evidence class: `serving_runtime`

## Why R5 exists

R4 completed all 128 physical responses and restored the original gateway, but
the harness compared volatile `ExecStart` PID/time metadata and recorded a false
restoration failure. R4 also used one predicted token; both arms emitted only a
newline, so relative noninferiority was scientifically empty. R4 remains
ABORTED and immutable.

## Hypothesis

On the same immutable experimental build and Qwen3.8 artifact, retaining 8 of
16 frozen 256-token prompt blocks will improve paired median real time to first
content token by at least 1.10x, without p95 regression, while both arms achieve
at least 90% accuracy and the lower bound of the paired accuracy delta remains
at least -0.03.

## Frozen inputs

- R4 terminal: `089cf385fea79885906485a84697be27e1bbf32b66652ac0147a0bf2c9fa9271`
- R4 physical samples: `940a99adbd8b6a35fc009b991d819edc5bf9520c207d8361752108f9c820a861`
- R4 service identity: `fe3344e921fb94717bea6178db570cdaf64476d7a5196a4e63427f43bd615fab`
- R4 actual scores: `6edfae68ec0ae70e357034d108ec040e2d55ac0695709c764a9bf4cb45fa33f0`
- R4 runner: `42141bffd6c51635f1b0ec6e1ff3b531f4c7ee2b8eca933ee3371b673b86bf6d`
- R4 experimental binary: `4395a601202ec76bcaef1d10db97849a92b311d8c31e4afce4d8b961609807a1`
- Qwen3.8 model: `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`

## Command

```powershell
python tools/research/run_slx08_physical_prefill_r5.py --outdir runs/research/BACKLOG-SLX08-PHYSICAL-PREFILL-05
```

## Factors

- 64 paired prompts, each exactly 4096 tokens and 16 blocks.
- Same prompt-token SHA-256 in OFF and ON.
- OFF retains 16/16 blocks; ON retains 8/16, including first and final blocks.
- Greedy decoding with `n_predict=8`, `temperature=0`, `top_k=1`, `seed=0`.
- TTFT is host monotonic time from POST until the first streamed chunk with
  non-empty content. The stream must finish and expose final route telemetry.
- Pair order alternates OFF/ON and ON/OFF.
- Stable restoration identity compares systemd command argv, gateway model,
  health and embedding health; PID and start timestamp are explicitly volatile.

## Acceptance gates

- `physical_treatment`: `physical_selected_block_prefill_requests ge 64`
- `dense_control`: `physical_dense_prefill_requests ge 64`
- `route_observation`: `selected_block_route_observation_rate eq 1.0`
- `retained_fraction`: `median_retained_attention_fraction eq 0.5`
- `dense_semantic_floor`: `dense_accuracy ge 0.9`
- `treatment_semantic_floor`: `selected_block_accuracy ge 0.9`
- `semantic_noninferiority`: `paired_accuracy_delta_ci95_low ge -0.03`
- `ttft_gain`: `paired_p50_ttft_speedup ge 1.1`
- `tail_safety`: `paired_p95_ttft_speedup ge 1.0`
- `service_restore`: `original_service_restored eq 1`
- `embedding_integrity`: `embedding_health eq 200`

## Abort conditions

Abort on source, binary, model, prompt, route, streaming, service or restoration
ambiguity. A dense request labeled ON is invalid. Empty semantic parity cannot
satisfy the absolute quality floors.

## Allowed claims

- `SLX08_PHYSICAL_SELECTED_BLOCK_PREFILL_QUALIFIED_R5`
- `SLX08_PHYSICAL_SELECTED_BLOCK_PREFILL_REJECTED_R5`

Claims outside these codes are forbidden even if a metric looks favorable.
This treatment is server-side token-block compaction before dense prefill, not
a generic sparse-attention kernel or production recommendation.
