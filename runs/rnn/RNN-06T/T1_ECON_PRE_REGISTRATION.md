# RNN-06T — Section 4 ECONOMICS PRE-REGISTRATION (true end-to-end)

Frozen BEFORE economics outcomes. Capture is INCLUDED in the end-to-end metric (unlike RNN-06D's
intrinsic-only figure, which is additionally reported for comparison, labelled separately).

## Runs measured (batch 16, warm = median of ≥5; cold = first; compile = model load / first kernel)

- **FINAL-only (normal deployment)** — fused prefill (`mamba_split_conv1d_scan_combined`, no
  inference_params) over the 768-token context + one FINAL readout. The baseline a deployment pays
  without recovery.
- **FINAL-only (step path)** — the single-pass step trajectory to FINAL + one readout (isolates the
  step-vs-fused penalty from the capture/restore/readout overhead).
- **Recovery-enabled** — single-pass step trajectory with in-run capture at 4 boundaries + FINAL,
  K+1 restores+readouts, MAX_CONFIDENCE selection.

## Reported (split compile / cold / warm)

single-pass snapshot capture overhead, state copy overhead, GPU→host transfer (if used), CPU RAM,
VRAM, snapshot bytes, restore, query/readout, selection, total added latency, total wall-clock,
throughput. Derived: net recovery / MiB, net recovery / added ms, quality delta / total added
latency, snapshot-count Pareto. Quality delta and net recovery are taken from the 3B qualification
(MAX_CONFIDENCE vs FINAL: Δacc = +0.490; net recovery count).

## Frozen envelope + gate

`COST_ENVELOPE_ADDED_MS_PER_QUERY_WARM = 1000` ms (recovery is an opt-in accuracy enhancement, not the
hot path; added latency measured vs FINAL-only fused, batch-amortized per query).

`END_TO_END_RECOVERY_UTILITY`:
- **QUALIFIED** iff net_recovery_count > 0 AND quality_delta ≥ 0.05 AND added_ms_per_query_warm ≤ 1000.
- **COST_FAIL** iff net_recovery_count > 0 and quality_delta ≥ 0.05 but added_ms_per_query_warm > 1000.
- **NOT_QUALIFIED** iff net_recovery_count ≤ 0.

The intrinsic (capture-excluded) figure is reported for comparison but does NOT set the verdict.
