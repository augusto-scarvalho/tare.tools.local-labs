# BACKLOG-SLX08-PHYSICAL-PREFILL-04 preregistration

Task: Materialize selected-block prefill in the physical serving runtime
Evidence class: `serving_runtime`

## Hypothesis

An explicit selected-block prefill route retaining 50% of frozen attention
blocks will improve paired median time-to-first-token by at least 1.10x on the
RTX 3090 while the lower bound of the paired semantic-accuracy delta remains at
least -0.03 and p95 TTFT does not regress.

## Frozen inputs

- `runs/research/BACKLOG-SLX08-REAL-FIDELITY-03/raw/receipt.json`: `d2f76165a548e08c1713d44f976f3a1d5aa30158e6745b90bce0d72aa688bddb`
- `runs/research/BACKLOG-SLX08-REAL-FIDELITY-03/REVIEW.json`: `e07df1828f2a8f1f86f32138dcae53735273587683c29701a9ccddf46503a778`
- `runs/research/BACKLOG-SLX08-REAL-FIDELITY-02/raw/context_vectors.safetensors`: `859ea9e3088de4e1f354a51a3c5502fd845ac1289dd0ad7b83d8c4f35b76cc58`
- `docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md`: `a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04`
- `tools/probes/slx08_speculative_prefill_oracle.py`: `5b85dd266c3fc72ae47a7cabe6e5ae3246e4aab544e87e6ee7cd47eab81bdc37`

## Command

```powershell
python tools/research/run_slx08_physical_prefill_r4.py --outdir runs/research/BACKLOG-SLX08-PHYSICAL-PREFILL-04
```

## Factors

- Explicit runtime OFF control versus selected-block ON treatment from the same immutable slop.cpp build and model identity.
- 64 paired long-context requests per arm, interleaved to control drift, with fixed greedy decoding and identical prompt bytes.
- Physical route telemetry must bind every ON request to selected-block execution and report exactly 0.5 median retained attention fraction.
- TTFT uses the same first-token contract in both arms; semantic scoring and its paired 95% lower bound are frozen beforehand.
- Harness requires restoration and seals request, telemetry, binary and service evidence before watcher advancement.

## Acceptance gates

- `physical_treatment`: `physical_selected_block_prefill_requests ge 64`
- `dense_control`: `physical_dense_prefill_requests ge 64`
- `route_observation`: `selected_block_route_observation_rate eq 1.0`
- `retained_fraction`: `median_retained_attention_fraction eq 0.5`
- `semantic_noninferiority`: `paired_accuracy_delta_ci95_low ge -0.03`
- `ttft_gain`: `paired_p50_ttft_speedup ge 1.1`
- `tail_safety`: `paired_p95_ttft_speedup ge 1.0`
- `service_restore`: `original_service_restored eq 1`
- `embedding_integrity`: `embedding_health eq 200`

## Abort conditions

Abort before measurement if the immutable build lacks an independently
observable selected-block prefill route. Also abort on source/build/model
mismatch, route telemetry ambiguity, unequal prompts or decoding, fewer than 64
complete pairs, nonfinite timing, service collision, embedding health loss,
restoration failure or unsealed raw evidence. A dense request labeled ON is not
a treatment and must never be scored as one.

## Allowed claims

- `SLX08_PHYSICAL_SELECTED_BLOCK_PREFILL_QUALIFIED_R4`
- `SLX08_PHYSICAL_SELECTED_BLOCK_PREFILL_REJECTED_R4`

Claims outside these codes are forbidden even if a metric looks favorable.
The historical 1.40x claim, production readiness, unobserved route execution
and generalization beyond the frozen model/panel remain forbidden.
