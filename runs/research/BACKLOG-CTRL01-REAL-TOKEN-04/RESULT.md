# BACKLOG-CTRL01-REAL-TOKEN-01 result

## Verdict

`FALSE_POSITIVE_CONFIRMED` pending independent AGY review. Historical promotion is not supported by this successor.

## Actual results

- Real llama-server outputs: `24`.
- Raw complete JSON validity: `1.000000`.
- Sidecar output complete JSON validity without repair: `0.750000`.
- Valid-token acceptance rate: `0.901639`.
- Valid-control exact preservation: `0.833333`.
- p50/p95 overhead: `20.072` / `34.850` microseconds/token.
- Pre-sampling runtime/logit binding found: `False`.
- Failed mandatory gates: `real_validity, valid_control_recall, valid_control_semantics, runtime_binding`.

## Interpretation and claim limit

The historical probe repaired filtered strings by appending `}` before parsing and operated on hand-built corrupted chunks. This successor applies no repair, uses exact tokenizer pieces for both real outputs and valid controls, and finds no production binding between the Python post-filter and the active sampler. It therefore cannot substantiate a guarantee of constrained decoding, even if some offline latency or validity submetrics pass.

Allowed pending claim: `CTRL01_FALSE_POSITIVE_CONFIRMED_R4`. This does not evaluate Python mode, semantic correctness, or a future grammar-integrated runtime.

## Evidence

- `raw/receipt.json`
- `raw/samples.jsonl`
- `raw/artifact_hashes.json`
