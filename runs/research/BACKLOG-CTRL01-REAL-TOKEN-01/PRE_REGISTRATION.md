# BACKLOG-CTRL01-REAL-TOKEN-01 preregistration

Task: Requalify CTRL-01 on real model token streams and valid-token controls
Evidence class: `mechanism_research`

## Hypothesis

The historical CTRL-01 implementation will accept every tokenizer piece belonging to valid JSON, preserve valid documents exactly, convert at least 24 real model streams into complete valid JSON without external repair, remain below 500 microseconds/token at p95, and be bound into the runtime as a pre-sampling logit mask. Failure of any mandatory gate falsifies the historical promotion as stated.

## Frozen inputs

- `runs/research/CTRL-01-AST-SIDECAR-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/CTRL-01-AST-SIDECAR-2026-08-25/RESULT.md`
- `runs/research/CTRL-01-AST-SIDECAR-2026-08-25/raw/receipt.json`
- `tools/analysis/ast_grammar_sidecar.py`
- `tools/probes/ctrl01_ast_sidecar_probe.py`

- Historical preregistration SHA-256: `c07cb593242fdd22dda6dbe9058967da0df87204d75c3c8d3aa4ff14a5510946`.
- Historical result SHA-256: `cd4319c345bdaa8224da7d9782fde7ee8e2f861fe1b3c6caefb3041adf4b31eb`.
- Historical receipt SHA-256: `0f37ae1d3ff33286a193353731f864d699ce738734fd8cc5b5a55384c2cf2c7c`.
- Sidecar SHA-256: `3cb90b1b5aa5aacdff93b7a8b0cdc38e689099e0d1365989f00b7b34acbb1463`.
- Historical probe SHA-256: `9e7ed6d27936952f20bbb27f0fbcf6530f2ebbec24dbf46faf46de8c22feb669`.
- Live endpoint: `http://127.0.0.1:8080`, queried through `/v1/chat/completions` and `/tokenize` with tokenizer pieces returned by the active model.

## Command

```powershell
python tools/research/run_ctrl01_real_token.py --outdir runs/research/BACKLOG-CTRL01-REAL-TOKEN-01
```

## Factors

- Generate 24 deterministic real responses: eight JSON schemas, three fixed seeds each, temperature zero, maximum 128 generated tokens.
- No JSON response-format grammar is requested from the server; the sidecar is the treatment under test.
- Replay exact tokenizer pieces from `/tokenize` through the historical `ASTGrammarSidecar`.
- The filtered output must parse as-is. Appending braces, brackets, quotes, defaults, or any other repair is prohibited.
- Valid controls cover nested objects/arrays, negatives, decimals, exponents, booleans, null, escapes, Unicode and whitespace. Exact byte-for-byte preservation is mandatory.
- Measure per-stream microseconds/token and report p50/p95.
- Runtime-binding gate is true only if repository/runtime evidence shows the sidecar controls candidate logits before sampling. Offline post-filtering does not satisfy it.
- Record raw requests, raw responses, tokenizer pieces, accepted output, per-piece decisions, hashes, server health/model metadata and independent JSON parses.

## Acceptance gates

- `real_coverage`: `real_model_outputs ge 24`
- `real_validity`: `sanitized_complete_valid_rate eq 1.0`
- `valid_control_recall`: `valid_token_acceptance_rate eq 1.0`
- `valid_control_semantics`: `valid_control_exact_preservation_rate eq 1.0`
- `overhead`: `p95_overhead_us_per_token le 500.0`
- `runtime_binding`: `logit_mask_runtime_integrated eq True`

## Abort conditions

- Active server or tokenizer endpoint is unavailable or changes identity mid-run.
- Fewer than 24 real generations or any frozen control is omitted.
- Outputs are repaired after filtering or tokenizer pieces are replaced by synthetic chunks.
- Any historical source hash changes.

## Allowed claims

- `CTRL01_RUNTIME_QUALIFIED_R1`
- `CTRL01_FALSE_POSITIVE_CONFIRMED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.

No production, Python-mode or semantic-correctness claim is allowed.
