# BACKLOG-SLX10-PACKED-RUNTIME-01 preregistration

Task: Materialize SLX-10 IQ2_XXS as immutable GGUF and live runtime
Evidence class: `serving_runtime`

## Hypothesis

An immutable IQ2_XXS GGUF physically derived from the frozen F16 Qwen3.5-0.8B will be at most 18% of the F16 file, reduce live process VRAM by at least 60%, retain at least 95% throughput, lose at most 2 percentage points of frozen-panel accuracy, and preserve at least 80% of exact outputs. Any failed gate rejects the historical extrapolation at this tested scale.

## Frozen inputs

- `runs/research/SLX-10-PHYSICAL-CODEC-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/SLX-10-PHYSICAL-CODEC-2026-08-25/RESULT.md`
- `runs/research/SLX-10-PHYSICAL-CODEC-2026-08-25/raw/receipt.json`
- `tools/probes/slx10_physical_codec_bakeoff.py`
- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-05/raw/qwen3.5-0.8b-base-f16.gguf`

- Historical prereg/result/receipt/probe: `f985a570fc485956cc7408077f1a07ae0e135e7c3d46e8d1e6b2a6948d0ed9b7`, `e33f412d2c3099844e10a202c80e41a9a65ada172c85df019b5f20876571469a`, `ff7365541552058c5a13d2475b1b47c1c9a7acb877d52ce0575f53fcb3dabb34`, `717f68023469179a65ad86cc4289de7b44670cf05c1de0c1fd31b87875a904d0`.
- Physical F16 source GGUF: `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-05/raw/qwen3.5-0.8b-base-f16.gguf`, SHA-256 `514133770c0e30367721334fb86a76a8647bf8ab4d51fedc62980ce86dda1ac1`.
- Quantizer/binary/model-source prompt identities are captured before execution; the first 32 unique base-arm math rows from the already frozen training sample file form the panel.

## Command

```powershell
python tools/research/run_slx10_packed_runtime.py --outdir runs/research/BACKLOG-SLX10-PACKED-RUNTIME-01
```

## Factors

- Use the deployed `llama-quantize` with `IQ2_XXS` and no imatrix to produce one immutable packed GGUF under this packet. Hash it before serving.
- Stop the original inference unit as root, serve F16 then IQ2 with identical binary, `ctx-size=4096`, four slots, flash attention, `n_predict=128`, greedy seed 0, and no prompt cache. Port 8081 stays untouched.
- Capture Linux server PID and `nvidia-smi` process VRAM after load, then run the same 32 prompt/gold cells in each arm. Record full responses, wall latency, predicted throughput, extracted GSM8K answer and correctness.
- File ratio uses physical byte sizes. VRAM reduction is `(F16-IQ2)/F16`; throughput uses total predicted tokens divided by total wall time; accuracy regression is F16 correct rate minus IQ2 correct rate; exact output rate is byte-identical response content across paired prompts.
- Terminate each temporary server and restore the immutable original systemd executable path/argv; require both health endpoints.

## Acceptance gates

- `packed_artifact`: `iq2_file_ratio le 0.18`
- `physical_load`: `loaded_arms eq 2`
- `memory_reduction`: `vram_reduction ge 0.6`
- `throughput`: `throughput_ratio ge 0.95`
- `quality`: `accuracy_regression le 0.02`
- `semantic_stability`: `exact_output_rate ge 0.8`
- `service_restore`: `original_service_restored eq 1`
- `embedding_integrity`: `embedding_health eq 200`

## Abort conditions

- Any hash/quantization/load/PID/VRAM/request failure, prompt panel drift, temporary route mismatch, 8081 failure, or inability to restore the original unit aborts.
- No thresholds, quantizer type, panel, decode control or metric definition may change after observation.

## Allowed claims

- `SLX10_IQ2_PHYSICAL_QUALIFIED_R1`
- `SLX10_FALSE_POSITIVE_CONFIRMED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.

No claim about 27B/35B capacity, other codecs, production deployment, or quality outside the frozen 0.8B panel is permitted.
