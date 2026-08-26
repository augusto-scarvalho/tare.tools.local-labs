# BACKLOG-CUDAGRAPH-SERVING-02 preregistration

Task: Causally validate CUDA Graph serving with explicit OFF and ON runtimes  
Evidence class: `serving_runtime`  
Executor: Codex executor  
Date: 2026-08-25

## Hypothesis

On the frozen Fable-TC serving tuple, CUDA Graph enabled by default will reduce deterministic decode wall latency relative to an otherwise identical runtime launched with `GGML_CUDA_DISABLE_GRAPHS=1`. Across a balanced four-block crossover, ON must preserve exact paired outputs, achieve median paired speedup of at least 1.10x, and avoid p95 tail regression.

This is a causal OFF/ON test. A warm request is never labeled a treatment merely because it follows a cold request.

## Frozen inputs

- Admission specification: `config/research_backlog_admissions/BACKLOG-CUDAGRAPH-SERVING-02.json`, SHA-256 `f436dbccf6c33e88241b7afba527e774a3a83fb9e334fffed0edaee619bc59d1`.
- Invalidation audit: `docs/AUDIT_2026-08-25_CODEX_INDEPENDENT_AGY_EXECUTION.md`, SHA-256 `e4364456156a3c2f015306d986192792fb1aa9ae9333b63a2237ec46e3ffc11f`.
- Prior service identity: `runs/research/BACKLOG-CUDAGRAPH-SERVING-01/raw/service_identity.json`, SHA-256 `e1ae69bd587bb266e0f5aadaf40555174234ea0e17885fe5087822345db6b315`.
- Prior claimed route: `runs/research/BACKLOG-CUDAGRAPH-SERVING-01/raw/effective_route.json`, SHA-256 `cddef60e4d0d7edce834ac50db8aadde1f18f5b246445f5759800ab7d4195b6a`.
- Prior invalid paired observations: `runs/research/BACKLOG-CUDAGRAPH-SERVING-01/raw/samples.jsonl`, SHA-256 `4e4a1f8bcde43f6cf45dfb98f00fe26790535c915eb1fbd885f4012879a6ba13`.
- Server binary: `/home/augus/opt/slop.cpp/b10159-068764d92-fable-tc/bin/llama-server`, 17,920 bytes, SHA-256 `5719c246ec3622ea1df3c3f498075879f12f1f70b969f8b591e87b3a1f3c8808`.
- CUDA backend: `/home/augus/opt/slop.cpp/b10159-068764d92-fable-tc/bin/libggml-cuda.so.0.17.0`, 63,388,824 bytes, SHA-256 `78ed3ef92d354a544231232f99a3c23b3a02a179daa7d21c5a3ea9ab6a811eb9`.
- Model: `/home/augus/models/merges/fable-tc-l1.0-Q4_K_M.gguf`, 16,810,714,400 bytes, SHA-256 `052c08ca13d75d8d88c9cc3f201d7bfa9167e2a1e69ad3e1e1f26ff73c1b390b`.
- GPU: NVIDIA RTX 3090, 24,576 MiB.
- The CUDA backend binary must contain the literal control `GGML_CUDA_DISABLE_GRAPHS` and CUDA Graph warmup/reuse strings before the service is stopped.

## Factors

### Treatments and block order

- OFF: transient server process environment contains exactly `GGML_CUDA_DISABLE_GRAPHS=1`.
- ON: transient server process environment does not contain `GGML_CUDA_DISABLE_GRAPHS`.
- Both treatments use the same binary, model, `LD_LIBRARY_PATH`, executable arguments and endpoint `127.0.0.1:18080`.
- Frozen block order: `B1=OFF`, `B2=ON`, `B3=ON`, `B4=OFF`.
- B1 and B2 evaluate prompt IDs 1–15; B3 and B4 evaluate prompt IDs 16–30. Thus every prompt has one OFF and one ON observation while treatment order is balanced.
- Every block launches a fresh transient systemd unit and performs four identical discarded warmup requests before the 15 recorded requests.
- Each request is deterministic: temperature 0, seed 20260824, `max_tokens=64`, no streaming, prompt cache disabled.
- Response comparison includes assistant content, reasoning content, finish reason and completion-token count. Timing fields are excluded.

## Command

```powershell
python tools/research/run_serving_cudagraph_benchmark_r2.py --outdir runs/research/BACKLOG-CUDAGRAPH-SERVING-02
```

The transient command is the deployed command with only the address and port changed to `127.0.0.1:18080`:

```text
/home/augus/opt/slop.cpp/b10159-068764d92-fable-tc/bin/llama-server -m /home/augus/models/merges/fable-tc-l1.0-Q4_K_M.gguf --alias fable-tc-l1.0 --host 127.0.0.1 --port 18080 -ngl 99 -fa on --ctx-size 8192 --spec-type draft-mtp --spec-draft-n-max 4 --jinja --metrics
```

The persistent `llm-inference.service` may be stopped only through systemd and must be restored afterward. Port 8081 must remain healthy throughout. Each transient block is managed by a uniquely named systemd unit and must be inactive before the next block begins.

## Metrics

- `response_mismatch_rate`: mismatched paired outputs divided by 30.
- `paired_wall_speedup_p50`: median of `OFF wall_ms / ON wall_ms` for the 30 prompt pairs.
- `on_vs_off_p95_regression`: `(ON p95 - OFF p95) / OFF p95`; negative is an improvement.
- Server-reported predicted milliseconds and tokens per second are recorded as supporting metrics but do not replace wall latency.
- GPU temperature, utilization, clock, power and memory telemetry are captured at each block boundary.

## Acceptance gates

- `treatment_identity`: `explicit_off_on_controls_verified eq True`
- `binary_identity`: `distinct_binary_hashes_across_treatments eq 1`
- `balanced_crossover`: `valid_abba_blocks eq 4`
- `semantic_parity`: `response_mismatch_rate eq 0`
- `paired_speedup`: `paired_wall_speedup_p50 ge 1.1`
- `tail_non_regression`: `on_vs_off_p95_regression le 0.0`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

- Any frozen input hash, model size, server argument or treatment environment differs.
- The persistent serving unit is not the captured Fable-TC tuple before stop.
- Port 8081 becomes unhealthy at any boundary.
- A transient unit cannot start, exposes the wrong PID environment, returns a 5xx/timeout, or remains active after its block.
- Any block produces fewer than 15 recorded responses or uses the wrong prompt IDs.
- The persistent service cannot be restored active and healthy with the same executable and argument vector.

A semantic mismatch does not abort the remaining safe blocks; it fails the semantic gate and is preserved for analysis.

## Allowed claims

- `SERVING_CUDAGRAPH_CAUSAL_QUALIFIED_R2`
- `SERVING_CUDAGRAPH_CAUSAL_REJECTED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.

This packet cannot attribute the effect exclusively to driver launch overhead, generalize beyond this frozen tuple, or promote a production configuration. The executor stops at `EXECUTED`; an independent actor must review it.
