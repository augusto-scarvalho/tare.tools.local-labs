# Host/WSL and canonical-build qualification

Frozen before execution on 2026-08-23.

## Objective

Measure the deployed Qwen3.8 service after the host/WSL stability changes, build the canonical
`slop.cpp` main revision in an isolated WSL ext4 clone, and compare the candidate against the
deployed binary before any production switch.

## Fixed workload

- Model: Qwen3.8-27B UD-Q4_K_XL.
- One slot, q4_0/q4_0 KV, flash attention, all GPU layers, MTP draft depth 3.
- Counterbalanced short/long prompts from `energy_phase_bench.py`.
- Five repetitions per prompt class, 128 forced decode tokens, cache disabled.
- Primary metrics: median prompt throughput and median decode throughput.
- Secondary metrics: TTFT, energy/token, peak power, peak temperature, and VRAM.

## Phases and gates

1. Record the live deployed build on port 8080 without changing the service.
2. Clone Windows canonical revision `71676e46c` into a separate ext4 source directory and build
   `llama-server` and `llama-bench` with the deployed Release/CUDA/sm_86/native/OpenMP settings.
3. Run build self-identification and bounded CPU/CUDA smoke tests before any service interruption.
4. Stop `llm-inference.service` through systemd, enter LAB mode, run the candidate on port 8092,
   and execute the identical workload. The pre-existing down state of port 8081 is recorded but
   the embedding endpoint is not mutated by this packet.
5. Restore SERVE mode and the original systemd service even if the candidate fails.

Candidate performance qualifies only if neither primary axis regresses by more than 3%. A default
runtime change additionally requires at least a 5% gain on one primary axis, deterministic output
equivalence, no CUDA/kernel alerts, and no lower clean-start VRAM reserve. Context allocation remains
131,072 unless a separately reported resource-envelope result supports a change.

No commit, push, BIOS flash, driver installation, or automatic production-binary replacement is
authorized by this packet.
