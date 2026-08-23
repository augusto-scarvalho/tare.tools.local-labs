# LAB-CLOSE-001 — mmap residual close-out replication

Frozen on 2026-08-22 before stopping the canonical text service.

## Question

Does disabling mmap cause the historically noted approximately 10.4% decode penalty
at the Qwen3.6-35B-A3B deployment placement, or was that residual confounded/noise?

## Fixed protocol

- Canonical `slop.cpp` checkout at commit
  `5e7f6271c06b9104862ab799278a1b7f1323a449`.
- Artifact: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, 22,663,387,424 bytes.
- Fixed placement: all GPU layers with six CPU MoE layers (`-ngl -1 -ncmoe 6`).
- FlashAttention on, depth 8,192, decode 64, prompt 0, six fresh-process
  repetitions per arm.
- Sole intended factor: `-mmp 1` versus `-mmp 0`.
- Arm order alternates by repetition. Each run gets a 25-second cooldown; temperature,
  SM clock, power draw, host available RAM, maximum RSS and major/minor page faults are
  retained. No GPU clock or voltage mutation is applied.
- The canonical `llm-inference.service` may be stopped for isolation. Port 8081 must
  remain healthy, and the text service must be restored and verified afterward.

## Validity and decision

- All 12 processes must exit zero and produce a parseable llama-bench JSON row.
- The model, placement, depth, build and power limit must remain invariant.
- Report median and bootstrap 95% CI for decode tok/s and paired relative deltas.
- `REAL` requires an absolute median paired effect above the standing 2.3% hardware
  noise floor and a bootstrap 95% CI excluding zero. Otherwise classify `NOISE`; if
  the old -10.4% effect does not reproduce, classify that historical residual
  `CONFOUNDED` as well.
- Keep mmap enabled unless no-mmap produces a reproducible operational advantage.

