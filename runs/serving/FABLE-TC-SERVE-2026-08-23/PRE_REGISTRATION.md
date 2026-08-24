# Fable-TC serving transition pre-registration

Authorized on 2026-08-23 at 14:50:04 -03:00.

## Objective

Replace only the primary text endpoint on port 8080 with the already-qualified
`fable-tc-l1.0` deploy artifact. Keep `llm-embedding.service` on port 8081 intact.
This target is the authorial Fable + ThinkingCap task-arithmetic merge, not the
separate Fable-Fusion-711 artifact whose thinking-enabled termination gate failed.

## Frozen target

- Model: `/home/augus/models/merges/fable-tc-l1.0-Q4_K_M.gguf`
- Model SHA-256: `052c08ca13d75d8d88c9cc3f201d7bfa9167e2a1e69ad3e1e1f26ff73c1b390b`
- Formula: `Fable + 1.0 * (ThinkingCap - Qwen3.6-27B base)`
- Engine: `b10159 (068764d92)` from the qualified lifecycle fork
- Engine executable SHA-256: `5719c246ec3622ea1df3c3f498075879f12f1f70b969f8b591e87b3a1f3c8808`
- Runtime: all GPU layers, FlashAttention, context 8192, one default slot, Jinja,
  native MTP self-draft with maximum draft length 4, and metrics enabled.

## Baseline before transition

- Primary: Qwen3.8, engine `b10165 (71676e46c)`, context 131072, PID 199264.
- Primary health: HTTP 200; `NRestarts=0`.
- Embedding PID 198647, health HTTP 200, `NRestarts=0`.
- GPU: 22615 MiB used, 1708 MiB free.
- Existing Qwen drop-in remains installed as the rollback substrate.

## Acceptance gates

1. Versioned engine copy matches the frozen engine SHA and resolves its libraries
   from the versioned deployment directory.
2. Model bytes match the frozen model SHA.
3. `llm-inference.service` is active/running with `NRestarts=0`, the expected
   executable, model, context 8192, and MTP n4 flags.
4. Ports 8080 and 8081 both return HTTP 200 health.
5. OpenAI-compatible chat returns non-empty final content and fingerprint
   `b10159-068764d92`.
6. A deterministic generation completes and reports MTP draft activity.
7. An embedding request still returns 768 dimensions.
8. At least 3072 MiB of VRAM remains free after load.
9. No new service, CUDA, kernel, or Windows NVIDIA fault appears, and the same
   primary PID remains healthy for at least 70 seconds.

## Automatic rollback

On any failed gate, remove only
`/etc/systemd/system/llm-inference.service.d/serve-fable-tc.conf`, reload systemd,
restart `llm-inference.service`, and verify the previous Qwen fingerprint plus both
health endpoints. The Fable model and versioned engine copy may remain as inert
artifacts.
