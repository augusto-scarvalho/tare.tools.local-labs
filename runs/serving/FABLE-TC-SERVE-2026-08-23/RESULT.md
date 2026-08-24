# Fable-TC persistent serving result

## Outcome

**PASS / ACTIVE SERVE.** At 2026-08-23 14:51:20 -03:00, the primary text
endpoint on port 8080 was changed from Qwen3.8 to the qualified
`fable-tc-l1.0` deploy artifact. The embedding endpoint on port 8081 was not
restarted and remained healthy. Automatic rollback was not triggered.

## Active identity

- Model alias: `fable-tc-l1.0`
- Model: `/home/augus/models/merges/fable-tc-l1.0-Q4_K_M.gguf`
- Model SHA-256: `052c08ca13d75d8d88c9cc3f201d7bfa9167e2a1e69ad3e1e1f26ff73c1b390b`
- Derivation: `Fable + 1.0 * (ThinkingCap - Qwen3.6-27B base)`
- Engine: `b10159 (068764d92)`
- Engine SHA-256: `5719c246ec3622ea1df3c3f498075879f12f1f70b969f8b591e87b3a1f3c8808`
- Versioned engine root: `/home/augus/opt/slop.cpp/b10159-068764d92-fable-tc`
- Persistent drop-in: `/etc/systemd/system/llm-inference.service.d/serve-fable-tc.conf`

`LD_LIBRARY_PATH` is fixed to the versioned engine directory. Inspection showed
that `libllama`, `libggml`, `libggml-cuda`, `libmtmd`, and the server implementation
were all resolved from that directory rather than from a mutable source worktree.

## Active runtime

- Port: 8080
- Context: 8192 tokens
- GPU layers: 99 / full offload
- FlashAttention: on
- Slots: server default, one slot for this profile
- Native MTP self-draft: on, maximum draft length 4
- Jinja and metrics: enabled
- systemd policy: enabled persistent service with bounded restart guard

This is a deliberate context reduction from the previous Qwen3.8 131072-token
profile. The qualified Fable-TC deploy profile is an 8k profile.

## Validation evidence

- Text health: HTTP 200.
- Embedding health: HTTP 200.
- OpenAI-compatible model: `fable-tc-l1.0`.
- Fingerprint: `b10159-068764d92`.
- Deterministic chat returned final content `FABLE TC READY` with
  `finish_reason=stop`.
- Deterministic generation: 128 predicted tokens at 77.11 tok/s.
- MTP activity: 134 drafted tokens, 93 accepted.
- Embedding request: 768 dimensions.
- Live engine mapped by PID matched the frozen executable SHA-256.
- Text PID during canary: 199862; same PID after 83.1 seconds.
- `llm-inference.service`: active/running, result success, `NRestarts=0`.
- `llm-embedding.service`: active/running on preserved PID 198647,
  result success, `NRestarts=0`.
- Final GPU state: 21035 MiB used, 3288 MiB free, 31 C, 35.64 W.
- No matching service/CUDA error, WSL kernel GPU alert, or Windows
  NVIDIA/Display event appeared after the transition.
- SERVE/LAB lock is coherent and records Fable-TC as the active SERVE reason.

## Behavioral boundary

This is the concise, uncensored Fable-TC task-arithmetic merge qualified by A2,
not Fable-Fusion-711. The latter's thinking-enabled termination failure does not
describe this artifact. Fable-TC previously passed the concision, accuracy,
alignment-preservation, writing-quality, and native-MTP gates recorded in
`docs/campaigns/a2-ablation-merging/`.

## Rollback retained

The Qwen b10165 drop-in remains installed underneath this higher-priority drop-in.
To restore it, remove only
`/etc/systemd/system/llm-inference.service.d/serve-fable-tc.conf`, reload systemd,
restart `llm-inference.service`, and verify the Qwen fingerprint and both health
endpoints.

No commit or push was performed.
