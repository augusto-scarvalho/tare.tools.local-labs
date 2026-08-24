# RWKV7 1.5B serving/quality qualification — pre-registration

## Trigger and identity

The publisher now explicitly states that the model weights use Apache-2.0 at
current Hub revision `bfb3a69a63e6681f729651c357f13ce0c774ea9c`. The current
Hub `model.safetensors` SHA-256 remains
`84ccbb857c84e00cefc48b233937ada79c411e491df25fb21aed23237f39a14f`,
exactly matching the local artifact whose recurrent mechanism already passed.

This removes the previous weight-license blocker without changing the weights.

## Frozen cheap gate

- Runtime: release-bundled official RWKV7 code, Transformers 5.15.0 overlay,
  BF16 weights, FP32 recurrent state, `backend=auto`.
- Workload: the existing immutable 48-item normal-question task file, SHA-256
  `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`.
- Greedy generation, official chat template, thinking disabled, task-specific
  maximum token limits.
- Primary quality floor: at least 36/48 overall, at least 50% in every category,
  48/48 non-empty, and at least 46/48 natural EOS.
- Serving floor: a supported OpenAI-compatible server must exist in the current
  isolated environment without mutating the shared SGLang environment.

The quality gate runs before any new serving dependency is installed. Failure
means `HOLD_QUALITY`; passing quality but lacking an existing server means
`HOLD_SERVING_RUNTIME`. Neither outcome promotes a 1.5B base model over Fable.

## Isolation and exit

Stop only `llm-inference.service`; preserve embeddings on 8081. After the run,
restore Fable on 8080, the locale proxy on 8082, and verify real requests. Fan
Control remains the fan-curve owner. No reboot or remote push.
