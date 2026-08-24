# RWKV7 1.5B serving/quality qualification — result

## Decision

`LICENSE_UNBLOCKED / HOLD_QUALITY`.

The previous license blocker is resolved by the publisher, but the model does
not meet the frozen serving-quality floor. No serving dependency was installed
and no persistent RWKV service was created.

## License and artifact binding

At current Hub revision `bfb3a69a63e6681f729651c357f13ce0c774ea9c`,
the official model card states that the weights use Apache-2.0 and the release
manifest records `license=apache-2.0`, locked-profile provenance, source revision
`ede85bf8ab2e59aff7d7ca909fbbc73317866d89`, and source hash
`32ef7b5bf4dc8bde843cf26dfad809a1f527e2e76a9e790e7d406e71bcd785da`.

The current Hub `model.safetensors` SHA-256 is
`84ccbb857c84e00cefc48b233937ada79c411e491df25fb21aed23237f39a14f`,
identical to the already qualified local file. No weight redownload was needed.

Primary publisher source:
<https://huggingface.co/RWKV/RWKV7-1.5B-20260805>.

## Frozen 48-item result

| Category | Pass |
|---|---:|
| Facts | 5/10 |
| Math/logic | 0/10 |
| Reading | 0/8 |
| Instruction | 1/8 |
| Calibration | 3/6 |
| Summary | 4/6 |
| **Total** | **13/48** |

Natural EOS was 47/48 and median wall time was 0.505 s after model load. The
model therefore runs quickly and usually terminates, but fails the frozen
36/48 overall floor and three category floors by wide margins.

The quality failure stopped the dependency ladder before serving expansion.
The current isolated environment has no vLLM installation; installing or
upgrading a server after a 13/48 gate would add maintenance cost without an
earned deployment role.

## Evidence

- Exact task file SHA-256:
  `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`.
- Result SHA-256:
  `b30dcf4a47770e999dcfe3db8417f2edde713644440241f57af303e32a75388c`.
- Runtime: release-bundled implementation, Transformers 5.15.0 overlay,
  Torch 2.11.0+cu130, BF16 weights, FP32 recurrent state, `backend=auto`.
- Prior mechanism receipt remains valid for the identical weight hash:
  recurrent continuation exact, state bytes constant, and isolation pass.

## Final operational state

Fable-TC was restored on 8080. Inference, embeddings, and locale proxy are
active; real responses returned `blockers-restored-ok` on 8080 and 8082, and
8081 returned a 768-dimensional embedding. Fan Control and MSI Afterburner are
active. No reboot, commit, or push was performed.
