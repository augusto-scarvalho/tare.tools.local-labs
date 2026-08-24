# Qwen3.8 HauhauCS math/general-reasoning expansion — result

## Bottom line

`MATERIAL_MATH_LOSS` for HauhauCS under the frozen 512-token operational
contract, driven by termination/format adherence rather than a broad arithmetic
collapse.

Fable-TC scored 195/200, HauhauCS 191/200, and vanilla Qwen3.8 188/200.
HauhauCS was three tasks better than vanilla but four behind Fable. Its 192/200
format adherence and 8/200 truncations fail the pre-registered minimums of 98%
format and at most 1% truncation.

## Primary result

| Arm | Strict | Wilson 95% | Format | Truncated | Median wall | Total wall |
|---|---:|---:|---:|---:|---:|---:|
| Fable-TC | 195/200 (97.5%) | 94.28–98.93% | 200/200 | 0/200 | 1.304 s | 282.8 s |
| HauhauCS aggressive | 191/200 (95.5%) | 91.67–97.61% | 192/200 | 8/200 | 3.735 s | 784.2 s |
| Vanilla Qwen3.8 | 188/200 (94.0%) | 89.81–96.53% | 191/200 | 9/200 | 3.844 s | 804.4 s |

The same 200 task IDs, order, seed, prompt, 512-token cap, and strict scorer
were used for all arms. Each manifest reports `server_identity_stable=true` and
200 rows.

## Paired outcomes

| Pair | Both pass | First only | Second only | Both fail | Exact two-sided p |
|---|---:|---:|---:|---:|---:|
| HauhauCS vs Fable | 188 | 3 | 7 | 2 | 0.34375 |
| HauhauCS vs vanilla | 184 | 7 | 4 | 5 | 0.54883 |
| Vanilla vs Fable | 186 | 2 | 9 | 3 | 0.06543 |

The exact paired tests are descriptive and do not override the frozen practical
thresholds. They do show that this 200-item panel is not strong evidence for a
large latent arithmetic-quality separation. The operational defect is clearer:
Fable finished every answer, while HauhauCS and vanilla frequently consumed the
entire 512-token budget before emitting the required final line.

## Failure mechanisms

- Fable: five completed but numerically wrong answers; zero truncations.
- HauhauCS: eight truncations plus one completed wrong answer (`gsm8k/403`).
- Vanilla: nine truncations plus three completed wrong answers (`gsm8k/1019`,
  `gsm8k/403`, and `gsm8k/649`).
- HauhauCS and vanilla both failed `331`, `403`, `640`, `649`, and `539`, but
  their remaining failures differed.

This extends the earlier 48-item normal-question panel. Together the evidence
supports Fable as the broad short-request default; HauhauCS remains attractive
for coding and low-friction requests, but not as a drop-in default for bounded
math automation without a larger response budget or a separately frozen
termination mitigation.

## Evidence identity

- Dataset file SHA-256:
  `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`
- Runner source SHA-256:
  `48d8923c9144555ced364f6df359269e6aaa1ec48fa42831a1adc8510d998028`
- Fable rows SHA-256:
  `3195115daf5fabc2637ccc845ab8318dd6e383e1bbe7b3b24f40a2f1d6a39dbb`
- HauhauCS rows SHA-256:
  `306368b7521da13a0ddf375cbd25a2e1bf903fd974c27b30211bf586a10c0efc`
- Vanilla rows SHA-256:
  `fc9e181c4d4dc2981a843fcafdae68ffba7a4ff64a727adbe75f177653053101`
- Result-bearing repository HEAD recorded by all manifests:
  `ed1fb68b2b7c73ffa311b14e0c7fceb32a7a62f5`.

## Final operational state

Verified after the experiment:

- inference, embedding, and locale-proxy services active;
- Fable-TC restored on 8080, build b10159 `068764d92`;
- real 8080 response `math-restored-ok`;
- real 8081 embedding with 768 dimensions;
- real 8082 proxied response `math-restored-ok`;
- experimental model drop-ins absent.

No reboot, commit, or push was performed.
