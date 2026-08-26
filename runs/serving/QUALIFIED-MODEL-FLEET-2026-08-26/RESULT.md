# Qualified model fleet serving result

Date: 2026-08-26
Decision: **ACTIVE / FIVE ROLE-QUALIFIED ROUTES / ONE RESIDENT MODEL**

## Outcome

The canonical OpenAI-compatible endpoint on port 8080 now advertises and routes six
role-qualified local artifacts. A Python gateway owns the public port and starts one
loopback-only `llama-server` child on port 18080. A route change terminates the old
child before loading the new child; the RTX 3090 therefore never intentionally holds
two generation models at once. The independent embedding service on 8081 was not
restarted.

The legacy port-8000 local-agent gateway was stopped after the replacement passed. It
had a conflicting design in which its private backend also attempted to own port 8080.
Its Windows Startup launcher was preserved but renamed from
`LocalAgentGateway.cmd` to `LocalAgentGateway.cmd.disabled`, preventing the conflict
from returning at the next login without deleting the rollback artifact.

## Frozen routable set

| Route | Decision admitted | Qualified role | Boundary |
|---|---|---|---|
| `qwen38` | `qualified_role` | general, long context, agent/tools | Not the bounded 512-token math default. |
| `qwen36-moe` | `qualified_role` | capped open-loop throughput and concurrency | Exact topology evidence, not a current-default or broad quality claim. |
| `fable-tc` | `promoted` | concise QA, Portuguese, math, writing | Qualified serving context is 8,192. |
| `hauhaucs` | `qualified_role` | coding, low-friction, agent/tools | Not promoted as broad default. |
| `gemma-vision` | `qualified_role` | screenshots, OCR, UI debugging | Bounded synthetic visual-coding evidence. |
| `muse-vision` | `qualified_role` | hard VQA, visual reasoning | Specialist only; overall model remains HOLD and DFlash rejected. |

HOLD, rejected, fit-only and proxy-only artifacts are rejected by registry validation.
The complete artifact digests, runtime binaries, arguments, evidence links and claim
limits are frozen in `config/qualified_model_fleet.json`.

## Live route canaries

Each request used the public `/v1/chat/completions` endpoint and selected the route in
the standard JSON `model` field. The reported fingerprint came from the actual private
backend response.

| Sequence | Route | Cold switch | Exact final content | Fingerprint |
|---:|---|---:|---|---|
| 1 | `qwen38` preload | 10.55 s | `QWEN_ROUTE_OK` | `b10165-71676e46c` |
| 2 | `fable-tc` | 13.54 s | `FABLE_ROUTE_OK` | `b10159-068764d92` |
| 3 | `hauhaucs` | 15.04 s | `HAUHAUCS_ROUTE_OK` | `b10165-71676e46c` |
| 4 | `gemma-vision` | 7.02 s | `GEMMA_VISION_ROUTE_OK` | `b10165-71676e46c` |
| 5 | `muse-vision` | 12.03 s | `MUSE_VISION_ROUTE_OK` | `b10573-d775b8967` |
| 6 | `qwen36-moe` | 18.05 s | `QWEN36_MOE_ROUTE_OK` with thinking disabled | `b10159-068764d92` |
| 7 | `qwen38` final restore | 5.03 s | `QWEN_ROUTE_RESTORED` | `b10165-71676e46c` |

The first Qwen3.6 canary with thinking enabled exhausted its 256-token cap inside
`reasoning_content` and emitted empty final content. This reproduces the registered
cap-bound reasoning limitation rather than a routing failure: identity and health were
correct. Repeating with `chat_template_kwargs.enable_thinking=false` returned the exact
canary in ten tokens. The CLI exposes this control as `modelctl chat --no-thinking`.

These are transport, identity and load/unload canaries. The vision routes' scientific
image-task claims continue to come from their pre-existing evidence packets; this
serving change did not rerun a new image-quality panel.

## Final state

- public gateway: `0.0.0.0:8080`, systemd main PID 46324 at final inspection;
- resident route: `qwen38`, private backend PID 46426 on `127.0.0.1:18080`;
- embedding: PID 1585 on `0.0.0.0:8081`, unchanged across activation;
- gateway status: six available models, `backend_healthy=true`,
  `max_resident_models=1`, `last_error=null`;
- SERVE/LAB lock: coherent SERVE, private backend normalized to its proven public
  gateway owner; an orphan 18080 backend remains fail-closed drift;
- legacy gateway port 8000: stopped;
- final resident canary: exact `QWEN_ROUTE_RESTORED`.

PIDs are operational observations, not durable identities. Artifact SHA-256 values and
qualified runtime fingerprints are the durable bindings.

## Validation

- focused fleet and mode-lock suite: 17/17 passing after gateway normalization;
- repository suite: 148/148 passing;
- backlog pipeline gate: PASS;
- `git diff --check`: PASS;
- activation retained a one-file rollback: remove
  `/etc/systemd/system/llm-inference.service.d/zz-qualified-model-gateway.conf`,
  reload systemd and restart `llm-inference.service` to recover the prior Qwen
  single-model route.
