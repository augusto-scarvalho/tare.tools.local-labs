# Qualified local model fleet

The canonical text and vision surface is an OpenAI-compatible API on
`http://127.0.0.1:8080/v1`. The request's `model` field selects a role-qualified
artifact. The RTX 3090 keeps exactly one generation model resident: a route change
stops the previous private backend before loading the next one. Port 8081 remains the
independent embedding service.

The machine-readable source of truth is
[`config/qualified_model_fleet.json`](../config/qualified_model_fleet.json). A model
enters this file only with a `promoted` or bounded `qualified_role` decision, explicit
evidence, intended roles and claim limits. Fit-only, HOLD, rejected and proxy-only
artifacts are deliberately absent from the routable pool.

## Agent CLI

Run from the repository root:

```powershell
python tools/agents/modelctl.py list
python tools/agents/modelctl.py recommend coding
python tools/agents/modelctl.py show hauhaucs
python tools/agents/modelctl.py example gemma-vision
python tools/agents/modelctl.py status
python tools/agents/modelctl.py chat qwen38 "Responda apenas: OK"
python tools/agents/modelctl.py chat qwen36-moe "Responda apenas: OK" --no-thinking
```

Every discovery command supports stable model ids. `list`, `show`, `recommend` and
`status` also support `--json` for agent automation. Clients that already speak the
OpenAI API only need:

```text
OPENAI_BASE_URL=http://127.0.0.1:8080/v1
model=qwen38 | qwen36-moe | fable-tc | hauhaucs | gemma-vision | muse-vision
```

Aliases such as `coding`, `math`, `vision` and `vision-hard` are accepted, but durable
automation should record the canonical model id returned by `modelctl recommend`.

## Qualified routes and boundaries

| Route | Qualified role | Important boundary |
|---|---|---|
| `qwen38` | general, long context, agent/tools | Not the bounded 512-token math default; nine GSM8K truncations. |
| `qwen36-moe` | capped open-loop throughput and concurrency | Configuration-specific legacy deploy evidence, not a broad quality/default claim. |
| `fable-tc` | concise ordinary QA, Portuguese, math, writing | Frozen serving context is 8,192; not a thinking-mode route. |
| `hauhaucs` | coding, low-friction, agent/tools | Role-qualified, not promoted as the broad default. |
| `gemma-vision` | screenshots, OCR, UI debugging | Bounded synthetic visual-coding qualification, not universal VQA. |
| `muse-vision` | hard VQA and visual reasoning | Specialist role only; overall model remains HOLD and DFlash remains rejected. |

`muse-vision` uses its frozen architecture-specific build. The gateway, rather than
the homogeneous native llama-server router, is intentional: it permits different
qualified binaries while retaining a single public endpoint and a one-resident-model
VRAM invariant.

The superseded port-8000 model cards (`local_agent_fast`, `coder`, `reviewer`,
`planner`, `auditor`, `compressor`) are not silently imported. Their role labels and
speed measurements were not accompanied by local-labs role-qualification decisions;
they remain admission candidates until those evidence packets are reconciled.

## Operational lifecycle

The public gateway runs as the existing `llm-inference.service`. Its child server is
loopback-only on port 18080. The systemd drop-in is installed by:

```bash
bash /mnt/c/projects/tare.tools.local-labs/ops/qualified-model-fleet/activate_qualified_model_gateway.sh
```

Activation preloads `qwen38`, verifies the gateway identity, confirms all six routes
are advertised and asserts that the embedding PID on 8081 did not change. Rollback to
the prior single-model Qwen service is:

```bash
bash /mnt/c/projects/tare.tools.local-labs/ops/qualified-model-fleet/restore_qwen38_single_model.sh
```

The first request after changing models includes unload and cold-load latency. Requests
are serialized across a switch so the old and new large artifacts cannot overlap in
VRAM. A model-load failure returns HTTP 503 and never silently falls through to a
different model.

`lmctl mode check serve` recognizes the private 18080 child only when the public 8080
endpoint proves the `qualified-model-gateway` identity. An orphan process on 18080 still
fails the mode lock as drift.
