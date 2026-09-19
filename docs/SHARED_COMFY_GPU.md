# Shared GPU for text and image generation

Requested on 2026-09-19: retain the model gateway on port 8080, move ComfyUI to
8188, and coordinate GPU use for **both browser jobs and TUI/API image jobs**.
This is serving integration, not a model-quality promotion. Publication remains
deferred until completion of the TUI experience plan.

## Execution contract

The managed text gateway and managed ComfyUI process share an exclusive Linux
file lock. A text request holds it through generation, streaming, counting or
profile observation. ComfyUI holds it through workflow execution and model
cleanup. The browser server remains available while a job waits.

An image job acquires the lock, asks the loopback gateway to release its resident
text backend, and starts only after confirmed backend exit. Before releasing the
lock, ComfyUI discards GPU caches and unloads its image models. The next text
request loads the requested qualified text model. Switching includes cold-load
latency; there is no fallback to a paid provider.

The gateway checks the image lease nonce before releasing its backend. The
nonce is local coordination evidence and does not appear in public status.
The lock does not expire while its owning process is alive. Failed ComfyUI
cleanup retains the lock and refuses further jobs until the managed process
is restarted. An unconfirmed text-backend stop also prevents a model switch.

The ComfyUI launcher runs upstream `main.py` with its normal arguments and
initialization. It installs the execution wrapper before the worker starts,
checks that custom-node initialization did not replace it, and verifies hashes
of the reviewed ComfyUI integration sources. A ComfyUI update therefore requires
review before this managed launcher starts. Arbitrary unmanaged GPU processes
and custom-node background GPU workers are outside this admission protocol.

## Delivery status

- [x] Gateway and cross-process GPU lease independently tested.
- [x] Browser/API executor, cancellation while waiting, cleanup failures and
      startup compatibility independently tested.
- [x] Image-model discovery and routed image-job API available to clients.
- [x] Existing workstation modifications preserved; release staged separately.
- [x] Migration admitted with an empty queue; 204 old history entries preserved.
- [x] Ports 8080 and 8188 prove gateway and ComfyUI coordinator identities.
- [x] Live API image and browser image jobs release Qwen and allow text to resume.
- [x] Gateway enabled for WSL startup; ComfyUI ordered after the gateway.
- [ ] Live cancellation/failure recovery and a real reboot remain unqualified.
- [ ] Dedicated TUI image controls and model-quality assessment remain pending.

## Access and installed assets

On the Tailscale network:

- ComfyUI browser: <http://100.107.245.30:8188>
- Model gateway: <http://100.107.245.30:8080/v1>

The reviewed ComfyUI version is 0.36.0, source revision
`7a0b5eede3f9721c8faab290689893f36edc6d66`. Its existing Python environment is
`/home/augus/.venvs/comfyui/bin/python`.

Installed assets include Illustrious XL, Juggernaut XL, Z-Image, Z-Image Turbo
and Krea, plus supporting encoders/VAEs. Browser workflows retain their chosen
assets. Z-Image and Krea require explicit compatible workflows; inventory is
not proof that every model works or meets a quality threshold.

## Image API

The configured gateway advertises the `comfyui` workflow engine and two SDXL
presets through `/v1/models`. Image cards are explicitly **unassessed** for model
quality. `/v1/images/models` also reports installed checkpoint, diffusion-model,
VAE and encoder assets without loading them.

Submit a preset job with `POST /v1/images/jobs`:

```json
{"model":"juggernaut-xl","prompt":"A blue teapot on a white table.","width":512,"height":512,"steps":20}
```

Use `illustrious-xl` for the other preset, or provide
`{"model":"comfyui","workflow":{...}}` with a ComfyUI API workflow to select
other installed models and nodes explicitly. A queued acknowledgement is not a
completed image. Preserve its returned `id`:

- `GET /v1/images/jobs/{id}` reports observed queue/history state and output links.
- `GET /v1/images/jobs/{id}/outputs/{index}` retrieves that job's recorded image.
- `POST /v1/images/jobs/{id}/cancel` with `{}` requests cancellation of that job
  only; it does not attest completed cancellation or GPU cleanup.

An uncertain submission reports its attempted ID so callers can inspect it
without blindly submitting the work twice. Missing history after a backend
restart is `UNKNOWN`. Existing completed-job history is privately backed up and
imported once during service migration; this is not a new durable ComfyUI job
database. Dedicated TUI image controls remain part of the broader TUI roadmap.

## Deployment and checks

The workstation uses a staged release with separate systemd overrides. Existing
repository modifications and model/workflow/output files remain in place.
The installer verifies source/configuration identities, requires an empty queue,
retains the previous override bytes and restores the original services if
readiness fails. It preserves the embedding service's process identity.

Windows mirrored WSL networking has a separate Hyper-V firewall. The scoped
[`allow_shared_gpu_tailscale.ps1`](../ops/qualified-model-fleet/allow_shared_gpu_tailscale.ps1)
rule permits TCP 8080 and 8188 from Tailscale addresses without changing the
global inbound policy. Existing port-proxy and Windows firewall rules are kept.

Fixture qualification: execution 330 passed 71 cases; execution 332 passed 17
final ComfyUI/history/migration cases. All wrappers verified cleanup. Execution
331 retains the Windows test-volume permission assertion failure; actual backup
permissions on the Linux workstation were observed as `0600` in execution 335.

Execution 333 migrated both services successfully using release
`2d718355cc48e552e549`. Execution 337 installed the scoped Hyper-V firewall rule;
340 verified its idempotent helper against the live rule. Execution 341 enabled
`llm-inference.service` and installed
[`comfyui-gateway.conf`](../ops/qualified-model-fleet/comfyui-gateway.conf) as
`/etc/systemd/system/comfyui.service.d/zzzz-shared-gpu-order.conf`.
This starts the gateway with ComfyUI and orders ComfyUI after it. Both original
services use `Restart=always`; the gateway listens without preloading a model.
All three service PIDs, including the inactive embedding service, were unchanged
by the startup configuration. A Windows/WSL boot itself was not exercised.

Live receipts:

| Execution | Result |
| --- | --- |
| 334 | Client network timeout before inference; retained failed run. |
| 338 | Juggernaut API job completed; text resumed. Browser test then failed before submission because its JavaScript client was not ready. |
| 339 | Browser client readiness wait corrected; Illustrious browser job completed and text resumed. |
| 340 | All 204 original history entries equal their private backup; 206 total including the two test jobs. Queue empty, Qwen healthy, GPU lease free. |
| 341 | Gateway enabled and ComfyUI startup dependency installed without restarting services. |

Each image run observed its own image lease, confirmed the text backend gone,
retrieved a valid 512×512 PNG, and received Qwen's response to a text request
submitted while the image job held the GPU. Waiting text took about 28 seconds,
including image execution and reloading Qwen. The tiny eight-step tests establish
serving behavior only: the Illustrious output did not match its teapot prompt
well. Both presets remain unassessed for quality. The browser test used its real
API client without modifying the user's workflow graph.

Evidence is retained outside the repositories at
`C:/projects/.reconciliation/evidence-store/reports/tui-experience-20260915/shared-gpu/`.
Service rollback receipts remain private under
`/home/augus/.local/state/tare-qualified-models/`. The migration backup is mode
`0600`; live verification confirmed its original history contents are unchanged.
No publication occurred.
