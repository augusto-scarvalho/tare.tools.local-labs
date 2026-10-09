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

## Training commands

### Renting the GPU: `gpu` (any agent or person)

One command in aaaaa's WSL wraps the lease below (`~/.local/bin/gpu`; from the other machine,
`ssh aaaaa "wsl -e bash -lc 'gpu status'"`):

```bash
gpu status                                   # holder, reason, how long, running jobs, who is waiting
gpu run --task cohort-26 -- ./cohort.sh      # supervised job; the task name learns its usual duration
gpu run --task flux-batch --kill-after 2h -- python batch.py
gpu hold --reason "debug flux workflow" 30m  # interactive shell holding the GPU; exit it to give it back
```

- `run` has no deadline. With `--task`, the median of that task's last ten successful runs (lease wait
  excluded, `~/.local/state/tare-gpu-training/runs.jsonl`) becomes its estimate, and `status` flags a run past
  1.5x of it as late; with no history it says "no estimate". Only `--kill-after` stops a job.
- `hold` needs a terminal (`ssh -t`) and a duration; five minutes before it ends it warns, then it stops the
  shell and its children and only then releases the lease. Agents without a terminal use `run`.
- A `gpu run` inside a hold or another job reuses the held lease and keeps its own task name, limits and log.
- Idle watch: every 30 s the supervisor reads board power; after ten minutes averaging under 80 W (2x the idle
  floor measured on this 3090) `status` shows the job as idle. A failed reading is "unknown", never idle. It
  only acts with `--idle-release 15m`, because 30 s samples miss short recurring bursts.
- While an image job (these, ComfyUI) holds or waits for the lease, the gateway answers text requests with
  `503 gpu_busy` within about 5 s instead of queueing them for the route timeout, and callers such as tare
  escalate elsewhere. Text behind text still queues. `qwen38-gsq` unloads after 10 idle minutes.
- How long text waits (contract `gpu-lease/1`, the kernel's `docs/GPU_LEASE_CONTRACT.md`): `/v1/fleet/status`
  gives `gpu_coordination.eta_seconds`, the time the holding `gpu run` job has left of its estimate plus the
  estimates of the image jobs waiting (each ticket carries its task's `expected_seconds`); null when one of
  them has no estimate (ComfyUI, a task without history, a run already late). Every `503` refusal carries
  `Retry-After` and `error.retry_after_seconds`: that estimate as it is, from 5 s up to 24 h (a 12-hour
  training run says 12 h), 60 s without one. `gpu status` shows each job's time left and when text gets the GPU. What tare does with it:
  tare.tools.os `docs/guides/SHARED_GPU.md`.
- Text keeps the lease for an idle grace after each request (`--text-idle-grace`, 20 s), so an agent's
  back-to-back calls are never split: an image or training job arriving mid-turn waits until text has been idle
  that long (`gpu status` shows the text owner with `idle_since`). Once a job has waited `--image-max-wait`
  (300 s) the grace ends after the running request, and the gateway logs a warning.

### Reusable entrypoint (installed 2026-09-20)

In aaaaa's WSL shell as `augus`, use the same command for any training script:

```bash
gpu-run /mnt/c/projects/imagen/train_yoshida_v2_lokr.sh
gpu-run ./another-training.sh --max_train_epochs 12
gpu-run -- python train.py --config experiment.toml
gpu-run --gpu-wait 32400 ./another-training.sh
```

`.sh` commands run through Bash, including scripts without executable permission.
Other commands use the current environment, so activate the intended Python
environment first when invoking Python/Accelerate directly. Arguments and exit
codes are preserved. The original `gpu-lease-run -- command ...` remains valid.
The installed executable is `~/.local/bin/gpu-run` if the shell lacks that PATH
entry. No changes to Accelerate or musubi-tuner are required.

For a Bash script that should also be protected when launched directly, add this
shared guard near the top, before environment activation or training:

```bash
source "$HOME/.local/share/tare-gpu-training/guard.sh" || exit $?
```

The guard re-executes the whole script under the supervisor. A protected command
inside an already protected job reuses its parent reservation after checking the
kernel lock, nonce, owner PID and process ancestry. An environment flag alone is
insufficient. This avoids recursive acquisition while keeping one owner through
training, samples, checkpoint conversion and descendant cleanup. Nested work is
part of the same supervised job; the lease is not a scheduler for its children.

All three existing `train_yoshida*.sh` entrypoints in `/mnt/c/projects/imagen`
now source this guard. Their remaining bytes, training arguments and output
paths were preserved, with originals and SHA-256 receipts backed up in
`~/.local/state/tare-gpu-training/entrypoint-1789948189850484140`.
New scripts need either `gpu-run script.sh` or the shared guard; arbitrary
unmanaged processes cannot be intercepted by this cooperative protocol.

Qualification: **55 Linux fixture tests passed**, including existing gateway,
ComfyUI, signals and process cleanup tests, plus nested reuse, inherited-proof
rejection, argument preservation and direct guarded-script execution. The
installed content-addressed release is `482d4c0b4f8d5cf7202b`. Gateway and ComfyUI
services do not need a restart for this training-entrypoint update.

A live harmless guarded script then acquired the real lease, unloaded resident
Qwen, verified the absent text backend from both its normal and nested command,
and exited with the lease released. It completed in 2.38 seconds and observed
23,487 MiB free afterward. No training or image inference was launched. The
backup directory contains `smoke.json`; portable deployment and fixture receipts
are under the external `gpu-training-entrypoint-20260920` evidence report.

SpecGraph reconciliation remains partial because the existing
`src/model_lifecycle/analysis/promotion.py` starts with a BOM rejected by its AST
extractor. That unrelated file was not changed by this delivery.

### Low-level launcher and lifetime

Run musubi-tuner under the same lease, **on aaaaa in WSL as `augus`**:

```bash
python3 tools/serving/gpu_lease_run.py -- \
  bash /mnt/c/projects/imagen/train_yoshida_v2_lokr.sh
```

The wrapper defaults to the serving lock at
`~/.local/state/tare-qualified-models/gpu.lock` and loopback gateway on 8080.
Use the Local Labs checkout containing this utility, or its installed
`~/.local/bin/gpu-lease-run` launcher. Running an ordinary training script without
the shared guard directly still bypasses coordination.

- `--gpu-wait 3600` is the default acquisition limit in seconds. Increase it
  explicitly for queued training, for example `--gpu-wait 32400` for nine hours.
  It must remain finite and positive. This deadline does **not** expire an acquired
  lease; no heartbeat is needed during a five-to-eight-hour training run.
- The wrapper acquires `kind=image`, authenticates its nonce with the gateway,
  and starts the command only after the gateway confirms its text backend exited.
  It holds the lease for the entire command, including sample generation and
  checkpoint conversion. It does not alter training arguments or outputs.
- SIGINT, SIGTERM and SIGHUP request shutdown of the child process group and
  detached descendants. `--stop-grace 30` allows cleanup before SIGKILL escalation.
  A Linux child subreaper tracks and reaps orphaned workers. The wrapper releases
  the lease only after observing that every descendant exited; elapsed grace
  alone is not evidence. Unconfirmed cleanup leaves the supervisor holding the
  lease and reporting the problem.
- Normal child exit codes are preserved; signal deaths use `128 + signal`.
  Acquisition timeout returns 124, coordination failure 125, and a missing
  executable 127. Diagnostics go to stderr; child input/output is inherited.
- Gateway requests and ComfyUI jobs do not preempt training. Their existing
  **600-second** acquisition limits remain in effect: they wait, then report a
  timeout if training is still running. Retrying later is explicit. A second
  training wrapper likewise waits up to its own limit and never starts concurrently.
  The next admitted text request reloads its model after training releases the GPU.

Process-exit evidence covers workers supervised by this wrapper. It is not a
GPU sandbox: unrelated processes and external services remain outside the lease.
Send signals to the wrapper, rather than force-killing it with SIGKILL. SIGKILL
cannot be handled; terminating the supervisor itself can drop its lock while
workers survive. Do not delete the lock file or forcibly stop the supervisor to
bypass an unconfirmed cleanup. Inspect and stop the remaining training workers.

### Training qualification, 2026-09-19

The installed launcher points to release `6c88e3f1cc88180314e1`. Registered
execution 415 passed **47 Linux tests** covering the wrapper, existing shared
lease, gateway admission and ComfyUI coordinator. Tests include SIGINT/SIGTERM,
detached workers, kill escalation, cleanup uncertainty, broken diagnostics,
competing trainers and unchanged lease ownership after the acquisition deadline.

Live execution 418 used the existing musubi-tuner LoKr configuration with a
two-step limit, disabled sampling and isolated output paths. It observed
**23,526 MiB free before training**, completed the first training step, and then
requested SIGTERM. A second trainer timed out with code 124 without starting;
a queued text request did not reload Qwen while training held the lease. The
wrapper reaped its process tree and exited 143 before releasing ownership;
Qwen reloaded and answered `READY`. The three serving service PIDs and restart
counts were unchanged. Existing training scripts and outputs were untouched.

Execution 417 preserves an earlier canary argument-expansion error that occurred
before any wrapper or training process started. Evidence and reviewed source
hashes are retained under the external `training-gpu/` report alongside the
shared-GPU evidence. This qualifies admission and cancellation, not a full
five-to-eight-hour run, sample-generation peaks or LoKr model quality.

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
global inbound policy. Windows firewall rules are kept. Mirrored networking
does not need a Windows port proxy for these WSL listeners.

On 2026-09-23, four stale `v4tov4` rules on `aaaaa` were removed: listen addresses
`0.0.0.0` and `100.107.245.30`, ports 8080 and 8188, each forwarding to
`127.0.0.1` on the same port. The wildcard rule caught its own forwarded
connections while the WSL service was absent. Windows accumulated roughly
24,000 established sockets and CI HTTP fixtures failed with `WinError 10055`.
Removing these four rules reduced established sockets to 99 without restarting
WSL, services or the machine. The original rules are backed up on that host at
`C:\Users\augus\portproxy-before-ci-repair-20260923.txt`.

When diagnosing recurrence, inspect `netsh interface portproxy show all`,
`Get-NetTCPConnection | Group-Object State` and the actual WSL listeners with
`wsl -d Ubuntu-24.04 -- ss -ltnp`. Do not restore same-port loopback proxies
while using mirrored networking. A missing application listener should be
diagnosed in its service; this network repair did not start any GPU workload.

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
