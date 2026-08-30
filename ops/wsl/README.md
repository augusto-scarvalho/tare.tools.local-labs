# ops/wsl — running WSL jobs reliably from the Windows side

Driving WSL (Ubuntu-24.04) from the Windows **Git Bash / Claude Bash tool** has two traps that cost
real time. This directory turns the fix into a reusable tool (`wslx.sh`) instead of tribal knowledge.

## The two gotchas

### 1. `$VAR` gets expanded to EMPTY before it reaches WSL
When you run `wsl -d Ubuntu-24.04 -- bash -lc '...$VAR...'` from the Windows side, an outer layer
expands `$VAR` / `$(...)` **before** WSL sees it — *even inside single quotes* when quote nesting is
imperfect. The variable arrives empty, so a redirect like `> "$LOG"` becomes `> ""` and dies with:
```
bash: line N: : No such file or directory        # $LOG was empty
bash: line 1: /logs/download.log: No such file   # $D was empty -> "/logs/..."
```
It fails **silently and identically** in foreground and background. Confirmed by `cat -A` on a staged
heredoc showing `LOG="/logs/download.log"` and `> ""` where the source had `$D`/`$LOG`.

**Fix (the only robust one): keep every `$` in a FILE on disk.** WSL bash reads the file directly, so
its `$VAR`s expand in WSL where they belong. Never put `$` in the inline tool command. Strip CR when
copying a Windows-authored file in: `tr -d '\r' < /mnt/c/.../x.sh > ~/x.sh`. Verify with `cat -A` —
real paths, lines end `$` (LF) not `^M$` (CRLF).

### 2. `huggingface-cli` no longer exists
huggingface_hub ≥ 1.24 removed it. The CLI is now **`hf`** at `~/.local/bin/hf`; same flags:
`hf download <repo> --include "*Q4_K_M*" --local-dir <dir>`. Reference it by absolute path
(`$HOME/.local/bin/hf`) in scripts so it works under non-login `setsid`/detached runs too.

## The tool: `wslx.sh`

Runs a bash script **inside WSL** the safe way: stages it (CR-stripped), runs it under a **login
shell** (so `~/.local/bin` is on PATH), logs combined output, optionally detaches. **Your invocation
contains zero `$`** — only literal path args — so nothing can be mangled; all variable logic lives in
the script file.

```bash
# run a repo gate inside WSL, see its output inline:
bash ops/wsl/wslx.sh ops/qwen38-bringup/mtp_tensor_check.sh

# pass env/args to the script:
bash ops/wsl/wslx.sh ops/qwen38-bringup/kv_recall_sweep.sh -- DEPTH=131072

# long job (download/train): detach and return immediately, then watch the log:
bash ops/wsl/wslx.sh /c/tmp/pull_models.sh --detached
wsl -d Ubuntu-24.04 -- bash -lc 'tail -f ~/.cache/wslx/pull_models.log'
```
Flags: `--detached`, `--log <wsl_path>`, `--distro <name>` (default `Ubuntu-24.04`), `-- ARGS...`.
Accepts Windows (`C:\..`), Git Bash (`/c/..`), `/mnt/c/..`, or WSL (`/home/..`) script paths.

**Long detached jobs — use the harness, not `--detached`.** `wslx --detached` uses `setsid` inside a
transient `wsl` call; that call exits immediately, and there is a race that can kill the job before it
writes anything (observed: empty log, no process). `--detached` now adds `disown` + a 3s settle to
mitigate, but the ROBUST pattern for a multi-minute job is to stage once, then launch by LITERAL PATH
(no `$`) through the harness's own background runner, which keeps `wsl.exe` alive for the whole run and
notifies on completion:
```bash
bash ops/wsl/wslx.sh path/to/job.sh --detached   # only to STAGE (ignore the detached launch)
# then run the staged copy via Bash run_in_background (literal path, zero $):
wsl -d Ubuntu-24.04 -- bash -lc 'bash /home/augus/.cache/wslx/job.sh > /home/augus/.cache/wslx/job.log 2>&1'
```

**Short detached convention:** end your script with an `echo "=== DONE $(date -u) ==="` marker, then
watch completion with a `$`-free Monitor:
```
wsl -d Ubuntu-24.04 -- bash -lc 'while true; do grep -q "=== DONE" ~/.cache/wslx/<name>.log && { echo FINISHED; break; }; sleep 30; done'
```

Related memory: `wsl-from-bashtool-gotchas`, `wsl-disk-and-compaction`.

## CPU policy: 24 for official serving, 20 for experiments

The WSL VM exposes all 24 host logical processors. The systemd manager defaults
new WSL work to CPUs `0-19`; this limit is inherited by experiment processes and
their subprocesses, including WSL calls opened by Windows-side orchestrators.
`llm-inference.service` alone overrides the default with CPUs `0-23`, so the
qualified-model gateway and its private backend may use the full machine.

Canonical files:

- `cpu-policy/90-local-labs-experiment-cpu.conf` installs to
  `/etc/systemd/system.conf.d/90-local-labs-experiment-cpu.conf`;
- `cpu-policy/llm-inference-cpu.conf` installs to
  `/etc/systemd/system/llm-inference.service.d/cpu-affinity.conf`;
- `cpu-policy/local-labs-wsl-interop-affinity.service` limits PID 2 (`/init`),
  whose descendants are commands opened directly by `wsl.exe`;
- `%USERPROFILE%/.wslconfig` contains `processors=24`.

Install the tracked policy files as root, enable the interop unit, then perform
the guarded WSL restart described below:

```powershell
wsl -d Ubuntu-24.04 -u root -- install -D -m 0644 /mnt/c/projects/tare.tools.local-labs/ops/wsl/cpu-policy/90-local-labs-experiment-cpu.conf /etc/systemd/system.conf.d/90-local-labs-experiment-cpu.conf
wsl -d Ubuntu-24.04 -u root -- install -D -m 0644 /mnt/c/projects/tare.tools.local-labs/ops/wsl/cpu-policy/llm-inference-cpu.conf /etc/systemd/system/llm-inference.service.d/cpu-affinity.conf
wsl -d Ubuntu-24.04 -u root -- install -D -m 0644 /mnt/c/projects/tare.tools.local-labs/ops/wsl/cpu-policy/local-labs-wsl-interop-affinity.service /etc/systemd/system/local-labs-wsl-interop-affinity.service
wsl -d Ubuntu-24.04 -u root -- systemctl enable local-labs-wsl-interop-affinity.service
```

The manager policy deliberately covers all ordinary WSL work rather than
wrapping only the Windows watcher PID. The real experiment commands are Windows
Python orchestrators that create independent WSL subprocesses, so parent-only
affinity would not constrain the measured workload.

After changing any of these files, verify the effective kernel and process
affinities:

```powershell
python tools/analysis/wsl_cpu_policy.py --json
```

Expected result: `kernel_online_vcpus=24`, `experiment_vcpus=20`, and
`serving_vcpus=24`. Treat a mismatch as a preflight failure for CPU-sensitive
experiments. Changing the policy requires a full `wsl --shutdown`; first ensure
no experiment is active, stop `llm-inference.service` through systemd, and
restore the gateway, embedding service, runners and `WSL-KeepAlive` afterward.
