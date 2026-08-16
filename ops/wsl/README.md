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

**Detached-job convention:** end your script with an `echo "=== DONE $(date -u) ==="` marker, then
watch completion with a `$`-free Monitor:
```
wsl -d Ubuntu-24.04 -- bash -lc 'while true; do grep -q "=== DONE" ~/.cache/wslx/<name>.log && { echo FINISHED; break; }; sleep 30; done'
```

Related memory: `wsl-from-bashtool-gotchas`, `wsl-disk-and-compaction`.
