# Repository path contract

**Effective:** 2026-08-21

This contract separates repository identity from historical filesystem names. New automation must use the
canonical paths below or an explicit environment override. The legacy names are compatibility aliases, not
the source of truth.

## Canonical paths

| Role | Canonical path |
|---|---|
| `tare.tools.local-labs` checkout | `C:\projects\tare.tools.local-labs` |
| `slop.cpp` primary worktree | `/home/augus/src/slop.cpp` |
| upstream/base comparison | `/home/augus/src/slop.cpp-base` |
| local lever worktree | `/home/augus/src/slop.cpp-local` |
| fork `main` worktree | `/home/augus/src/slop.cpp-main` |
| rebase worktree | `/home/augus/src/slop.cpp-rebase` |
| stacked-lever worktree | `/home/augus/src/slop.cpp-stack` |

The primary worktree owns the common Git directory. `git worktree list --porcelain` must report only the
canonical `slop.cpp*` paths.

## Compatibility aliases

The old paths remain available so historical receipts, external scripts, and operator muscle memory do not
break during migration:

- Windows junction: `C:\projects\local-model-lifecycle` -> `C:\projects\tare.tools.local-labs`;
- WSL symlinks: `/home/augus/src/llama.cpp*` -> the corresponding `/home/augus/src/slop.cpp*` worktree;
- historical `llama.cpp-master` maps to canonical `slop.cpp-main`.

Do not remove these aliases merely because tracked source no longer references them. First audit systemd,
scheduled tasks, shell history promoted into scripts, and any external orchestration outside this repository.

## Runtime overrides

Python serving code accepts these environment variables:

- `SLOP_CPP_SERVER_BIN` for the primary deploy binary;
- `SLOP_CPP_MAIN_SERVER_BIN` for profiles pinned to the `main` worktree.

Scripts should prefer repository-root discovery (`git rev-parse --show-toplevel`) and explicit arguments over
new machine-specific absolute paths. Existing scripts now use the canonical paths and continue to work through
the aliases during rollback.

## Migration receipt

The 2026-08-21 migration performed the following operations:

1. stopped only `llm-inference.service`; port 8080 went down and port 8081 remained healthy;
2. moved five linked worktrees with `git worktree move`;
3. moved the primary worktree and ran `git worktree repair` with the legacy alias absent, forcing canonical
   `.git` pointers;
4. created compatibility symlinks for all old WSL paths;
5. changed `/etc/systemd/system/llm-inference.service` to execute
   `/home/augus/src/slop.cpp/build/bin/llama-server` and reloaded systemd;
6. moved the Windows checkout and created the compatibility junction;
7. preserved untracked `FABLE_BUILD_COMMIT.txt`, `a4_spec_metrics_probe.py`, and the cancelled-soak directory.

### 2026-08-28 stale-probe reconciliation

The preserved root-level `a4_spec_metrics_probe.py` was later proven to be a
stale duplicate of the tracked canonical probe at
`tools/probes/a4_spec_metrics_probe.py`. The only logical difference was its
obsolete default binary path, `/home/augus/src/llama.cpp-master`; the canonical
copy uses `/home/augus/src/slop.cpp-main`. The stale WSL copy (SHA-256
`56ece83d830a9db7011de6a891593f3f67781eae360034b979722e9c05f6534d`) was
removed after comparison. No unique experiment logic was discarded.

The WSL primary checkout was then fast-forwarded from immutable experiment
commit `87a416bd7` to `origin/main` commit `34b3dac7c`. Historical reproduction
remains bound by tag `main-b10161-87a416b`; create a detached worktree from that
tag instead of leaving `main` behind. The existing ignored SLX-03 builds were
not rebuilt or moved. Their callable server hashes remained:

- audit R3: `0267affe48ff9d49a13dbe0891b33598ead1179edd5db85ecb3b2c86c7e1fd0b`;
- instrumented R1: `c00261d903f722214511f0f6b999de77ff98dedbf9b8da292501b7743bbaecac`.

Gateway port 8080 and embedding port 8081 stayed healthy throughout the
fast-forward. This reconciliation changed repository metadata and documentation
only; it did not change, rebuild or redeploy the qualified engine binaries.

The pre-migration Git worktree metadata, sanitized Git config, and systemd unit are copied under
`/home/augus/src/.slop-path-migration-backup-2026-08-21/`. The backup remote URL deliberately contains no embedded
credential.

## Validation and rollback

Before starting inference after any path change, require all of the following:

- all six canonical worktrees appear in `git worktree list --porcelain`;
- every worktree resolves `git status` and preserves its previous branch/detached state;
- `.git` pointer files reference `/home/augus/src/slop.cpp/.git/worktrees/...`;
- the server binary exists at the canonical path;
- `systemctl cat llm-inference.service` contains the canonical path;
- embeddings on port 8081 remain healthy;
- after an authorized start, port 8080 reaches `/health` successfully.

Rollback is directory-safe: stop the text service, restore the backed-up Git metadata and unit if needed,
reverse the exact worktree moves, run `git worktree repair`, then restore the compatibility aliases. Never
delete a worktree directory as a rollback mechanism.
