# fork-consolidation — audit & preservation scripts for the llama.cpp build trees

One-time-ish tooling written during the 2026-08-04 fork consolidation (see `../../FORK.md` and the
project memory). The box has five interlinked `llama.cpp-*` trees under `/home/augus/src/`; these scripts
map them and preserve stranded work **without deleting anything**. Run from the Bash tool with
`MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl.exe -d Ubuntu-24.04 -- bash -lc 'bash <path>'`
(the MSYS prefix stops Git Bash mangling `/home` and `/mnt/c` paths).

| script | what it does |
|---|---|
| `enum_builds.sh` | list every `llama.cpp-*` tree: branch, HEAD, bins, sizes, commits-ahead-of-pinned `720d7fa40`. |
| `check_recoverable.sh` | per tree: remotes + whether HEAD is pushed/recoverable (guards against deleting unpushed work). |
| `consolidate_audit.sh` | from the `lifecycle` fork, fetch all siblings and `git cherry` each line vs lifecycle to find what's NOT yet consolidated. |
| `preserve_branches.sh` | branch the fragile detached Turbo HEAD in the `stack` tree, then create local preservation branches in the fork. |

Outcome that day: the fork repo (`llama.cpp-master`) now holds every campaign line as a local branch
(`lifecycle`, `turbo-stack`, `prefetch-skip-pinned`, `fable5-prefetch-experts`); prefetch-skip-pinned was
folded onto `lifecycle` and re-blessed 3/3. Re-run `enum_builds.sh` + `consolidate_audit.sh` if the tree
set changes.
