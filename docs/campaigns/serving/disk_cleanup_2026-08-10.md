# Disk cleanup — 2026-08-10

Motivation: C: (the only large SSD) was down to ~400 GB free. Swept for models no longer used
by any open experiment, plus an unused WSL distro and Docker leftovers. Nothing with a pending
experiment was touched. **Net: ~378 GB returned to C: (~400 → 778.5 GB free).**

| Target | Before | After |
|---|---|---|
| C: free | ~400 GB | **778.5 GB** |
| WSL `Ubuntu-24.04` VHDX (project distro) | 657.3 GB | 384.5 GB |
| WSL `/home/augus/models` | 557 GB | 356 GB |
| Docker `docker_data.vhdx` | 30.2 GB | 1.9 GB |
| Store `Ubuntu` distro (unused) | 37 GB | removed |

## 1. WSL model deletions (`Ubuntu-24.04`, `/home/augus/models`) — ~207 GB models + ~10 GB caches

All from **closed** experiments (verified against STATUS/DEPLOY before deleting):

| Deleted | Size | Why safe |
|---|---|---|
| `merges/` fable-plain, fable-tc-l0.4, fable-tc-l0.7 (kept l1.0) | 48 GB | lambda sweep closed; l1.0 is the deploy artifact |
| `qwen36-35b-a3b/` q5, q6, q8 (kept q4) | 88 GB | quant/KV screen done (q4 KV lossless); deploy runs the `-mtp` q4 |
| `qwen3-vl-30b`, `qwen3-vl-8b` | 24 GB | VLM comparison subjects; track DONE (Gemma won) |
| `granite-4.0-h-small`, `ernie-4.5-21b`, `mistral-small-24b` | 46 GB | genpin/pinning triangulation closed (STATUS §B1) |
| `.cache/uv`, `.cache/ccache`, `.cache/pip` | ~10 GB | regenerable caches, not models |

Kept deliberately: `fp16/` (156 GB source weights — "kept for future recipes"), the two `-heretic`
abliteration variants, `gemma-4-26b-a4b`, `qwen36-27b-dense`, `qwen36-27b-mtp`, both ThinkingCap
27B (best coders on HumanEval — see `runs/quality-market/LEADERBOARD.md`).
Deploy/keep: `qwen36-35b-a3b-mtp`, `gpt-oss-20b`, `merges/fable-tc-l1.0`, `fable-fusion-711`,
`gemma-4-12b-vision`, `thinkingcap-lora`.

Registry (`src/model_lifecycle/models.py`) pruned to match — commit `52f7a73` (dated deletion
notes left per the repo's discard-comment convention; self-test passes: 10 entries, 10 archs).

## 2. Store `Ubuntu` distro removed (`wsl --unregister Ubuntu`) — 37 GB

A separate personal environment (home `aaaaa`), NOT the project. Was the WSL *default* distro;
`Ubuntu-24.04` set as default afterward. Contents were a Stable-Diffusion-webui setup:
- `models/Stable-diffusion` 18 GB (downloaded checkpoints, Jan 2024, re-downloadable)
- `venv` 5.3 GB + `repositories` + code (regenerable)
- `outputs` 28 MB (the only irreplaceable content) — backed up, then discarded on the owner's
  instruction ("não precisa fazer backup … deleta tudo").

## 3. Docker — full prune + compaction

`docker system df` showed **0 images, 0 containers, 0 build cache** (nothing matched the original
"unused images" ask). The only occupant was one orphaned named volume:
`hermes-ollama_ollama_models` (16.5 GB, Ollama models from a `hermes-ollama` compose project,
no container attached). Removed on the owner's instruction ("deleta … tudinho"):
`docker volume rm hermes-ollama_ollama_models`, then `docker system prune -a --volumes -f`.
Docker Desktop quit + `wsl --shutdown` + `docker_data.vhdx` compacted 30.2 → 1.9 GB.

## Method notes (for next time — see also the `wsl-disk-and-compaction` memory)

- WSL/Docker VHDXs are dynamic and **do NOT auto-shrink**; deleting files inside frees space only
  *inside* the VHDX. To return it to C:, compact the VHDX.
- `wsl --manage <distro> --set-sparse true` is **blocked** on this build ("potential data
  corruption", wants `--allow-unsafe`) — do NOT force it.
- Safe compaction: `wsl --shutdown` (stops ALL distros incl. docker-desktop; quit Docker Desktop
  first so it doesn't restart the backend), then an **elevated** `diskpart` script:
  `select vdisk file="<vhdx>"` / `attach vdisk readonly` / `compact vdisk` / `detach vdisk`.
  ~5 min per large file; runs long (background it).
- VHDX paths: project distro `C:\Users\augus\AppData\Local\wsl\{275f8505-…}\ext4.vhdx`
  (found via `HKCU:\…\Lxss\{guid}\BasePath`); Docker `…\AppData\Local\Docker\wsl\disk\docker_data.vhdx`.

## Follow-up state

- Distros remaining: `Ubuntu-24.04` (project, now default) + `docker-desktop`. Both stopped; start
  on next use. **Docker Desktop is quit** — reopen when Docker is needed.
- Session git: harness/benchmark-correction + registry-prune commits are LOCAL on `master` (no push).
